
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from sem_kge.model.embedder import MultipleEmbedder, GrowingMultipleEmbedder
from sem_kge import TypedDataset

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from kge.model.transe import TransEScorer

class TransT(KgeModel):
    """ """

    def __init__(
        self, 
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        # convert dataset
        dataset = TypedDataset.create(dataset)

        # whether or not to use reciprocal relations
        self.recrel = config.get("transt.reciprocal_relations")

        R = dataset.num_relations()
        self.num_relations = R

        # double nr of relations before constructor, so embedders have correct size.
        if self.recrel:
            dataset._num_relations = 2 * dataset.num_relations()

        super().__init__(
            config=config,
            dataset=dataset,
            scorer=TransEScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
    
        # Create structure with each entity's type set.
        entity_types = dataset.entity_types()
        N = dataset.num_entities()
        T = dataset.num_types()

        config.log("Creating typeset binary vectors")
        device = self.config.get("job.device")
        self.device=device

        # We represent each set as a binary vector.
        # A 'True' value means that the type corresponding
        #  to the column is in the set.
        types_tensor = torch.zeros(
                (N,T), 
                dtype=torch.bool, 
                device=device,
                requires_grad=False
            )
        for idx, typeset_str in enumerate(entity_types):
            if typeset_str is None:
                continue
            typelist = list(int(t) for t in typeset_str.split(','))
            for t in typelist:
                types_tensor[idx, t] = True


        config.log("Creating common typeset vectors")
        # Create structure with each relation's common type set. 
        type_head_counts = torch.zeros((R,T), device=device, requires_grad=False)
        type_tail_counts = torch.zeros((R,T), device=device, requires_grad=False)
        type_totals      = torch.zeros((R,1), device=device, requires_grad=False)

        triples = dataset.split('train').long()
        for triple in triples:
            h,r,t = triple
            type_head_counts[r] += types_tensor[h]
            type_tail_counts[r] += types_tensor[t]
            type_totals[r] += 1

        RHO = self.get_option("rho")

        self.rel_common_head = type_head_counts / type_totals > RHO
        self.rel_common_tail = type_tail_counts / type_totals > RHO

        self.types_tensor = types_tensor

        def logit(lmbda):
            return torch.log(lmbda / (1-lmbda))

        lambda_head = self.get_option("lambda_head")
        lambda_relation = self.get_option("lambda_relation")
        lambda_tail = self.get_option("lambda_tail")

        lambda_head = torch.full((1,), lambda_head, device=device)
        lambda_relation = torch.full((1,), lambda_relation, device=device)
        lambda_tail = torch.full((1,), lambda_tail, device=device)

        learn_lambda = self.get_option("learn_lambda")

        if learn_lambda:
            # TODO add log statements informing of the change
            if lambda_head == 0:
                lambda_head += torch.finfo().eps
                config.log(f"To allow positive gradients lambda_head has been set to {lambda_head.item()}")
            if lambda_relation == 0:
                lambda_relation += torch.finfo().eps
                config.log(f"To allow positive gradients lambda_relation has been set to {lambda_relation.item()}")
            if lambda_tail == 0:
                lambda_tail += torch.finfo().eps
                config.log(f"To allow positive gradients lambda_tail has been set to {lambda_tail.item()}")

            if lambda_head == 1:
                lambda_head -= torch.finfo().eps
                config.log(f"To allow positive gradients lambda_head has been set to {lambda_head.item()}")
            if lambda_relation == 1:
                lambda_relation -= torch.finfo().eps
                config.log(f"To allow positive gradients lambda_relation has been set to {lambda_relation.item()}")
            if lambda_tail == 1:
                lambda_tail -= torch.finfo().eps
                config.log(f"To allow positive gradients lambda_tail has been set to {lambda_tail.item()}")

        self._loglambda_head = Parameter(logit(lambda_head), requires_grad=learn_lambda)
        self._loglambda_relation = Parameter(logit(lambda_relation), requires_grad=learn_lambda)
        self._loglambda_tail = Parameter(logit(lambda_tail), requires_grad=learn_lambda)

        if not self.get_s_embedder() is self.get_o_embedder():
            raise NotImplementedError("TransT currently does not support \
                the use of different embedders for subjects and objects.")

        # TODO Rather than checking for a specific class, create interface
        # for all embedders that might need type information

        # initialize the GrowingMultipleEmbedder
        if isinstance(self.get_s_embedder(), GrowingMultipleEmbedder):
            self.get_s_embedder().initialize_semantics(types_tensor)
 
        # initialize the distribution weights
        if isinstance(self.get_s_embedder(), MultipleEmbedder):
            nr_embeddings = self.get_s_embedder().get_nr_embeddings()
            M = self.get_s_embedder().nr_embeds
            idxs = torch.arange(M, device=device).unsqueeze(0).expand(N,-1)
            weights = (idxs < nr_embeddings.unsqueeze(1).expand(-1,M)).float()
            weights /= nr_embeddings.unsqueeze(1)
        else:
            weights = torch.ones((N,1), device=device) 
        self.weights = torch.nn.Parameter(weights)

    def _s(self, T_1, T_2):
        intersection_size = (T_1 & T_2).sum(dim=1).float()
        other_size = T_1.sum(dim=1).float()
        
        result = intersection_size / other_size
        result += 1e-2 #18 # prevent log(0)
       
        # If |T_1| = 0, than the numerator and denominator are both 0
        #   therefore the outcome should be ... TODO 
        result[other_size.expand_as(result) == 0] = 0.5
        return result

    def _log_prior(self, T_h, T_r_head, T_r_tail, T_t, corrupted):
        result1 = 0
        result2 = 0
        
        lambda_head = self._loglambda_head.sigmoid()
        lambda_relation = self._loglambda_relation.sigmoid()
        lambda_tail = self._loglambda_tail.sigmoid()

        lambda_head = 1 if lambda_head.isinf().all() else lambda_head 
        lambda_relation = 1 if lambda_relation.isinf() else lambda_relation 
        lambda_tail = 1 if lambda_tail.isinf() else lambda_tail 

        if corrupted == "s":
            if lambda_head > 0:
                result1 = lambda_head * self._s(T_r_head, T_h).log() 
            if lambda_relation > 0:
                result2 = lambda_relation * self._s(T_t, T_h).log()
        elif corrupted == "p":
            if lambda_head > 0:
                result1 = lambda_head * self._s(T_r_head, T_h).log() 
            if lambda_tail > 0:
                result2 = lambda_tail * self._s(T_r_tail, T_t).log()
        elif corrupted == "o":
            if lambda_tail > 0:
                result1 = lambda_tail * self._s(T_r_tail, T_t).log() 
            if lambda_relation > 0:
                result2 = lambda_relation * self._s(T_h, T_t).log()
        return result1 + result2

    def _batch_log_prior(self, s_typ, p_typ, o_typ, corrupted: str, combine: str):
        BS = 32 #TODO: make this configurable
        p_typ_h, p_typ_t = p_typ
        if combine == "sp_":
            s_typ, p_typ_h, p_typ_t = s_typ.unsqueeze(2), p_typ_h.unsqueeze(2), p_typ_t.unsqueeze(2)
            logprior = [];  N = o_typ.shape[0]
            for i in range(0, N, BS):
                j = min(i + BS, N)
                temp = o_typ[i:j,:].T.unsqueeze(0)
                logprior.append(self._log_prior(s_typ, p_typ_h, p_typ_t, temp, corrupted))
            logprior = torch.cat(logprior, dim=1)
        elif combine == "_po":
            p_typ_h, p_typ_t, o_typ = p_typ_h.unsqueeze(2), p_typ_t.unsqueeze(2), o_typ.unsqueeze(2)
            logprior = [];  N = s_typ.shape[0]
            for i in range(0, N, BS):
                j = min(i + BS, N)
                temp = s_typ[i:j,:].T.unsqueeze(0)
                logprior.append(self._log_prior(temp, p_typ_h, p_typ_t, o_typ, corrupted))
            logprior = torch.cat(logprior, dim=1)
        elif combine == "s_o":
            raise NotImplementedError()

        return logprior

    def prepare_job(self, job, **kwargs):
        super().prepare_job(job, **kwargs)

        if not self.get_option("learn_lambda"):
            return

        def trace_lambda(trace_job):
            trace_job.current_trace["batch"]["loglambda_head"] = self._loglambda_head.item()
            trace_job.current_trace["batch"]["loglambda_relation"] = self._loglambda_relation.item()
            trace_job.current_trace["batch"]["loglambda_tail"] = self._loglambda_tail.item()

        from kge.job import TrainingOrEvaluationJob
        if isinstance(job, TrainingOrEvaluationJob):
            job.pre_batch_hooks.append(trace_lambda)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        """
        """
        B = s.shape[0]
        device = self.device

        s, p, o = s.long(), p.long(), o.long()
        s_emb = self.get_s_embedder().embed(s)
        if not self.recrel or direction == "o":
            p_emb = self.get_p_embedder().embed(p)
        elif self.recrel and direction == "s":
            p_emb = self.get_p_embedder().embed(p + self.num_relations)
        o_emb = self.get_o_embedder().embed(o)
      
        # expand dimensions in case 
        for emb in (s_emb, o_emb):
            if emb.ndim == 2:
                emb.unsqueeze_(2)

        # get typesets
        s_t = self.types_tensor[s]
        r_t = self.rel_common_head[p], self.rel_common_tail[p]
        o_t = self.types_tensor[o]

        logprior = self._log_prior(s_t, r_t[0], r_t[1], o_t, direction)

        M = 1 # by default only 1 embedding
        w_s, w_o = self.weights[s], self.weights[o]                 # B x M
        if isinstance(self.get_s_embedder(), MultipleEmbedder):
            nr_embeddings = self.get_s_embedder().get_nr_embeddings()
            nr_s = nr_embeddings[s].unsqueeze(1).expand(-1,M)
            nr_o = nr_embeddings[o].unsqueeze(1).expand(-1,M)

            M = self.get_s_embedder().nr_embeds
            idxs = torch.arange(M, device=device).unsqueeze(0).expand(B,-1)

            w_s[idxs >= nr_s] = float('-inf')
            w_o[idxs >= nr_o] = float('-inf')
        w_s, w_o = F.softmax(w_s, dim=1), F.softmax(w_o, dim=1)     # B x M
        
        part_loglikelihoods = torch.full((B,M,M), float('-inf'), device=device)
        for i in range(s_emb.shape[2]):
            for j in range(o_emb.shape[2]):

                # check if whole batch inactive and skip if so
                if (w_s[:,i] == 0).all() or (w_o[:,j] == 0).all():
                    continue
               
                s_emb_i = s_emb[:,:,i]
                o_emb_j = o_emb[:,:,j]

                # weights of inactive vectors will be 0, so won't be summed
                loglikelihood = self._scorer.score_emb(
                    s_emb_i, p_emb, o_emb_j, combine="spo"
                ).squeeze()
                part_loglikelihoods[:,i,j] = loglikelihood
        w_s, w_o = w_s.unsqueeze(2), w_o.unsqueeze(1)

        # perform modified logsumexp trick
        x = part_loglikelihoods
        c,_ = x.max(dim=1, keepdims=True)
        c,_ = c.max(dim=2, keepdims=True)
        weighted_exp = w_s * w_o * (x - c).exp()
        loglikelihood = c.squeeze() + weighted_exp.sum(dim=1).sum(dim=1).log()
    
        if isinstance(self.get_s_embedder(), GrowingMultipleEmbedder):
            self.get_s_embedder().update(
                (s,p,o),(s_emb,p_emb,o_emb),(s_t,r_t,o_t),
                loglikelihood, self._s)
            
        logposterior = loglikelihood + logprior
        return logposterior.view(-1)


    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        raise NotImplementedError() 

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        raise NotImplementedError()

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        raise NotImplementedError()

    def score_sp_po(
        self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        B = s.shape[0]
        device = self.device

        s, p, o = s.long(), p.long(), o.long()
        s_e = self.get_s_embedder().embed(s)
        p_e = self.get_p_embedder().embed(p)
        if self.recrel:
           p_i = self.get_p_embedder().embed(p + self.num_relations)
        o_e = self.get_o_embedder().embed(o)

        # get typesets
        s_t = self.types_tensor[s]
        p_t = self.rel_common_head[p], self.rel_common_tail[p]
        o_t = self.types_tensor[o]

        if entity_subset is not None:
            entity_subset = entity_subset.long()
            all_entity_types = self.types_tensor[entity_subset]
            all_entities = self.get_s_embedder().embed(entity_subset)
            w_all = self.weights[entity_subset]
        else:
            all_entity_types = self.types_tensor
            all_entities = self.get_s_embedder().embed_all()
            w_all = self.weights
                       
        for emb in (s_e, o_e, all_entities):
            if emb.ndim == 2:
                emb.unsqueeze_(2)

        sp_logprior = self._batch_log_prior(
                s_t, p_t, all_entity_types, "o", "sp_")
        po_logprior = self._batch_log_prior(
                all_entity_types, p_t, o_t, "s", "_po")
       
        M = 1 # by default only 1 embedding
        N = w_all.shape[0]
        if isinstance(self.get_s_embedder(), MultipleEmbedder):
            nr_embeddings = self.get_s_embedder().get_nr_embeddings()
            nr_all = nr_embeddings.unsqueeze(1).expand(-1,M)

            M = self.get_s_embedder().nr_embeds
            idxs = torch.arange(M, device=device).unsqueeze(0).expand(N,-1)
            w_all[idxs >= nr_all] = float('-inf')
        w_all = F.softmax(w_all, dim=1)
        w_s, w_o = w_all[s], w_all[o]

        sp_loglike_part = torch.full((B,N,M,M), float('-inf'), device=device)
        po_loglike_part = torch.full((B,N,M,M), float('-inf'), device=device)
        for i in range(s_e.shape[2]):
            for j in range(o_e.shape[2]):
 
                # check if whole batch inactive and skip if so
                if not ((w_s[:,i] == 0).all() or (w_all[:,j] == 0).all()):
                    s_emb_i = s_e[:,:,i]
                    all_entities_j = all_entities[:,:,j]

                    sp_loglikelihood = self._scorer.score_emb(
                        s_emb_i, p_e, all_entities_j, combine="sp_"
                    ).squeeze()
                    sp_loglike_part[:,:,i,j] = sp_loglikelihood

                if not ((w_all[:,i] == 0).all() or (w_o[:,j] == 0).all()):
                    o_emb_j = o_e[:,:,j]
                    all_entities_i = all_entities[:,:,i]
                    
                    if self.recrel:
                        po_loglikelihood = self._scorer.score_emb(
                            o_emb_j, p_i, all_entities_i, combine="sp_"
                        ).squeeze()
                    else:
                        po_loglikelihood = self._scorer.score_emb(
                            all_entities_i, p_e, o_emb_j, combine="_po"
                        ).squeeze()
                    po_loglike_part[:,:,i,j] = po_loglikelihood
        
        w_sp = w_s.view(B, 1, M, 1) * w_all.view(1, N, 1, M)
        w_po = w_o.view(B, 1, 1, M) * w_all.view(1, N, M, 1)

        # perform modified logsumexp trick
        def logsumexp(x, w):
            c,_ = x.max(dim=2, keepdims=True)
            c,_ = c.max(dim=3, keepdims=True)
            weighted_exp = w * (x - c).exp()
            return c.squeeze() + weighted_exp.sum(dim=2).sum(dim=2).log()

        sp_loglikelihood = logsumexp(sp_loglike_part, w_sp)
        po_loglikelihood = logsumexp(po_loglike_part, w_po)
        
        logprior = torch.cat((sp_logprior, po_logprior), dim=1)
        loglikelihood = torch.cat((sp_loglikelihood, po_loglikelihood), dim=1)

        logposterior = logprior + loglikelihood
        return logposterior 
