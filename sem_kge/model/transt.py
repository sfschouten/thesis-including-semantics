
import torch
import torch.nn.functional as F
from torch import Tensor

from sem_kge.model.embedder import GrowingMultipleEmbedder
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

        R = dataset.num_relations()

        config.log("Creating common typeset vectors")

        # Create structure with each relation's common type set. 
        type_head_counts = torch.zeros((R,T), device=device, requires_grad=False)
        type_tail_counts = torch.zeros((R,T), device=device, requires_grad=False)
        type_totals      = torch.zeros((R,1), device=device, requires_grad=False)

        triples = dataset.split('train').long()
        for triple in triples:
            h,r,t = triple

            type_head_counts[r] += types_tensor[triple[0]]
            type_tail_counts[r] += types_tensor[triple[2]]
            type_totals[r] += 1

        RHO = self.get_option("rho")

        self.rel_common_head = type_head_counts / type_totals > RHO
        self.rel_common_tail = type_tail_counts / type_totals > RHO

        self.types_tensor = types_tensor

        # TODO Rather than checking for a specific class, create interface
        # for all embedders that might need type information

        # initialize the GrowingEmbeddersEmbedders
        for embedder in (self.get_s_embedder(), self.get_o_embedder()):
            if isinstance(embedder, GrowingMultipleEmbedder):
                embedder.initialize_semantics(T)

        self._lambda_head = self.get_option("lambda_head")
        self._lambda_relation = self.get_option("lambda_relation")
        self._lambda_tail = self.get_option("lambda_tail")
        
    def _s(self, T_1, T_2):
        intersection_size = (T_1 & T_2).sum(dim=1)
        other_size = T_1.sum(dim=1)
        
        result = intersection_size / other_size
        result += 1e-18 # prevent log(0)
        
        # If |T_1| = 0, than the numerator and denominator are both 0
        #   therefore the outcome should be 1 ? TODO: check
        result[other_size.expand_as(result) == 0] = 1
        return result

    def _log_prior(self, T_h, T_r_head, T_r_tail, T_t, corrupted):
        result = 0
        if   corrupted == "s":
            if self._lambda_head > 0:
                result += self._s(T_r_head, T_h).log() 
            if self._lambda_relation > 0:
                result += self._s(T_t, T_h).log()
        elif corrupted == "p":
            if self._lambda_head > 0:
                result += self._s(T_r_head, T_h).log() 
            if self._lambda_tail > 0:
                result += self._s(T_r_tail, T_t).log()
        elif corrupted == "o":
            if self._lambda_tail > 0:
                result += self._s(T_r_tail, T_t).log() 
            if self._lambda_relation > 0:
                result += self._s(T_h, T_t).log()
        return result

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


    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        """
        """
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        o_emb = self.get_o_embedder().embed(o)
        
        for emb in (s_emb, o_emb):
            if emb.ndim == 2:
                emb.unsqueeze_(2)

        # get typesets
        s, p, o = s.long(), p.long(), o.long()
        s_t = self.types_tensor[s]
        r_t_h = self.rel_common_head[p]
        r_t_t = self.rel_common_tail[p]
        o_t = self.types_tensor[o]

        logprior = self._log_prior(s_t, r_t_h, r_t_t, o_t, direction)
        
        loglikelihood = 0
        for i in range(s_emb.shape[2]):
            for j in range(o_emb.shape[2]):

                # TODO: check for empty and skip

                s_emb_i = s_emb[:,:,i]
                o_emb_j = o_emb[:,:,j]
                
                loglikelihood += self._scorer.score_emb(
                    s_emb_i, p_emb, o_emb_j, combine="spo"
                ).squeeze()
        loglikelihood /= s_emb.shape[2] * o_emb.shape[2]


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

        s_e = self.get_s_embedder().embed(s)
        p_e = self.get_p_embedder().embed(p)
        o_e = self.get_o_embedder().embed(o)

        # get typesets
        s, p, o = s.long(), p.long(), o.long()
        s_t = self.types_tensor[s]
        p_t = self.rel_common_head[p], self.rel_common_tail[p]
        o_t = self.types_tensor[o]

        if entity_subset is not None:
            all_entity_types = self.types_tensor[entity_subset]
        else:
            all_entity_types = self.types_tensor
        
        if self.get_s_embedder() is self.get_o_embedder():
            if entity_subset is not None:
                all_entities = self.get_s_embedder().embed(entity_subset)
            else:
                all_entities = self.get_s_embedder().embed_all()
            all_objects = all_entities
            all_subjects = all_entities
        else:
            if entity_subset is not None:
                all_objects = self.get_o_embedder().embed(entity_subset)
                all_subjects = self.get_s_embedder().embed(entity_subset)
            else:
                all_objects = self.get_o_embedder().embed_all()
                all_subjects = self.get_s_embedder().embed_all()
        
        for emb in (s_e, o_e, all_subjects, all_objects):
            if emb.ndim == 2:
                emb.unsqueeze_(2)

        sp_logprior = self._batch_log_prior(
                s_t, p_t, all_entity_types, "o", "sp_")
        po_logprior = self._batch_log_prior(
                all_entity_types, p_t, o_t, "s", "_po")
        
        sp_loglikelihood = 0
        po_loglikelihood = 0
        for i in range(s_e.shape[2]):
            for j in range(o_e.shape[2]):

                # TODO: check for empty and skip

                s_emb_i = s_e[:,:,i]
                o_emb_j = o_e[:,:,j]
                all_subjects_i = all_subjects[:,:,i]
                all_objects_j = all_objects[:,:,j]
                
                sp_loglikelihood += self._scorer.score_emb(
                        s_emb_i, p_e, all_objects_j, combine="sp_"
                    )
                po_loglikelihood += self._scorer.score_emb(
                        all_subjects_i, p_e, o_emb_j, combine="_po"
                    )


        logprior = torch.cat((sp_logprior, po_logprior), dim=1)
        loglikelihood = torch.cat((sp_loglikelihood, po_loglikelihood), dim=1)
        loglikelihood /= s_e.shape[2] * o_e.shape[2] #TODO replace with appropriate value after Growing is added


        logposterior = logprior + loglikelihood
        return logposterior 
