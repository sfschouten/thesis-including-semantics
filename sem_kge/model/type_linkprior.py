
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from sem_kge.model.embedder import DiscreteStochasticEmbedder, TransTEmbedder
from sem_kge import TypedDataset

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from kge.model.transe import TransEScorer

class TypeLinkPrior(KgeModel):
    """ """

    PRIOR_ONLY_VALUE = "prior-only"

    def __init__(
        self, 
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)
        configuration_key = self.configuration_key
        scorer = None

        self.has_base = self.get_option("base_model.type") != self.PRIOR_ONLY_VALUE
        if self.has_base:
            # Initialize base model
            base_model = KgeModel.create(
                config=config,
                dataset=dataset,
                configuration_key=self.configuration_key + ".base_model",
                init_for_load_only=init_for_load_only,
            )
            scorer = base_model.get_scorer()

        # Initialize this model
        super().__init__(
            config=config,
            dataset=dataset,
            configuration_key=configuration_key,
            scorer=scorer,
            create_embedders=False,
            init_for_load_only=init_for_load_only,
        )

        if self.has_base:
            self._base_model = base_model
            
            self._entity_embedder = self._base_model.get_s_embedder()
            self._relation_embedder = self._base_model.get_p_embedder()
    
        # convert dataset
        dataset = TypedDataset.create(dataset)

        device = self.config.get("job.device")
        self.device = device

        relation_type_freqs = dataset.index("relation_type_freqs").to(device)
        RHO = self.get_option("rho")
        self.rel_common_head = relation_type_freqs[0] > RHO
        self.rel_common_tail = relation_type_freqs[1] > RHO

        self.types_tensor = dataset.index("entity_type_set").to(device)

        def logit(lmbda):
            return torch.log(lmbda / (1-lmbda))

        lambda_head = self.get_option("lambda_head")
        lambda_relation = self.get_option("lambda_relation")
        lambda_tail = self.get_option("lambda_tail")

        if config.get("train.auto_correct"):
            lambda_head = min(max(0, lambda_head), 1)
            lambda_tail = min(max(0, lambda_tail), 1)
            lambda_relation = min(max(0, lambda_relation), 1)
            config.log(f"lambdas: {(lambda_head, lambda_relation, lambda_tail)}")

        lambda_head = torch.full((1,), lambda_head, device=device)
        lambda_relation = torch.full((1,), lambda_relation, device=device)
        lambda_tail = torch.full((1,), lambda_tail, device=device)

        learn_lambda = self.get_option("learn_lambda")
        if learn_lambda:        
            def pos_grad(name, var):
                if var == 0 or var == 1:
                    var += (-1 if var == 1 else 1) * torch.finfo().eps
                    config.log(f"To allow positive gradients {name} has been set to {var.item()}")

            pos_grad("lambda_head", lambda_head)
            pos_grad("lambda_relation", lambda_relation)
            pos_grad("lambda_tail", lambda_tail)

        self._loglambda_head = Parameter(logit(lambda_head), requires_grad=learn_lambda)
        self._loglambda_relation = Parameter(logit(lambda_relation), requires_grad=learn_lambda)
        self._loglambda_tail = Parameter(logit(lambda_tail), requires_grad=learn_lambda)

        if self.has_base:
            # The rest of the initialization is only relevant when there's a base model.
            
            if not self.get_s_embedder() is self.get_o_embedder():
                raise NotImplementedError("TransT currently does not support \
                    the use of different embedders for subjects and objects.")

            # TODO Rather than checking for a specific class, create interface
            # for all embedders that might need type information

            # initialize the TransTEmbedder
            if isinstance(self.get_s_embedder(), TransTEmbedder):
                self.get_s_embedder().initialize_semantics(self.types_tensor)

    def prepare_job(self, job, **kwargs):
        if self.has_base:
            self._base_model.prepare_job(job, **kwargs)
        
        if self.get_option("learn_lambda"):
            # trace the lambda parameters
            def trace_lambda(trace_job):
                trace_job.current_trace["batch"]["loglambda_head"] = self._loglambda_head.item()
                trace_job.current_trace["batch"]["loglambda_relation"] = self._loglambda_relation.item()
                trace_job.current_trace["batch"]["loglambda_tail"] = self._loglambda_tail.item()

            from kge.job import TrainingOrEvaluationJob
            if isinstance(job, TrainingOrEvaluationJob):
                job.pre_batch_hooks.append(trace_lambda)
        
        # If we are not learning anything.
        if not self.has_base and not self.get_option("learn_lambda"):
            from kge.job import TrainingJob
            if isinstance(job, TrainingJob):
                # Manually validate.
                trace_entry = job.valid_job.run()
                job.valid_trace.append(trace_entry)
                
                # Make sure to run post_valid_hooks in case of search.
                for f in job.post_valid_hooks:
                    f(job)

            job.config.set("train.max_epochs", 0)

    def penalty(self, **kwargs):
        if self.has_base:
            return self._base_model.penalty(**kwargs)
        else:
            return []

    def _s(self, T_1, T_2):
        intersection_size = (T_1 & T_2).sum(dim=1).float()
        other_size = T_1.sum(dim=1).float()
        
        result = intersection_size / other_size
        result += 1e-2 # prevent log(0)
       
        # If |T_1| = 0, than the numerator and denominator are both 0
        #   therefore the outcome should be 0.5?
        result[other_size.expand_as(result) == 0] = 0.5
        return result

    def _log_prior(self, T_h, T_r_head, T_r_tail, T_t, corrupted):
        lambda_head = self._loglambda_head.sigmoid()
        lambda_relation = self._loglambda_relation.sigmoid()
        lambda_tail = self._loglambda_tail.sigmoid()

        lambda_head = 1 if lambda_head.isinf() else lambda_head 
        lambda_relation = 1 if lambda_relation.isinf() else lambda_relation 
        lambda_tail = 1 if lambda_tail.isinf() else lambda_tail 

        def default_tensor(T_1, T_2, ignores=frozenset((1,))):
            shape = tuple( max(T_1.shape[i], T_2.shape[i]) for i in range(len(T_1.shape)) if i not in ignores )
            return torch.zeros(tuple(1 for _ in shape), device=self.device).expand(shape)
        
        if corrupted == "s":
            if lambda_head > 0:
                result1 = lambda_head * self._s(T_r_head, T_h).log()
            else:
                result1 = default_tensor(T_r_head, T_h)
            if lambda_relation > 0:
                result2 = lambda_relation * self._s(T_t, T_h).log()
            else:
                result2 = default_tensor(T_t, T_h)
        elif corrupted == "p":
            if lambda_head > 0:
                result1 = lambda_head * self._s(T_r_head, T_h).log()
            else:
                result1 = default_tensor(T_r_head, T_h)
            if lambda_tail > 0:
                result2 = lambda_tail * self._s(T_r_tail, T_t).log()
            else:
                result2 = default_tensor(T_r_tail, T_t)
        elif corrupted == "o":
            if lambda_tail > 0:
                result1 = lambda_tail * self._s(T_r_tail, T_t).log()
            else:
                result1 = default_tensor(T_r_tail, T_t)
            if lambda_relation > 0:
                result2 = lambda_relation * self._s(T_h, T_t).log()
            else:
                result2 = default_tensor(T_h, T_t)
        return result1 + result2

    def _batch_log_prior(self, s_typ, p_typ, o_typ, corrupted: str, combine: str):
        BS = 16 #TODO: make this configurable
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
        s, p, o = s.long(), p.long(), o.long()

        # get typesets
        s_t = self.types_tensor[s]
        r_t = self.rel_common_head[p], self.rel_common_tail[p]
        o_t = self.types_tensor[o]

        logprior = self._log_prior(s_t, r_t[0], r_t[1], o_t, direction)
        loglikelihood = 0
        if self.has_base:
            loglikelihood = self._base_model.score_spo(s, p, o, direction=direction)
        
        if hasattr(self, '_entity_embedder') and \
           isinstance(self._entity_embedder, TransTEmbedder):
            p_emb = self.get_p_embedder().embed(p)
            self._entity_embedder.update(
                (s,p,o),p_emb,(s_t,r_t,o_t),
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
        s, p, o = s.long(), p.long(), o.long()

        # get typesets
        s_t = self.types_tensor[s]
        p_t = self.rel_common_head[p], self.rel_common_tail[p]
        o_t = self.types_tensor[o]

        if entity_subset is not None:
            entity_subset_ = entity_subset.long()
            all_entity_types = self.types_tensor[entity_subset_]
        else:
            all_entity_types = self.types_tensor
    
        sp_logprior = self._batch_log_prior(
                s_t, p_t, all_entity_types, "o", "sp_")
        po_logprior = self._batch_log_prior(
                all_entity_types, p_t, o_t, "s", "_po")
       
        logprior = torch.cat((sp_logprior, po_logprior), dim=1)
        loglikelihood = 0
        if self.has_base:
            loglikelihood = self._base_model.score_sp_po(s, p, o, entity_subset)

        logposterior = logprior + loglikelihood
        return logposterior 

