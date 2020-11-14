
import torch
import torch.nn.functional as F
from torch import Tensor

from sem_kge import TypedDataset

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class TransTScorer(RelationalScorer):
    """  """

    def __init__(self, config: Config, dataset: TypedDataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

        self._norm = self.get_option("l_norm")

        self._lambda_head = self.get_option("lambda_head")
        self._lambda_relation = self.get_option("lambda_relation")
        self._lambda_tail = self.get_option("lambda_tail")
        

    def _s(self, T_1, T_2):
        intersection_size = (T_1 & T_2).sum(dim=1)
        other_size = T_1.sum(dim=1)

        result = intersection_size / other_size
        result += 1e-18

        result[other_size == 0] = 1
        return result
   

    def _log_prior_h(self, T_h, T_r_head, T_t):
        return  self._lambda_head * self._s(T_r_head, T_h).log() \
            +   self._lambda_relation * self._s(T_t, T_h).log()
    
    def _log_prior_r(self, T_r_head, T_r_tail, T_h, T_t):
        return  self._lambda_head * self._s(T_r_head, T_h).log() \
            +   self._lambda_tail * self._s(T_r_tail, T_t).log()

    def _log_prior_t(self, T_t, T_r_tail, T_h):
        return  self._lambda_tail * self._s(T_r_tail, T_t).log() \
            +   self._lambda_relation * self._s(T_h, T_t).log()

    def score_emb(
        self, 
        s_emb, p_emb, o_emb, 
        s_typ, p_typ, o_typ,
        corrupted: str,
        combine: str
    ):
        """
        """
        
        # now calculate distance and log-likelihood
        n = p_emb.size(0)
        if combine == "spo":
            distance = F.pairwise_distance(s_emb + p_emb, o_emb, p=self._norm)
        elif combine == "sp_":
            distance = torch.cdist(s_emb + p_emb, o_emb, p=self._norm)
        elif combine == "_po":
            distance = torch.cdist(o_emb - p_emb, s_emb, p=self._norm)
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        p_typ_h, p_typ_t = p_typ
        if corrupted == "s":
            logprior = self._log_prior_h(s_typ, p_typ_h, o_typ)
        elif corrupted == "p":
            logprior = self._log_prior_r(p_typ_h, p_typ_t, s_typ, o_typ)
        elif corrupted == "o":
            logprior = self._log_prior_t(o_typ, p_typ_t, s_typ)

        loglikelihood = F.logsigmoid(distance)
        logposterior = loglikelihood + logprior 

        #print(logposterior.isnan().any().item())
        #print(logposterior.isinf().any().item())

        return logposterior.view(n, -1)

        
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
            scorer=TransTScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
    
        # Create structure with each entity's type set.
        entity_types = dataset.entity_types()
        N = dataset.num_entities()
        T = dataset.num_types()


        config.log("Creating typeset binary vectors")

        # We represent each set as a binary vector.
        # A True value means that the type corresponding
        #  to the column is in the set.
        types_tensor = torch.zeros(
                (N,T), 
                dtype=torch.bool, 
                device=self.config.get("job.device"),
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
        type_head_counts = torch.zeros((R,T), device=self.config.get("job.device"), requires_grad=False)
        type_tail_counts = torch.zeros((R,T), device=self.config.get("job.device"), requires_grad=False)

        triples = dataset.split('train').long()
        for triple in triples:
            h,r,t = triple

            type_head_counts[r] += types_tensor[triple[0]]
            type_tail_counts[r] += types_tensor[triple[2]]

        #type_head_counts /= type_head_counts.sum(dim=)
        #type_tail_counts /= type_tail_counts.sum(dim=)

        RHO = self.get_option("rho")

        self.rel_common_head = type_head_counts > RHO
        self.rel_common_tail = type_tail_counts > RHO

        
        self.types_tensor = types_tensor


    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        """
        
        Args:
            s,p,o: torch.Tensor of shape (batch)
        """
        
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        o_emb = self.get_o_embedder().embed(o)
        
        # get typesets
        s_t = self.types_tensor[s]
        r_t_h = self.rel_common_head[p]
        r_t_t = self.rel_common_tail[p]
        o_t = self.types_tensor[o]
       
        return self._scorer.score_emb(
            s_emb, p_emb, o_emb, 
            s_t, (r_t_h, r_t_t), o_t,
            corrupted=direction,
            combine="spo"
        ).view(-1)


    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        raise NotImplementedError() 

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        raise NotImplementedError()

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        raise NotImplementedError()

    def score_sp_po(
        self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        raise NotImplementedError()


