import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from torch.nn import functional as F

from sem_kge.model import MultipleEmbedder, GrowingMultipleEmbedder

class TransTScorer(RelationalScorer):
    r"""Implementation of the TransE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self._norm = self.get_option("l_norm")
        
    def set_transt_model(self, transt_model):
        #self.parent_model = lambda: transt_model
        
        # Assuming identical s and o embedders.
        self.embedder = transt_model.get_s_embedder()

        self.device = self.config.get("job.device")
        N = self.dataset.num_entities()
        
        # initialize the GrowingMultipleEmbedder
        if isinstance(self.embedder, GrowingMultipleEmbedder):
            self.embedder.initialize_semantics(types_tensor)
 
        # initialize the distribution weights
        if isinstance(self.embedder, MultipleEmbedder):
            nr_embeddings = self.embedder.get_nr_embeddings()
            M = self.embedder.nr_embeds
            idxs = torch.arange(M, device=self.device).unsqueeze(0).expand(N,-1)
            weights = (idxs < nr_embeddings.unsqueeze(1).expand(-1,M)).float()
            weights /= nr_embeddings.unsqueeze(1)
        else:
            weights = torch.ones((N,1), device=self.device) 
        self.weights = torch.nn.Parameter(weights)        

    def _score_translation(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)
        if combine == "spo":
            out = -F.pairwise_distance(s_emb + p_emb, o_emb, p=self._norm)
        elif combine == "sp_":
            out = -torch.cdist(s_emb + p_emb, o_emb, p=self._norm)
        elif combine == "_po":
            out = -torch.cdist(o_emb - p_emb, s_emb, p=self._norm)
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)
        return out.view(n, -1)
        
    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        
        # obtain distribution's weights
        M = 1 # by default only 1 embedding
        w_s, w_o = self.weights[s], self.weights[o]                 # B x M
        if isinstance(self.embedder, MultipleEmbedder):
            nr_embeddings = self.embedder.get_nr_embeddings()
            nr_s = nr_embeddings[s].unsqueeze(1).expand(-1,M)
            nr_o = nr_embeddings[o].unsqueeze(1).expand(-1,M)

            M = self.embedder.nr_embeds
            idxs = torch.arange(M, device=self.device).unsqueeze(0).expand(B,-1)

            w_s[idxs >= nr_s] = float('-inf')
            w_o[idxs >= nr_o] = float('-inf')
        w_s, w_o = F.softmax(w_s, dim=1), F.softmax(w_o, dim=1)     # B x M
        
        
        part_loglikelihoods = torch.full((B,M,M), float('-inf'), device=self.device)
        for i in range(s_emb.shape[2]):
            for j in range(o_emb.shape[2]):

                # check if whole batch inactive and skip if so
                if (w_s[:,i] == 0).all() or (w_o[:,j] == 0).all():
                    continue
               
                s_emb_i = s_emb[:,:,i]
                o_emb_j = o_emb[:,:,j]

                # weights of inactive vectors will be 0, so won't be summed
                loglikelihood = self._score_translation(
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
        
        if isinstance(self.embedder, GrowingMultipleEmbedder):
            self.embedder.update(
                (s,p,o),(s_emb,p_emb,o_emb),(s_t,r_t,o_t),
                loglikelihood, self._s)
                
        return loglikelihood
        
        
class TransT(KgeModel):
    r"""Implementation of the TransT KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=TransTScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
        
        self._scorer.set_transt_model(self)
        
        
