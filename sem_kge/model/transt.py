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
        self.device = self.config.get("job.device")
        
    def set_transt_model(self, transt_model):
        # Assuming identical s and o embedders.
        self.embedder = transt_model.get_s_embedder()

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
      
        if isinstance(self.embedder, MultipleEmbedder):
            s_emb, w_s = s_emb                                      # B x M
            o_emb, w_o = o_emb
        else:
            w_s = torch.ones((B,1), device=self.device)
            w_o = torch.ones((B,1), device=self.device)

        B, M = w_s.shape
        N = 1 if combine == "spo" else w_o.shape[0]

        part_loglikelihoods = torch.full((B,M,M,N), float('-inf'), device=self.device)
            
        for i in range(s_emb.shape[2]):
            for j in range(o_emb.shape[2]):
                # check if whole batch inactive and skip if so
                if (w_s[:,i] == 0).all() or (w_o[:,j] == 0).all():
                    continue

                s_emb_i = s_emb[:,:,i]
                o_emb_j = o_emb[:,:,j]

                # weights of inactive vectors will be 0, so won't be summed
                loglikelihood = self._score_translation(
                    s_emb_i, p_emb, o_emb_j, combine=combine
                ).squeeze()
                part_loglikelihoods[:,i,j] = loglikelihood.view(-1, N)
        
        if combine == "spo":
            w_s, w_o = w_s.view(B, M, 1, 1), w_o.view(B, 1, M, 1)
        elif combine == "sp_":
            w_s, w_o = w_s.view(B, M, 1, 1), w_o.view(1, 1, M, N)

        # perform modified logsumexp trick
        x = part_loglikelihoods                         # B x M x M x [1|N]
        c,_ = x.max(dim=1, keepdims=True)               # B x 1 x M x [1|N]
        c,_ = c.max(dim=2, keepdims=True)               # B x 1 x 1 x [1|N]
        weighted_exp = w_s * w_o * (x - c).exp()        # B x 1 x 1 x [1|N]

        loglikelihood = c.squeeze(dim=1).squeeze(dim=1) \
                      + weighted_exp.sum(dim=1).sum(dim=1).log()
        loglikelihood = loglikelihood.squeeze()

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
