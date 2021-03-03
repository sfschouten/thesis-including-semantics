import torch
import torch.nn.functional as F

from torch import Tensor

from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

import pyro
from pyro.nn import ConditionalAutoRegressiveNN
from pyro.distributions.transforms.affine_autoregressive import ConditionalAffineAutoregressive
from pyro.distributions import ConditionalTransformedDistribution

from kge.model import KgeEmbedder
from sem_kge.model import GaussianEmbedder

from typing import List


class PyroXformEmbedder(GaussianEmbedder):
    """ 
    
    """

    def __init__(
        self, config, dataset, configuration_key, vocab_size, init_for_load_only=False
    ):
        super().__init__(
            config, dataset, configuration_key, vocab_size, init_for_load_only=init_for_load_only
        )
        
        base_dim = self.get_option("dim")
        hidden_dims = self.get_option("hidden_dims")
        context_dim = self.get_option("context_dim")
        
        # initialize cntx_embedder
        config.set(self.configuration_key + ".cntx_embedder.dim", context_dim)
        if self.configuration_key + ".cntx_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".cntx_embedder.type",
                self.get_option("cntx_embedder.type"),
            )
        self.cntx_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".cntx_embedder", vocab_size 
        )
        
        hypernet = ConditionalAutoRegressiveNN(base_dim, context_dim, hidden_dims)
        self.transform = ConditionalAffineAutoregressive(hypernet)

        pyro.module("my_transform", self.transform)        
        
        self.last_kl = None
        
    def _sample(self, cntx, mu, sigma):
        base_dist = self.DIST(mu, sigma)
        flow_dist = ConditionalTransformedDistribution(base_dist, [self.transform])
        flow_dist = flow_dist.condition(cntx)
        sample = flow_dist.rsample().squeeze()
        self.last_kl = flow_dist.log_prob(sample)
        self.last_kl -= self.prior.log_prob(sample).sum(dim=-1)
        return sample
        
    def embed(self, indexes):
        self.last_indexes = indexes
        cntx = self.cntx_embedder.embed(indexes)
        mu = self.mean_embedder.embed(indexes)
        sigma = self.stdv_embedder.embed(indexes)
        return self._sample(cntx, mu, sigma)

    def embed_all(self):
        self.last_indexes = None
        cntx = self.cntx_embedder.embed_all()
        mu = self.mean_embedder.embed_all()
        sigma = self.stdv_embedder.embed_all()
        return self._sample(cntx, mu, sigma)

    def penalty(self, **kwargs) -> List[Tensor]:
        #result = super().penalty(**kwargs)
        result = []
        
        #indexes = self.last_indexes
        #if indexes != None:
        #    cntx = self.cntx_embedder.embed(indexes)
        #    mu = self.mean_embedder.embed(indexes)
        #    sigma = self.stdv_embedder.embed(indexes)
        #else:
        #    cntx = self.cntx_embedder.embed_all()
        #    mu = self.mean_embedder.embed_all()
        #    sigma = self.stdv_embedder.embed_all()

        #base_dist = self.DIST(mu, sigma)
        #flow_dist = ConditionalTransformedDistribution(base_dist, [self.transform])
        #flow_dist = flow_dist.condition(cntx)
        
        result += [
            (
                f"{self.configuration_key}.regularization_loss",
                self.last_kl.mean()
            )
        ]
        
        return result
        
        
        
        

