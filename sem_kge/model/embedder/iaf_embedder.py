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
from kge.job.train import TrainingJob

from typing import List


class IAFEmbedder(GaussianEmbedder):
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
        
        self.last_ldj = None
   
    def prepare_job(self, job: "Job", **kwargs):
        super().prepare_job(job, **kwargs)
        
        if isinstance(job, TrainingJob):
            original_loss = job.loss
            
            def modified_loss(*args, **kwargs):
                return -self.last_ldj + original_loss(*args, **kwargs)
            
            job.loss = modified_loss
   
    def _sample(self, cntx, mu, sigma):
        B,_ = cntx.shape
    
        transform = self.transform.condition(cntx)
        
        sample1 = super()._sample(mu, sigma)
        sample2 = transform(sample1)
    
        self.last_ldj = transform.log_abs_det_jacobian(sample1, sample2).mean()
    
        return sample2
        
    def embed(self, indexes):
        self.last_indexes = indexes
        cntx = self.cntx_embedder.embed(indexes)
        mu = self.mean_embedder.embed(indexes)
        sigma = F.softplus(self.stdv_embedder.embed(indexes))
        return self._sample(cntx, mu, sigma)

    def embed_all(self):
        self.last_indexes = None
        cntx = self.cntx_embedder.embed_all()
        mu = self.mean_embedder.embed_all()
        sigma = F.softplus(self.stdv_embedder.embed_all())
        return self._sample(cntx, mu, sigma)


        
        
        

