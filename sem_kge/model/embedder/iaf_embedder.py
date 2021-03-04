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
   
    def _transform(self, sample, cntx):
        transform = self.transform.condition(cntx)        
        iaf_sample = transform(sample)
    
        self.last_ldj = transform.log_abs_det_jacobian(sample, iaf_sample).mean()
    
        return iaf_sample
        
    def embed(self, indexes):
        sample = super().embed(indexes)
        cntx = self.cntx_embedder.embed(indexes)
        return self._transform(sample, cntx)

    def embed_all(self):
        sample = super().embed_all()
        cntx = self.cntx_embedder.embed_all()
        return self._transform(sample, cntx)


        
        
        

