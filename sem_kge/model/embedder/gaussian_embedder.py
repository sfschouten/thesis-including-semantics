import torch
import torch.nn.functional as F

from torch import Tensor

from torch.distributions.kl import kl_divergence

from kge.model import KgeEmbedder
from kge.job.train import TrainingJob

from typing import List


class GaussianEmbedder(KgeEmbedder):
    """ 
    
    """

    DIST = torch.distributions.normal.Normal

    def __init__(
        self, config, dataset, configuration_key, 
        vocab_size, init_for_load_only=False
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )

        self.vocab_size = vocab_size

        base_dim = self.get_option("dim")

        # initialize mean_embedder
        config.set(self.configuration_key + ".mean_embedder.dim", base_dim)
        if self.configuration_key + ".mean_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".mean_embedder.type",
                self.get_option("mean_embedder.type"),
            )
        self.mean_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".mean_embedder", vocab_size 
        )
        
        # initialize stdv_embedder
        config.set(self.configuration_key + ".stdv_embedder.dim", base_dim)
        if self.configuration_key + ".stdv_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".stdv_embedder.type",
                self.get_option("stdv_embedder.type"),
            )
        self.stdv_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".stdv_embedder", vocab_size 
        )

        self.device = self.config.get("job.device")
        self.kl_div_factor = self.get_option("kl_div_factor")
    
        self.prior = self.DIST(
            torch.tensor([0.0], device=self.device, requires_grad=False).expand(base_dim).unsqueeze(0),
            torch.tensor([1.0], device=self.device, requires_grad=False).expand(base_dim).unsqueeze(0)
        )
    
        self.last_regularization_loss = None
        
    def prepare_job(self, job: "Job", **kwargs):     
    
        if isinstance(job, TrainingJob):   
            original_loss = job.loss
            
            def modified_loss(*args, **kwargs):
                return self.last_regularization_loss + original_loss(*args, **kwargs)
            
            job.loss = modified_loss

    def _sample(self, mu, sigma):
        dist = self.DIST(mu, sigma)
        sample = dist.rsample().squeeze()
        self.last_regularization_loss = kl_divergence(dist, self.prior).mean()
        return sample

    def embed(self, indexes):
        self.last_indexes = indexes
        mu, sigma = self.mean_embedder.embed(indexes), F.softplus(self.stdv_embedder.embed(indexes))
        return self._sample(mu, sigma)

    def embed_all(self):
        self.last_indexes = None
        mu, sigma = self.mean_embedder.embed_all(), F.softplus(self.stdv_embedder.embed_all())
        return self._sample(mu, sigma)

        

