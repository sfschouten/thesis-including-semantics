import torch
import torch.nn.functional as F

from torch import Tensor
from torch.distributions.kl import kl_divergence
from torch.distributions.utils import _standard_normal

from functools import partial
from typing import List

from kge.model import KgeEmbedder
from kge.job.train import TrainingJob

from sem_kge import misc

import mdmm


class GaussianEmbedder(KgeEmbedder):
    DIST = torch.distributions.normal.Normal

    def __init__(
        self, config, dataset, configuration_key, 
        vocab_size, init_for_load_only=False
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )
        
        base_dim = self.get_option("dim")
        self.device = self.config.get("job.device")
        self.vocab_size = vocab_size
        self.kl_loss = self.get_option("kl_loss")

        # initialize loc_embedder
        config.set(self.configuration_key + ".loc_embedder.dim", base_dim)
        if self.configuration_key + ".loc_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".loc_embedder.type",
                self.get_option("loc_embedder.type"),
            )
        self.loc_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".loc_embedder", vocab_size 
        )
        
        # initialize scale_embedder
        config.set(self.configuration_key + ".scale_embedder.dim", base_dim)
        if self.configuration_key + ".scale_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".scale_embedder.type",
                self.get_option("scale_embedder.type"),
            )
        self.scale_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".scale_embedder", vocab_size 
        )

        self.prior = self.DIST(
            torch.tensor([0.0], device=self.device, requires_grad=False).expand(base_dim).unsqueeze(0),
            torch.tensor([1.0], device=self.device, requires_grad=False).expand(base_dim).unsqueeze(0)
        )
        
        self.last_kl_divs = []
        
        
    def prepare_job(self, job: "Job", **kwargs):
        super().prepare_job(job, **kwargs)
        self.loc_embedder.prepare_job(job, **kwargs)
        self.scale_embedder.prepare_job(job, **kwargs)
        
        if self.kl_loss and isinstance(job, TrainingJob):
            # use Modified Differential Multiplier Method for regularization loss
            max_kl_constraint = mdmm.MaxConstraint(
                lambda: self.last_kl_divs[-1],
                self.get_option("kl_max_threshold"),
                scale = self.get_option("kl_max_scale"), 
                damping = self.get_option("kl_max_damping")
            )
            dummy_val = torch.zeros((1), device=self.device)
            kl_max_module = mdmm.MDMM([max_kl_constraint])
            misc.add_constraints_to_job(job, kl_max_module)
            self.kl_max_module = partial(kl_max_module, dummy_val)

        # trace the regularization loss
        def trace_regularization_loss(job):
            last_kl_avg = sum( kl.item() / len(self.last_kl_divs) for kl in self.last_kl_divs )
            self.last_kl_divs = []
            key = f"{self.configuration_key}.kl"
            job.current_trace["batch"][key] = last_kl_avg
            if self.kl_loss and isinstance(job, TrainingJob):
                job.current_trace["batch"][f"{key}_lambda"] = max_kl_constraint.lmbda.item()

        from kge.job import TrainingOrEvaluationJob
        if isinstance(job, TrainingOrEvaluationJob):
            job.pre_batch_hooks.append(trace_regularization_loss)


    def dist(self, indexes=None, use_cache=False, cache_action='push'):
        """
        Instantiates `self.DIST` using the parameters obtained 
        from embedding `indexes` or all indexes if `indexes' is None.
        """
        def mod_rsample(dist, sample_shape=torch.Size()):
            """Modified rsample that saves samples so we can calculate penalty later."""
            if not use_cache or cache_action=='push':
                shape = dist._extended_shape(sample_shape)
                eps = _standard_normal(shape, dtype=dist.loc.dtype, device=dist.loc.device)
                
                if use_cache and cache_action=='push':
                    self.sample_stack.append(eps.detach())
            
            if use_cache and cache_action=='pop':
                eps = self.sample_stack.pop(0)
            
            return dist.loc + eps * dist.scale
        
        if indexes==None:
            mu = self.loc_embedder.embed_all() 
            sigma = F.softplus(self.scale_embedder.embed_all())
        else:
            mu = self.loc_embedder.embed(indexes)
            sigma = F.softplus(self.scale_embedder.embed(indexes))
            
        dist = self.DIST(mu, sigma)
        dist.rsample = mod_rsample
        return dist

    def log_pdf(self, points, indexes):
        """
        points:  the points at which the pdf is to be evaluated [* x D]
        indexes: the indices of the loc/scale that are to parameterize the
                 distribution [*]
        returns: log of pdf [*]
        """
        dist = self.dist(indexes)
        return dist.log_prob(points).mean(dim=-1)

    def sample(self, indexes=None, use_cache=False, cache_action='push'):
        dist = self.dist(indexes, use_cache, cache_action)
        sample_shape = torch.Size([1 if self.training else 10])
        #TODO set '1' and '10' values in config
        
        sample = dist.rsample(sample_shape=sample_shape)
        return sample
        
    def embed(self, indexes):
        return self.sample(indexes).mean(dim=0)
    def embed_all(self):
        return self.sample().mean(dim=0)

    def penalty(self, **kwargs):
        terms = super().penalty(**kwargs)
        terms += self.loc_embedder.penalty(**kwargs)
        terms += self.scale_embedder.penalty(**kwargs)

        indexes = kwargs['indexes']
        kl_div = kl_divergence(self.dist(indexes), self.prior).mean()
        self.last_kl_divs.append(kl_div)

        if self.kl_loss:
            terms += [ ( 
                f"{self.configuration_key}.kl", 
                self.kl_max_module().value
            ) ]
        
        return terms

