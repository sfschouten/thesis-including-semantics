import torch
import torch.nn.functional as F

from torch import Tensor

from torch.distributions.kl import kl_divergence

from kge.model import KgeEmbedder
from kge.job.train import TrainingJob

from functools import partial
from typing import List
import importlib

import mdmm


class LocScaleEmbedder(KgeEmbedder):
    """ 
    Class for stochastic embeddings with distributions that take a `loc` and
    a `scale` argument.
    """

    def __init__(
        self, config, dataset, configuration_key, 
        vocab_size, init_for_load_only=False
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )

        base_dim = self.get_option("dim")
        dist_class = self.get_option("dist_class")
        
        self.dist_class = eval(dist_class)
        self.vocab_size = vocab_size

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

        self.device = self.config.get("job.device")
    
        self.prior = self.dist_class(
            torch.tensor([0.0], device=self.device, requires_grad=False).expand(base_dim).unsqueeze(0),
            torch.tensor([1.0], device=self.device, requires_grad=False).expand(base_dim).unsqueeze(0)
        )
    
        self.last_regularization_loss = torch.tensor(0)
        
    def prepare_job(self, job: "Job", **kwargs):
        super().prepare_job(job, **kwargs)
        self.loc_embedder.prepare_job(job, **kwargs)
        self.scale_embedder.prepare_job(job, **kwargs)
        
        if isinstance(job, TrainingJob):
            # use Modified Differential Multiplier Method for regularization loss
            max_kl_constraint = mdmm.MaxConstraint(
                lambda: self.last_regularization_loss,
                self.get_option("kl_max_threshold"),
                scale = self.get_option("kl_max_scale"), 
                damping = self.get_option("kl_max_damping")
            )
            mdmm_module = mdmm.MDMM([max_kl_constraint])
            misc.add_constraints_to_job(job, self.mdmm_module)
            
            # update loss
            original_loss = job.loss
            def modified_loss(*args, **kwargs):
                return mdmm_module(original_loss(*args, **kwargs)).value
            job.loss = modified_loss        

        # trace the regularization loss
        def trace_regularization_loss(job):
            key = f"{self.configuration_key}.kl"
            job.current_trace["batch"][key] = self.last_regularization_loss.item()
            if isinstance(job, TrainingJob):
                job.current_trace["batch"][f"{key}_lambda"] = max_kl_constraint.lmbda.item()

        from kge.job import TrainingOrEvaluationJob
        if isinstance(job, TrainingOrEvaluationJob):
            job.pre_batch_hooks.append(trace_regularization_loss)


    def dist(self, indexes, calc_kl=True):
        """
        Instantiates `self.dist_class` using the parameters obtained 
        from embedding `indexes`.
        """
        mu = self.loc_embedder.embed(indexes)
        sigma = F.softplus(self.scale_embedder.embed(indexes))
        dist = self.dist_class(mu, sigma)
        if calc_kl:
            self.last_regularization_loss = kl_divergence(dist, self.prior).mean()
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

    def _sample(self, dist):
        sample_shape = torch.Size([1 if self.training else 10])
        sample = dist.rsample(sample_shape=sample_shape)
        return sample

    def _embed(self, indexes):
        dist = self.dist(indexes)
        return self._sample(dist)

    def embed(self, indexes):
        return self._embed(indexes).mean(dim=0)

    def _embed_all(self):
        mu = self.loc_embedder.embed_all() 
        sigma = F.softplus(self.scale_embedder.embed_all())
        dist = self.dist_class(mu, sigma)
        return self._sample(dist)

    def embed_all(self):
        return self._embed_all().mean(dim=0)

        
