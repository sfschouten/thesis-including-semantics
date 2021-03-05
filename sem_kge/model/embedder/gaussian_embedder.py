import torch
import torch.nn.functional as F

from torch import Tensor

from torch.distributions.kl import kl_divergence

from kge.model import KgeEmbedder
from kge.job.train import TrainingJob

from typing import List

import mdmm


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
    
        self.prior = self.DIST(
            torch.tensor([0.0], device=self.device, requires_grad=False).expand(base_dim).unsqueeze(0),
            torch.tensor([1.0], device=self.device, requires_grad=False).expand(base_dim).unsqueeze(0)
        )
    
        self.last_regularization_loss = torch.zeros((1))
        
    def prepare_job(self, job: "Job", **kwargs):

        if isinstance(job, TrainingJob):
            # use Modified Differential Multiplier Method for regularization loss
            max_kl_constraint = mdmm.MaxConstraint(
                lambda: self.last_regularization_loss,
                self.get_option("kl_max_threshold"),
                scale = self.get_option("kl_max_scale"), 
                damping = self.get_option("kl_max_damping")
            )
            mdmm_module = mdmm.MDMM([max_kl_constraint])
            
            # update optimizer
            lambdas = [max_kl_constraint.lmbda]
            slacks = [max_kl_constraint.slack]
            
            lr = next(g['lr'] for g in job.optimizer.param_groups if g['name'] == 'default')
            job.optimizer.add_param_group({'params': lambdas, 'lr': -lr})
            job.optimizer.add_param_group({'params': slacks, 'lr': lr})
            
            original_loss = job.loss
            
            def modified_loss(*args, **kwargs):
                return mdmm_module(original_loss(*args, **kwargs)).value
            
            job.loss = modified_loss        

        # trace the regularization loss
        def trace_regularization_loss(job):
            job.current_trace["batch"]["regularization_loss"] = self.last_regularization_loss.item()
            if isinstance(job, TrainingJob):
                job.current_trace["batch"]["regularization_lambda"] = max_kl_constraint.lmbda.item()

        from kge.job import TrainingOrEvaluationJob
        if isinstance(job, TrainingOrEvaluationJob):
            job.pre_batch_hooks.append(trace_regularization_loss)


    def _sample(self, mu, sigma):
        dist = self.DIST(mu, sigma)
        self.last_regularization_loss = kl_divergence(dist, self.prior).mean()
        
        sample_shape = torch.Size([1 if self.training else 10])
        sample = dist.rsample(sample_shape=sample_shape)
        return sample

    def _embed(self, indexes):
        mu = self.mean_embedder.embed(indexes)
        sigma = F.softplus(self.stdv_embedder.embed(indexes))
        return self._sample(mu, sigma)

    def embed(self, indexes):
        return self._embed(indexes).mean(dim=0)

    def _embed_all(self):
        mu = self.mean_embedder.embed_all() 
        sigma = F.softplus(self.stdv_embedder.embed_all())
        return self._sample(mu, sigma)

    def embed_all(self):
        return self._embed_all().mean(dim=0)

        

