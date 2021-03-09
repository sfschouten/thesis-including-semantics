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
from sem_kge.model import LocScaleEmbedder
from kge.job.train import TrainingJob

from typing import List

import mdmm

class IAFEmbedder(KgeEmbedder):
    """ 
    
    """

    def __init__(
        self, config, dataset, configuration_key, vocab_size, init_for_load_only=False
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )
        
        base_dim = self.get_option("dim")
        
        # initialize base_embedder
        config.set(self.configuration_key + ".base_embedder.dim", base_dim)
        if self.configuration_key + ".base_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".base_embedder.type",
                self.get_option("base_embedder.type"),
            )
        self.base_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".base_embedder", vocab_size
        )
        
        # initialize cntx_embedder
        context_dim = self.get_option("context_dim")
        config.set(self.configuration_key + ".cntx_embedder.dim", context_dim)
        if self.configuration_key + ".cntx_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".cntx_embedder.type",
                self.get_option("cntx_embedder.type"),
            )
        self.cntx_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".cntx_embedder", vocab_size 
        )
        
        hidden_dims = self.get_option("hidden_dims")
        hypernet = ConditionalAutoRegressiveNN(base_dim, context_dim, hidden_dims)
        self.transform = ConditionalAffineAutoregressive(hypernet)
        
        self.direction = self.check_option('direction', ['density-estimation', 'sampling'])
        
        self.last_ldj = torch.zeros((1))
        
   
    def prepare_job(self, job: "Job", **kwargs):
        super().prepare_job(job, **kwargs)
        self.base_embedder.prepare_job(job, **kwargs)
        
        if isinstance(job, TrainingJob):
            # use Modified Differential Multiplier Method for ldj
            min_ldj_constraint = mdmm.MinConstraint(
                lambda: self.last_ldj,
                self.get_option("ldj_min_threshold"),
                scale = self.get_option("ldj_min_scale"), 
                damping = self.get_option("ldj_min_damping")
            )
            mdmm_module = mdmm.MDMM([min_ldj_constraint])
            
            # update optimizer
            lambdas = [min_ldj_constraint.lmbda]
            slacks = [min_ldj_constraint.slack]
            
            lr = next(g['lr'] for g in job.optimizer.param_groups if g['name'] == 'default')
            job.optimizer.add_param_group({'params': lambdas, 'lr': -lr})
            job.optimizer.add_param_group({'params': slacks, 'lr': lr})
            
            original_loss = job.loss
            
            def modified_loss(*args, **kwargs):
                return mdmm_module(original_loss(*args, **kwargs)).value
            
            job.loss = modified_loss
            
        # trace the ldj
        def trace_ldj(job):
            job.current_trace["batch"]["ldj"] = self.last_ldj.item()
            if isinstance(job, TrainingJob):
                job.current_trace["batch"]["ldj_lambda"] = min_ldj_constraint.lmbda.item()

        from kge.job import TrainingOrEvaluationJob
        if isinstance(job, TrainingOrEvaluationJob):
            job.pre_batch_hooks.append(trace_ldj)
            
            
    def log_pdf(self, points, indexes):
        """
        points:  the points at which the pdf is to be evaluated [* x D]
        indexes: the indices of the loc/scale that are to parameterize the
                 distribution [*]
        returns: log of pdf [*]
        """
        cntx = self.cntx_embedder.embed(indexes)
        
        if self.direction == 'density-estimation':
            transform = self.transform
        else:
            transform = self.transform.inv
        transform = transform.condition(cntx)
        
        points2 = transform(points)
        log_pdf = self.base_embedder.log_pdf(points2, indexes)
        log_pdf += transform.log_abs_det_jacobian(points, points2)
        return log_pdf

    def _transform(self, samples, cntx):
        if self.direction == 'sampling':
            transform = self.transform
        else:
            transform = self.transform.inv
        transform = transform.condition(cntx)
        
        iaf_samples = transform(samples)
    
        self.last_ldj = transform.log_abs_det_jacobian(samples, iaf_samples).mean()
        return iaf_samples
        
    def embed(self, indexes):
        samples = self.base_embedder._embed(indexes)
        cntx = self.cntx_embedder.embed(indexes)
        return self._transform(samples, cntx).mean(dim=0)

    def embed_all(self):
        sample = self.base_embedder.embed_all()
        cntx = self.cntx_embedder.embed_all()
        return self._transform(sample, cntx).mean(dim=0)

        

