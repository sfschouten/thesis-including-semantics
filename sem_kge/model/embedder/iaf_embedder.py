import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.kl import kl_divergence

from pyro.nn import ConditionalAutoRegressiveNN
from pyro.distributions.transforms.affine_autoregressive import ConditionalAffineAutoregressive

from functools import partial

from kge.model import KgeEmbedder
from kge.job.train import TrainingJob

from sem_kge import misc

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
        self.device = self.config.get('job.device')
        self.ldj_loss = self.get_option("ldj_loss")
        
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
        hidden_dims = [ max(h, base_dim+context_dim) for h in hidden_dims ]
        hypernet = ConditionalAutoRegressiveNN(base_dim, context_dim, hidden_dims)
        self.transform = ConditionalAffineAutoregressive(hypernet)
        
        self.direction = self.check_option('direction', ['density-estimation', 'sampling'])
        
        self.last_ldjs = []
        
   
    def prepare_job(self, job: "Job", **kwargs):
        super().prepare_job(job, **kwargs)
        self.base_embedder.prepare_job(job, **kwargs)
        self.cntx_embedder.prepare_job(job, **kwargs)
        
        if self.ldj_loss and isinstance(job, TrainingJob):
            # use Modified Differential Multiplier Method for ldj
            min_ldj_constraint = mdmm.MinConstraint(
                lambda: self.last_ldjs[-1],
                self.get_option("ldj_min_threshold"),
                scale = self.get_option("ldj_min_scale"), 
                damping = self.get_option("ldj_min_damping")
            )
            dummy_val = torch.zeros((1), device=self.device)
            ldj_min_module = mdmm.MDMM([min_ldj_constraint])
            misc.add_constraints_to_job(job, ldj_min_module)
            self.ldj_min_module = partial(ldj_min_module, dummy_val)
            
        # trace the ldj
        def trace_ldj(job):        
            last_ldj_avg = sum( ldj.item() / len(self.last_ldjs) for ldj in self.last_ldjs )
            self.last_ldjs = []
            key = f"{self.configuration_key}.ldj"
            job.current_trace["batch"][key] = last_ldj_avg
            if self.ldj_loss and isinstance(job, TrainingJob):
                job.current_trace["batch"][f"{key}_lambda"] = min_ldj_constraint.lmbda.item()

        from kge.job import TrainingOrEvaluationJob
        if isinstance(job, TrainingOrEvaluationJob):
            job.pre_batch_hooks.append(trace_ldj)
            
    def _transform(self, points, cntx, direction):
        transform = self.transform
        if self.direction != direction:
            transform = self.transform.inv
            #TODO warning about performance/memory
            
        transform = transform.condition(cntx)
        transformed = transform(points)
        return transformed, transform
    
    def log_pdf(self, points, indexes):
        """
        points:  the points at which the pdf is to be evaluated [* x D]
        indexes: the indices of the loc/scale that are to parameterize the
                 distribution [*]
        returns: log of pdf [*]
        """
        cntx = self.cntx_embedder.embed(indexes)
        transformed, transform = self._transform(points, cntx, 'density-estimation')
        ldj = transform.log_abs_det_jacobian(points, transformed)
        return ldj + self.base_embedder.log_pdf(transformed, indexes)
        
    def embed(self, indexes):
        samples = self.base_embedder.sample(indexes, use_cache=True, cache_action='push')
        cntx = self.cntx_embedder.embed(indexes)
        return self._transform(samples, cntx, 'sampling')[0].mean(dim=0)

    def embed_all(self):
        samples = self.base_embedder.sample(use_cache=True, cache_action='push')
        cntx = self.cntx_embedder.embed_all()
        return self._transform(samples, cntx, 'sampling')[0].mean(dim=0)

    def penalty(self, **kwargs):
        terms = super().penalty(**kwargs)
        terms += self.base_embedder.penalty(**kwargs)
        terms += self.cntx_embedder.penalty(**kwargs)
        
        indexes = kwargs['indexes']
        cntx = self.cntx_embedder.embed(indexes)
        if self.direction == 'sampling':
            points = self.base_embedder.sample(indexes, use_cache=True, cache_action='pop')
            transformed, transform = self._transform(points, cntx, 'sampling')
        elif self.direction == 'density-estimation':
            # When this embedder is used (primarily) for density-estimation, the
            # points at which density is to be estimated, come from the object
            # that called the log_pdf function. We expect that object to also pass
            # those points in its penalty function to us here using `kwargs` so 
            # that we may calculate the relevant penalty terms.
            points = kwargs['points']
            transformed, transform = self._transform(points, cntx, 'density-estimation')
             
        ldj = transform.log_abs_det_jacobian(points, transformed).mean()
        self.last_ldjs.append(ldj)

        if self.ldj_loss:
            terms += [ ( 
                f"{self.configuration_key}.ldj", 
                self.ldj_min_module().value
            ) ]
        
        return terms

