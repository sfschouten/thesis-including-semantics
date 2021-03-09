import torch
import torch.nn.functional as F

from torch.nn import MultiheadAttention

from kge.model import KgeEmbedder
from kge.job.train import TrainingJob

from sem_kge import TypedDataset

from functools import partial
import random

import mdmm

class TypePriorEmbedder(KgeEmbedder):
    """ 
    """

    def __init__(
        self, config, dataset, configuration_key, 
        vocab_size, init_for_load_only=False
    ):
        
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )


        dim = self.get_option("dim")

        self.vocab_size = vocab_size
        self.device = self.config.get("job.device")       
        
        # initialize base_embedder
        config.set(self.configuration_key + ".base_embedder.dim", dim)
        if self.configuration_key + ".base_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".base_embedder.type",
                self.get_option("base_embedder.type"),
            )
        self.base_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".base_embedder", vocab_size 
        )
        
        # convert dataset
        self.dataset = TypedDataset.create(dataset)
        entity_types = self.dataset.entity_types()
        N = self.dataset.num_entities()
        T = self.dataset.num_types()
        
        self.PADDING_IDX = T
        
        T_ = max( types_str.count(',') + 1 for types_str in entity_types if types_str is not None )
        self.entity_types = torch.full((N, T_), self.PADDING_IDX, device=self.device)
        
        for i,types_str in enumerate(entity_types):
            if types_str is None:
                continue
            for j,t in enumerate(types_str.split(',')):
                self.entity_types[i,j] = int(t)

        self.type_padding = self.entity_types == self.PADDING_IDX
        
        # initialize type embedder
        config.set(self.configuration_key + ".prior_embedder.dim", dim)
        if self.configuration_key + ".prior_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".prior_embedder.type",
                self.get_option("prior_embedder.type"),
            )
        self.prior_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".prior_embedder", T + 1 # +1 for pad embed
        )
        
        self.nll_type_prior = torch.zeros((1))
        
    def prepare_job(self, job: "Job", **kwargs):
        super().prepare_job(job, **kwargs)
        self.base_embedder.prepare_job(job, **kwargs)
        #self.prior_embedder.prepare_job(job, **kwargs)
        
        if isinstance(job, TrainingJob):
            # use Modified Differential Multiplier Method for regularization loss
            max_prior_nll_constraint = mdmm.MaxConstraint(
                lambda: self.nll_type_prior,
                self.get_option("nll_max_threshold"),
                scale = self.get_option("nll_max_scale"), 
                damping = self.get_option("nll_max_damping")
            )
            self.mdmm_module = mdmm.MDMM([max_prior_nll_constraint])
            
            # update optimizer
            lambdas = [max_prior_nll_constraint.lmbda]
            slacks = [max_prior_nll_constraint.slack]
            
            lr = next(g['lr'] for g in job.optimizer.param_groups if g['name'] == 'default')
            job.optimizer.add_param_group({'params': lambdas, 'lr': -lr})
            job.optimizer.add_param_group({'params': slacks, 'lr': lr})
        
        # trace the regularization loss
        def trace_regularization_loss(job):
            job.current_trace["batch"]["prior_nll"] = self.nll_type_prior.item()
            if isinstance(job, TrainingJob):
                job.current_trace["batch"]["prior_nll_lambda"] = max_prior_nll_constraint.lmbda.item()

        from kge.job import TrainingOrEvaluationJob
        if isinstance(job, TrainingOrEvaluationJob):
            job.pre_batch_hooks.append(trace_regularization_loss)
        
    def embed(self, indexes):
        return self.base_embedder.embed(indexes)

    def embed_all(self):
        return self.base_embedder.embed_all()

    def _calc_prior_loss(self, indexes):
        types = self.entity_types[indexes]              # B x 2 x T
        
        embeds = self.base_embedder.embed(indexes)      # B x 2 x D
        embeds = embeds.unsqueeze(2)                    # B x 2 x 1 x D
        shape = list(types.shape) + [embeds.shape[3]]
        embeds = embeds.expand(shape)                   # B x 2 x T x D
        
        log_pdf = self.prior_embedder.log_pdf(embeds, types)
        log_pdf[types == self.PADDING_IDX] = 0
        
        self.nll_type_prior = -log_pdf.sum(2).mean()
        
        #c = 1 #random.randint(0, types.shape[0])
        #false_types = torch.cat((types[c:,:,:], types[:c,:,:]), dim=0)
        
        #log_pdf = self.prior_embedder.log_pdf(embeds, false_types)
        #log_pdf[false_types == self.PADDING_IDX] = 0
        
        #self.nll_type_prior += log_pdf.sum(2).mean()

    def penalty(self, **kwargs):
        terms = super().penalty(**kwargs)
        terms += self.base_embedder.penalty(**kwargs)
        terms += self.prior_embedder.penalty(**kwargs)
        
        indexes = kwargs['indexes']                     # B x 2
        self._calc_prior_loss(indexes)
        
        terms += [ ( 
            f"{self.configuration_key}.type_prior_nll", 
            self.mdmm_module(torch.zeros((1), device=self.device)).value
        ) ]
        return terms
    

