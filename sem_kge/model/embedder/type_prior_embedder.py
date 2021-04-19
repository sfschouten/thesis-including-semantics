import torch
import torch.nn.functional as F

from torch.nn import MultiheadAttention

from kge.model import KgeEmbedder
from kge.job.train import TrainingJob

from sem_kge import TypedDataset
from sem_kge import misc

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
        
        self.entity_type_set = dataset.index("entity_type_set").to(self.device)
        
        T_ = max( types_str.count(',') + 1 for types_str in entity_types if types_str is not None )

        self.entity_types = torch.full((N, T_), self.PADDING_IDX, device=self.device)
        
        self.avg_nr_types = 0
        for i,types_str in enumerate(entity_types):
            if types_str is None:
                continue
            types = types_str.split(',')
            self.avg_nr_types += len(types)
            for j,t in enumerate(types):
                self.entity_types[i,j] = int(t)
        self.avg_nr_types /= N

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
        
        # wether we average or sum the logprobs of types for a given entity
        # the latter results prioritization of entities with many types.
        self.aggr_fun_types = self.check_option('aggr_fun_types', ['mean', 'sum'])
        self.nll_max_threshold = self.get_option("nll_max_threshold")

        self.nll_type_prior = torch.tensor(0)
        
    def prepare_job(self, job: "Job", **kwargs):
        super().prepare_job(job, **kwargs)
        self.base_embedder.prepare_job(job, **kwargs)
        self.prior_embedder.prepare_job(job, **kwargs)
        
        if isinstance(job, TrainingJob):
            # use Modified Differential Multiplier Method for regularization loss
            max_prior_nll_constraint = mdmm.MaxConstraint(
                lambda: self.nll_type_prior,
                self.nll_max_threshold,
                scale = self.get_option("nll_max_scale"),
                damping = self.get_option("nll_max_damping")
            )
            self.mdmm_module = mdmm.MDMM([max_prior_nll_constraint])
            misc.add_constraints_to_job(job, self.mdmm_module)
        
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
        types = self.entity_types[indexes]              # B x 2 x T_
        B, _, T_ = types.shape
        
        def negative_sample_types(positives):
            T = self.dataset.num_types()
            indices = indexes.view(2*B)
            all_indices = torch.arange(T, device=self.device).unsqueeze(1).expand(T,2*B)
            type_indices = (~self.entity_type_set[indices].T) * all_indices
            type_indices[self.entity_type_set[indices].T] = self.PADDING_IDX
            
            idxs = list(range(T))
            random.shuffle(idxs)
            shuffled = type_indices[idxs]
            subset = shuffled[:T_]
            subset[positives==self.PADDING_IDX] = self.PADDING_IDX
            return subset

        def calc(embeds, type_indices):
            shape = list(type_indices.shape) + [embeds.shape[3]]
            embeds = embeds.expand(shape)                   # B x 2 x T x D

            padding = type_indices == self.PADDING_IDX

            log_pdf = self.prior_embedder.log_pdf(embeds, type_indices)
            log_pdf[padding] = 0                            # B x 2 X T
            nll = -log_pdf.sum(2)

            if self.aggr_fun_types == 'mean':
                nr_types = (~padding).sum(2)                # B x 2
                nr_types[nr_types==0] = 1     # prevent divide_by_zero 
                                              # (where nr_types is zero, so will nll)
                nll = nll.div(nr_types)
            return nll.mean()
        
        neg_types = negative_sample_types(types.view(2*B,T_).T).T.view(B, 2, T_)
        
        embeds = self.base_embedder.embed(indexes)      # B x 2 x D
        embeds = embeds.unsqueeze(2)                    # B x 2 x 1 x D
        
        self.nll_type_prior  = calc(embeds, types)
        self.nll_type_prior -= calc(embeds, neg_types)

    def penalty(self, **kwargs):
        terms = super().penalty(**kwargs)
        terms += self.base_embedder.penalty(**kwargs)
        
        # add prior_embedder penalty terms which requires setting 
        # the 'indexes' to the indices of the types.
        indexes = kwargs['indexes']                     # B x 2
        embeds = self.base_embedder.embed(indexes)
        type_indexes = self.entity_types[indexes.view(-1)]
        type_indexes = type_indexes[type_indexes!=self.PADDING_IDX]
        prior_kwargs = dict(points=embeds, **kwargs)
        prior_kwargs['indexes'] = type_indexes
        terms += self.prior_embedder.penalty(**prior_kwargs)
        
        self._calc_prior_loss(indexes)
        
        terms += [ ( 
            f"{self.configuration_key}.type_prior_nll", 
            self.mdmm_module(torch.zeros((1), device=self.device)).value
        ) ]
        return terms
    

