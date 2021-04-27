from typing import Optional, Tuple
import random

import torch
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn import Linear, MultiheadAttention

from kge.model import KgeEmbedder
from kge.job.train import TrainingJob

from sem_kge import TypedDataset
from sem_kge import misc

import mdmm

class TypeAttentiveEmbedder(KgeEmbedder):
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
        if dim == -1:
            dim = self.get_option("base_embedder.dim")

        config.set(self.configuration_key + ".base_embedder.dim", dim)
        config.set(self.configuration_key + ".type_embedder.dim", dim)

        mi_min = self.get_option('mi_min_threshold')
        self.mi_loss = mi_min > 0
        self.mi_min_threshold = mi_min
        
        self.vocab_size = vocab_size
        self.dropout = self.get_option("dropout")
        self.device = self.config.get("job.device")
        self.mutual_info = torch.tensor(0)

        # initialize base_embedder
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

        # construct tensors with type indices
        T_ = max( types_str.count(',') + 1 for types_str in entity_types if types_str is not None )

        self.entity_types = torch.full((N, T_), self.PADDING_IDX, device=self.device)
        for i,types_str in enumerate(entity_types):
            if types_str is None:
                continue
            types = set(int(x) for x in types_str.split(','))
            self.entity_types[i,:len(types)] = torch.tensor(list(types), device=self.device)
        self.type_padding = self.entity_types == self.PADDING_IDX
        
        # initialize type embedder
        if self.configuration_key + ".type_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".type_embedder.type",
                self.get_option("type_embedder.type"),
            )
        self.type_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".type_embedder", T + 1 # +1 for pad embed
        )

        nhead = self.get_option("attn_nhead")
        self.self_attn = MultiheadAttention(dim, nhead, dropout=self.dropout)
        
        self.mi_proj = Linear(dim, dim)
        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.eye(dim), requires_grad=False)
        
        # init dummy module, in case we wish to load parameters from file.
        self.init_mdmm_module()
        

    def init_mdmm_module(self):
        self.mdmm_module = mdmm.MDMM([ mdmm.MinConstraint(
            lambda: self.mutual_info,
            self.mi_min_threshold,
            scale = self.get_option("mi_min_scale"),
            damping = self.get_option("mi_min_damping")
        )])

    def prepare_job(self, job: "Job", **kwargs):
        super().prepare_job(job, **kwargs)
        self.base_embedder.prepare_job(job, **kwargs)
        self.type_embedder.prepare_job(job, **kwargs)
        
        if self.mi_loss and isinstance(job, TrainingJob):
            self.init_mdmm_module()
            misc.add_constraints_to_job(job, self.mdmm_module)
        
        # trace the regularization loss
        def trace_regularization_loss(job):
            key = f"{self.configuration_key}.mi"
            job.current_trace["batch"][key] = self.mutual_info.item()
            if self.mi_loss and isinstance(job, TrainingJob):
                job.current_trace["batch"][f"{key}_lambda"] = self.mdmm_module[0].lmbda.item()

        from kge.job import TrainingOrEvaluationJob
        if isinstance(job, TrainingOrEvaluationJob):
            job.pre_batch_hooks.append(trace_regularization_loss)

    def _embed(self, embeds, type_embeds, type_padding_mask, return_weights=False):
        query = embeds.unsqueeze(0)                  #    1 x B x D
        key = torch.cat((query, type_embeds), dim=0) # T'+1 x B x D
        value = key

        B,T = type_padding_mask.shape
        prepend = torch.zeros((B,1), device=self.device).bool()
        mask = torch.cat((prepend, type_padding_mask), dim=1)

        attn_output, attn_weights = self.self_attn(query, key, value, key_padding_mask=mask)
        attn_output = attn_output.squeeze()          # 1 x B x D, B x 1 x T'+1

        return attn_output if not return_weights else (attn_output, attn_weights)

    def embed(self, indexes):
        indexes = indexes.long()
        entity_embeds = self.base_embedder.embed(indexes)
        types = self.entity_types[indexes]
        type_embeds = self.type_embedder.embed(types.T)
        type_paddin = self.type_padding[indexes]
        return self._embed(entity_embeds, type_embeds, type_paddin)

    def embed_all(self):
        embeds = self.base_embedder.embed_all()
        type_embeds = self.type_embedder.embed(self.entity_types.T)
        return self._embed(embeds, type_embeds, self.type_padding)

    def calc_mi(self, indexes, type_embeds, embeds):
        """
        type_embeds [ T' x B x D ]
        embeds           [ B x D ]
        """
        T_, B, D = type_embeds.shape
        embeds = embeds.view(1, B, 1, D)                        # 1  x B x 1 x D

        def negative_sample_types(positives):
            T = self.dataset.num_types()
            
            all_indices = torch.arange(T, device=self.device).unsqueeze(1).expand(T,B)
            type_indices = (~self.entity_type_set[indexes].T) * all_indices
            type_indices[self.entity_type_set[indexes].T] = self.PADDING_IDX
            
            idxs = list(range(T))
            random.shuffle(idxs)
            shuffled = type_indices[idxs]
            subset = shuffled[:T_]
            subset[positives==self.PADDING_IDX] = self.PADDING_IDX
            return subset

        def calc(type_indices, type_paddin):
            type_embeds = self.type_embedder.embed(type_indices)
            type_embeds = type_embeds.unsqueeze(-1)           # T' x B x D x 1

            proj_embeds = self.mi_proj(embeds)
            mi = torch.matmul(proj_embeds, type_embeds).squeeze()  # T' x B
            mi[type_paddin] = 0
            nr_types = (~type_paddin).sum(dim=0)
            mi = mi.sum(dim=0) / nr_types                     # B
            mi[nr_types==0] = 0
            return mi.mean()

        types = self.entity_types[indexes].T
        neg_types = negative_sample_types(types)

        type_paddin = self.type_padding[indexes].T
        neg_type_paddin = neg_types == self.PADDING_IDX

        self.mutual_info  = calc(types, type_paddin)
        self.mutual_info -= calc(neg_types, neg_type_paddin)
        
    def penalty(self, **kwargs):
        terms = super().penalty(**kwargs)
        terms += self.base_embedder.penalty(**kwargs)
        terms += self.type_embedder.penalty(**kwargs)
        
        indexes = kwargs['indexes'].view(-1)               # B x 2
        
        entity_embeds = self.base_embedder.embed(indexes)
        types = self.entity_types[indexes]
        type_embeds = self.type_embedder.embed(types.T)
        type_paddin = self.type_padding[indexes]
        embeds = self._embed(entity_embeds, type_embeds, type_paddin).squeeze()
        self.calc_mi(indexes, type_embeds, embeds)
        if self.mi_loss:
            terms += [ ( 
                f"{self.configuration_key}.type_mi", 
                self.mdmm_module(torch.zeros((1), device=self.device)).value
            ) ]

        return terms
