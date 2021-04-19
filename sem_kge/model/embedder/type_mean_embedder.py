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


class TypeMeanEmbedder(KgeEmbedder):
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
            
        config.set(self.configuration_key + ".entity_embedder.dim", dim)
        config.set(self.configuration_key + ".type_embedder.dim", dim)

        self.vocab_size = vocab_size
        self.device = self.config.get("job.device")
        self.use_entity_embedder = self.get_option("use_entity_embedder")

        if self.use_entity_embedder:
            # initialize entity_embedder
            if self.configuration_key + ".entity_embedder.type" not in config.options:
                config.set(
                    self.configuration_key + ".entity_embedder.type",
                    self.get_option("entity_embedder.type"),
                )
            self.entity_embedder = KgeEmbedder.create(
                config, dataset, self.configuration_key + ".entity_embedder", vocab_size 
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
        

    def _embed(self, type_embeds, type_padding_mask, entity_embeds):
        nr_types = (~type_padding_mask).sum(dim=1).unsqueeze(-1)
        embeds = type_embeds.sum(dim=0) / nr_types
        
        if entity_embeds != None:
            embeds += entity_embeds
          
        return embeds

    def embed(self, indexes):
        indexes = indexes.long()
        types = self.entity_types[indexes]
        type_embeds = self.type_embedder.embed(types.T)
        type_paddin = self.type_padding[indexes]
        
        entity_embeds = None
        if self.use_entity_embedder and (not self.training or random.uniform(0,1) > 0.5):
            entity_embeds = self.entity_embedder.embed(indexes)

        return self._embed(type_embeds, type_paddin, entity_embeds)

    def embed_all(self):
        type_embeds = self.type_embedder.embed(self.entity_types.T)
        
        entity_embeds = None
        if self.use_entity_embedder and (not self.training or random.uniform(0,1) > 0.5):
            entity_embeds = self.entity_embedder.embed_all()
        
        return self._embed(type_embeds, self.type_padding, entity_embeds)
        
    def penalty(self, **kwargs):
        terms = super().penalty(**kwargs)
        terms += self.type_embedder.penalty(**kwargs)
        if self.use_entity_embedder:
            terms += self.entity_embedder.penalty(**kwargs)
        return terms
