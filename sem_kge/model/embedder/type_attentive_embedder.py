import torch
import torch.nn.functional as F

from torch.nn import MultiheadAttention

from kge.model import KgeEmbedder

from sem_kge import TypedDataset


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

        self.vocab_size = vocab_size
        self.dropout = self.get_option("dropout")
        self.device = self.config.get("job.device")       

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

        # construct tensor with type indices
        T_ = max( types_str.count(',') + 1 for types_str in entity_types if types_str is not None )
        self.entity_types = torch.full((N, T_), self.PADDING_IDX, device=self.device)
        for i,types_str in enumerate(entity_types):
            if types_str is None:
                continue
            for j,t in enumerate(types_str.split(',')):
                self.entity_types[i,j] = int(t)

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
        embeds = self.base_embedder.embed(indexes)
        types = self.entity_types[indexes]
        type_embeds = self.type_embedder.embed(types.T)
        type_paddin = self.type_padding[indexes]
        return self._embed(embeds, type_embeds, type_paddin)

    def embed_all(self):
        embeds = self.base_embedder.embed_all()
        type_embeds = self.type_embedder.embed(self.entity_types.T)
        return self._embed(embeds, type_embeds, self.type_padding)
