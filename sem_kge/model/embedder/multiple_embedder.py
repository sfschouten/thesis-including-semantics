import torch
import torch.nn.functional as F

from kge.model import KgeEmbedder


class MultipleEmbedder(KgeEmbedder):
    """ 
    Embedder that assigns multiple embeddings to each entity/relation.
    """

    def __init__(
        self, config, dataset, configuration_key, 
        vocab_size, init_for_load_only=False
    ):

        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )

        self.vocab_size = vocab_size

        # we construct a base embedder with dimensionality of nr_embeds*dim
        dimension = self.get_option("dim")
        self.nr_embeds = self.get_option("nr_embeds")
        base_dim = dimension * self.nr_embeds
        config.set(self.configuration_key + ".base_embedder.dim", base_dim)

        # initialize base_embedder
        if self.configuration_key + ".base_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".base_embedder.type",
                self.get_option("base_embedder.type"),
            )
        self.base_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".base_embedder", vocab_size 
        )

        self.dropout = self.get_option("dropout")

    def get_nr_embeddings(self):
        """ returns a tensor with the number of embeddings for each object in vocabulary """
        device = self.config.get("job.device")
        return torch.ones((self.vocab_size), device=device) * self.nr_embeds

    def _embed(self, embeddings):
        # apply dropout
        if self.dropout > 0:
            embeddings = torch.nn.functional.dropout(
                embeddings, p=self.dropout, training=self.training
            )

        # reshape and return
        B, _ = embeddings.shape
        return embeddings.reshape(B, -1, self.nr_embeds)

    def embed(self, indexes):
        return self._embed(self.base_embedder.embed(indexes))

    def embed_all(self):
        return self._embed(self.base_embedder.embed_all())

