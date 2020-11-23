import torch
import torch.nn.functional as F

from kge.model import KgeEmbedder


class MultipleEmbedder(KgeEmbedder):
    """ """

    def __init__(
        self, config, dataset, configuration_key, 
        vocab_size, init_for_load_only=False
    ):

        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )

        dimension = self.get_option("dim")
        self.nr_embeds = self.get_option("nr_embeds")

        lookup_dim = dimension * self.nr_embeds

        config.set(
            self.configuration_key + ".base_embedder.dim",
            lookup_dim
        )

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




    def _embed(self, embeddings):
        if self.dropout > 0:
            embeddings = torch.nn.functional.dropout(
                embeddings, p=self.dropout, training=self.training
            )
        B, _ = embeddings.shape
        return embeddings.reshape(B, -1, self.nr_embeds)

    def embed(self, indexes):
        return self._embed(self.base_embedder.embed(indexes))

    def embed_all(self):
        return self._embed(self.base_embedder.embed_all())

