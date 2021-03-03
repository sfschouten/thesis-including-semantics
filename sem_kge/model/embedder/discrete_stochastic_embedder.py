import torch
import torch.nn.functional as F

from kge.model import KgeEmbedder


class DiscreteStochasticEmbedder(KgeEmbedder):
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
        self.device = self.config.get("job.device")
        self.stochastic = self.get_option("stochastic")
        distribution = self.get_option("distribution")
        if distribution == "relaxed_one_hot_categorical":
            self.distribution = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical
        else:
            raise Exception()
        
        # initialize the distribution weights
        N = dataset.num_entities()
        M = self.nr_embeds
        nr_embeddings = self.get_nr_embeddings()
        idxs = torch.arange(M, device=self.device).unsqueeze(0).expand(N,-1)
        weights = (idxs < nr_embeddings.unsqueeze(1).expand(-1,M)).float()
        weights /= nr_embeddings.unsqueeze(1)
        self.weights = torch.nn.Parameter(weights)

    def get_nr_embeddings(self):
        """ returns a tensor with the number of embeddings for each object in vocabulary """
        return torch.ones((self.vocab_size), device=self.device) * self.nr_embeds

    def _embed(self, embeddings):
        # apply dropout
        if self.dropout > 0:
            embeddings = torch.nn.functional.dropout(
                embeddings, p=self.dropout, training=self.training
            )

        # reshape and return
        B, _ = embeddings.shape
        return embeddings.reshape(B, -1, self.nr_embeds)

    def _softmax_weights(self, weights, indexes=None):
        return F.softmax(weights, dim=1)

    def _sample_or_softmax(self, w, e, indexes=None):
        w = self._softmax_weights(w, indexes)
        if self.stochastic and self.training:
            dist = self.distribution(0.1, probs=w)
            idx = dist.rsample()
            idx = idx.argmax(dim=1)                                     # B
            idx = F.one_hot(idx, num_classes=self.nr_embeds).bool()     # B x M
            idx = idx.unsqueeze(1).expand_as(e)                         # B x 1 x M
            e = e[idx].view(e.shape[0],-1).unsqueeze(2)
            w = torch.ones((w.shape[0],1), device=self.device)
            
        return e,w

    def embed(self, indexes):
        e = self._embed(self.base_embedder.embed(indexes))
        w = self.weights[indexes]
        return self._sample_or_softmax(w, e, indexes)

    def embed_all(self):
        e = self._embed(self.base_embedder.embed_all())
        w = self.weights
        return self._sample_or_softmax(w, e)

