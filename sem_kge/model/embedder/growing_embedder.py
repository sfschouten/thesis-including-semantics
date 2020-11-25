import torch
import torch.nn.functional as F

from kge.model import KgeEmbedder
from sem_kge.model.embedder import MultipleEmbedder

class GrowingMultipleEmbedder(MultipleEmbedder):
    """ Embedder that associates each entity with multiple embeddings, and allows
    the number of embeddings for each entity to grow according to a Chinese 
    Restaurant Process. It was presented in the paper "TransT: Type-Based 
    Multiple Embedding Representations for Knowledge Graph Completion" by 
    Ma et al.   """

    def __init__(
        self, config, dataset, configuration_key, 
        vocab_size, init_for_load_only=False
    ):

        super().__init__(
            config, dataset, configuration_key, vocab_size, 
            init_for_load_only=init_for_load_only
        )

        self.device = config.get("job.device")
        self.crp_beta = self.get_option("crp_beta")
        self.max_nr_semantics = self.get_option("nr_embeds")
        self.vocab_size = vocab_size

    def initialize_semantics(self, types_tensor):
        E, S = types_tensor.shape
        M = self.max_nr_semantics

        self.nr_semantics = torch.ones((E), device=self.device).long()
        self.semantics = torch.zeros((E,S,M), device=self.device).bool() 

        # The semantics of the first vector of each entity is simply its own type set.
        self.semantics[:,:,0] = types_tensor
        

    def update(self, idxs, embeds, types, loglikelihood, similarity_fn):
        h_idx, r_idx, t_idx = idxs
        h_emb, r_emb, t_emb = embeds
        h_typ, r_typ, t_typ = types
        r_typ_h, r_typ_t = r_typ
        
        # The following probability represents the first term in formula 9 in the TransT paper.
        # Because the probability is a product, we can sample from the two terms separately.
        prob_new_component = self.crp_beta * torch.exp(-r_emb.sum(dim=1))
        prob_new = prob_new_component / (prob_new_component + torch.exp(loglikelihood))

        dist = torch.distributions.bernoulli.Bernoulli(prob_new)

        # TODO: decide wether to replace this with sampling head/tail randomly 
        #           (the original implementation does this)
        for e_idx, e_typ, r_typ_ in ((h_idx,h_typ,r_typ_h),(t_idx,t_typ,r_typ_t)):

            full = self.nr_semantics[e_idx] >= self.max_nr_semantics

            candidates = dist.sample().bool() & ~full
            candidate_idx = candidates * e_idx
            candidate_idx = candidate_idx[candidates]                   # C
            
            if len(candidate_idx) is 0:
                # There are no candidates so we move on.
                continue

            candidate_nr_semantics = self.nr_semantics[candidate_idx]   # C
            candidate_semantics = self.semantics[candidate_idx,:,:]     # C x S x M

            # The following code calculates the right side of formula 9 in the TransT paper.
            
            # Note that we also calculate similarity with the 'empty' semantics-vectors, 
            # because otherwise we could no longer have batched operations 
            # (different entities have different number of semantics)
            r_typ_max = r_typ_[candidates].unsqueeze(2)
            sim = similarity_fn(candidate_semantics, r_typ_max)         # C x M
            max_sim, _ = sim.max(dim=1)                                 # C

            new_semantics_prob = 1 - max_sim                            # C
            new_dist = torch.distributions.bernoulli.Bernoulli(new_semantics_prob)
            to_be_extended = new_dist.sample().bool()                   # EXT
            to_be_extended_idx = to_be_extended * candidate_idx
            to_be_extended_idx = to_be_extended_idx[to_be_extended]

            # The semantics of the newly added vector is the set of common types
            #   of the relation in the triple.
            self.semantics[                                     \
                    to_be_extended_idx, :,                      \
                    self.nr_semantics[to_be_extended_idx],      \
                ] = r_typ_[candidates][to_be_extended]
            self.nr_semantics[to_be_extended_idx] += 1

    def embed(self, indexes):
        embeddings = self._embed(self.base_embedder.embed(indexes))     # B x D x M
        B, D, M = embeddings.shape

        # mark embeddings that are inactive
        nr_semantics = self.nr_semantics[indexes].unsqueeze(1)          # B x 1
        nr_semantics = nr_semantics.expand(-1, M)                       # B x M

        idxs = torch.arange(M, device=self.device) \
                .unsqueeze(0).expand(B, -1)                             # B x M
        inactive = idxs >= nr_semantics 
        inactive = inactive.unsqueeze(1).expand(-1, D, -1)              # B x D x M
      
        embeddings[inactive] = -1
        return embeddings


    def embed_all(self):
        embeddings = self._embed(self.base_embedder.embed_all())
        E, D, M = embeddings.shape

        # mark embeddings that are inactive
        nr_semantics = self.nr_semantics.unsqeeze(1)
        nr_semantics = nr_semantics.expand(-1, M)

        idxs = torch.arange(M, device=self.device) \
                .unsqueeze(0).expand(E, -1)
        inactive = idxs >= nr_semantics
        inactive = inactive.unsqueeze(1).expand(-1, D, -1)

        embeddings[inactive] = -1
        return embeddings
