import torch

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

        self.entropy_mode = self.get_option('entropy_mode')
        self.entropy_threshold = self.get_option('entropy_threshold')
        self.entropy = torch.tensor(0)
        
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
        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.eye(dim), requires_grad=False)
        
        # init dummy module, in case we wish to load parameters from file.
        self.init_mdmm_module()
        
        self.add_entity_to_keyvalue = self.get_option("add_entity_to_keyvalue")
        
    def init_mdmm_module(self):
        if self.entropy_mode != "off":
            cls = mdmm.MinConstraint if self.entropy_mode == "min" else mdmm.MaxConstraint
            
            self.mdmm_module = mdmm.MDMM([ cls(
                lambda: self.entropy,
                self.entropy_threshold,
                scale=self.get_option("entropy_scale"),
                damping=self.get_option("entropy_damping")
            )])

    def prepare_job(self, job: "Job", **kwargs):
        super().prepare_job(job, **kwargs)
        self.base_embedder.prepare_job(job, **kwargs)
        self.type_embedder.prepare_job(job, **kwargs)
        
        if self.entropy_mode != 'off' and isinstance(job, TrainingJob):
            self.init_mdmm_module()
            misc.add_constraints_to_job(job, self.mdmm_module)

        # trace the regularization loss
        def trace_loss(job):
            key = f"{self.configuration_key}.entropy"
            job.current_trace["batch"][key] = self.entropy.item()
            if self.entropy_mode != 'off' and isinstance(job, TrainingJob):
                job.current_trace["batch"][f"{key}_lambda"] = self.mdmm_module[0].lmbda.item()

        from kge.job import TrainingOrEvaluationJob
        if isinstance(job, TrainingOrEvaluationJob):
            job.pre_batch_hooks.append(trace_loss)

    def _embed(self, embeds, type_embeds, type_padding_mask, return_weights=False):
    
        query = embeds.unsqueeze(0)                         # 1 x B x D
        if self.add_entity_to_keyvalue:
            key = torch.cat((query, type_embeds), dim=0)    # T'+1 x B x D
        else:
            key = type_embeds                               # T' x B x D
        value = key
        
        B, T = type_padding_mask.shape
        if self.add_entity_to_keyvalue:
            prepend = torch.zeros((B, 1), device=self.device).bool()
            mask = torch.cat((prepend, type_padding_mask), dim=1)
        else:
            nr_types = (~type_padding_mask).sum(dim=1)
            mask = type_padding_mask
            
            # set at least one to False
            mask[nr_types == 0, 0] = False
        
        attn_output, attn_weights = self.self_attn(query, key, value, key_padding_mask=mask)
        attn_output = attn_output.squeeze()          # 1 x B x D, B x 1 x T'[+1]

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

    def calc_entropy(self, indexes):
        indexes = indexes.long()
        entity_embeds = self.base_embedder.embed(indexes)
        types = self.entity_types[indexes]
        type_embeds = self.type_embedder.embed(types.T)
        type_padding = self.type_padding[indexes]
        out, weights = self._embed(entity_embeds, type_embeds, type_padding, return_weights=True)

        # calculate entropy of weight distributions
        from torch.distributions.categorical import Categorical
        entropy = Categorical(probs=weights.squeeze()).entropy()

        # calculate number of outcomes
        nr_outcomes = (~type_padding).sum(dim=1)
        if self.add_entity_to_keyvalue:
            nr_outcomes += 1

        # normalize to metric entropy (while preventing divide by zero with just one outcome)
        entropy[nr_outcomes > 1] /= torch.log(nr_outcomes.float())[nr_outcomes > 1]

        self.entropy = entropy.mean()
        
    def penalty(self, **kwargs):
        terms = super().penalty(**kwargs)
        terms += self.base_embedder.penalty(**kwargs)

        indexes = kwargs['indexes'].view(-1)               # B x 2

        type_indexes = self.entity_types[indexes]
        type_indexes = type_indexes[type_indexes != self.PADDING_IDX]
        type_kwargs = dict(**kwargs)
        type_kwargs['indexes'] = type_indexes
        terms += self.type_embedder.penalty(**type_kwargs)

        if self.entropy_mode != 'off':
            self.calc_entropy(indexes)
            terms += [(
                f"{self.configuration_key}.entropy", 
                self.mdmm_module(torch.zeros((1), device=self.device)).value
            )]

        return terms
