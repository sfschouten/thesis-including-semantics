import torch
import torch.nn.functional as F

from kge.job import EvaluationJob, Job
from kge.config import Configurable

from sem_kge.model import TypeAttentiveEmbedder

import seaborn as sns


SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]

class TypeAttentiveExperimentJob(EvaluationJob):
    """
    Job to analyse the attention weights of the type-attentive embedders in a
    given model.
    """

    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)

        embedders = {
            S : model.get_s_embedder(),
            P : model.get_p_embedder(),
            O : model.get_o_embedder()
        }
        is_tae = lambda x: isinstance(x, TypeAttentiveEmbedder)
        self.embedders = { s : e for s,e in embedders.items() if is_tae(e) }

        config_key = "type_attentive_experiment"
        self.trace_entity_level = self.config.get(config_key + ".trace_entity_level")

    def _evaluate(self):

        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type="type_attentive_experiment",
            scope="epoch",
        )

        for slot, embedder in self.embedders.items():

            e_embeds = embedder.base_embedder.embed_all()

            types = embedder.entity_types.T   # T_ x E
            t_embeds = embedder.type_embedder.embed(types)
            t_paddin = embedder.type_padding

            _, attn_w = embedder._embed(e_embeds, t_embeds, t_paddin, True)
                                              # E x 1 x T'+1

            # calculate the attention paid to all types for each entity
            t_attn = attn_w[:,:,1:].squeeze().sum(dim=1)

            # and the average over entities
            t_attn_avg = t_attn.mean()


            # add to trace
            embedder_results = dict(event="attn_computed")

            if self.trace_entity_level:
                embedder_results.update(dict(
                    type_total_attn={ i : w.item() for i,w in enumerate(t_attn) },
                ))

            embedder_results.update(dict(
                type_total_attn_avg=t_attn_avg.item(),
            ))

            self.current_trace["epoch"].update({
                f"{SLOT_STR[slot]}_embedder" : embedder_results
            })
