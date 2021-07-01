import torch
import torch.nn.functional as F

from torch.nn.functional import cosine_similarity as cos_sim

from torch.distributions.categorical import Categorical

from kge.job import EvaluationJob, Job
from kge.config import Configurable

from sem_kge.model import TypeAttentiveEmbedder

import pandas as pd
import seaborn as sns
import numpy as np

import math


SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]


HUMAN_READABLE = {
    "nr_outcomes" : "Nr. Outcomes",
}

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
        self.chunk_size = self.config.get("type_attentive_experiment.chunk_size")
        

    def _perform_chunked(self, operation):
        """ Performs `operation` on entities in chunks. """
        num_entities = self.dataset.num_entities()
        
        # calculate scores in chunks to not have the complete score matrix in memory
        # a chunk here represents a range of entity_values to score against
        chunk_size = self.chunk_size if self.chunk_size > -1 else num_entities
        
        nr_of_chunks = math.ceil(num_entities / chunk_size)
        
        all_results = []
        for chunk_number in range(nr_of_chunks):
            chunk_start = chunk_size * chunk_number
            chunk_end = min(chunk_size * (chunk_number + 1), num_entities)
            
            self.config.log(f"Chunk {chunk_number+1} / {nr_of_chunks}")
            
            indexes = torch.arange(chunk_start, chunk_end, device=self.device)
            
            chunk_results = operation(indexes, nr_of_chunks, chunk_start)
            all_results.append(chunk_results)
            
        return { 
            key : torch.cat(tuple(r[key] for r in all_results))
            for key in all_results[0].keys()
        }

    def _add_results(self, new_results, trace_dict):
        for key,value in new_results.items():
            trace_dict[key] = {
                i : v.item() for i,v in enumerate(value) 
            }
            
    def _add_average(self, new_results, trace_dict):
        for key,value in new_results.items():
            mask = ~( value.isnan().logical_or(value.isinf()) )
            trace_dict[f'{key}_avg'] = value[mask].mean().item()

    def _generate_plot(self, new_results, name):
        sns.set_theme(style="whitegrid")
        
        for key, value in new_results.items():
            mask = ~( value.isnan().logical_or(value.isinf()) )
            value = value[mask].tolist()
            
            g = sns.violinplot(data=value)
            g.set(xticklabels=[])
            if key in HUMAN_READABLE:
                g.set_ylabel(HUMAN_READABLE[key])
            fig = g.get_figure()
            fig.savefig(f"{name}_{key}.png")
            fig.clf()

    def _calc_attn_distribution(self, embedder, results_dict):

        def _perform_calculation(indexes, nr_of_chunks, chunk_start):
            e_embeds = embedder.base_embedder.embed(indexes)

            types = embedder.entity_types[indexes].T    # T_ x E
            t_embeds = embedder.type_embedder.embed(types)
            t_paddin = embedder.type_padding[indexes]
            _, attn_w = embedder._embed(e_embeds, t_embeds, t_paddin, True)
                                                        # E x 1 x T'+1
                                                    
            nr_outcomes = (~t_paddin).sum(dim=1).float()
            entropy = Categorical(probs = attn_w.squeeze()).entropy()

            self.config.log(f"add_entity_to_keyvalue: {embedder.get_option('add_entity_to_keyvalue')}")
            if hasattr(embedder, 'add_entity_to_keyvalue') and embedder.add_entity_to_keyvalue:
                nr_outcomes += 1
                self.config.log("Increasing outcomes by 1, because this embedder also adds entity to key/value.")
                
                # calculate the attention paid to all types for each entity
                t_attn = attn_w[:,:,1:]
                t_attn_sum = t_attn.sum(dim=2).squeeze()
                
            else:
                t_attn_sum = attn_w.squeeze().sum(dim=1)
       
            entropy /= torch.log(nr_outcomes)
            
            return dict(
                type_total_attn=t_attn_sum,
                metric_entropy=entropy,
                nr_outcomes=nr_outcomes
            )

        self.config.log("\nPerforming analysis of attention distribution.")

        calc_results = self._perform_chunked(_perform_calculation)

        if self.trace_entity_level:
            self._add_results(calc_results, results_dict)

        df = pd.DataFrame({ key : tensor.tolist() for key,tensor in calc_results.items() })
        df['log_nr_outcomes'] = np.log(df['nr_outcomes'])
        
        g = sns.scatterplot(x='nr_outcomes', y='metric_entropy', data=df, size=0.05, legend=False)
        g.set_xlabel("Nr. Outcomes")
        g.set_ylabel("Metric Entropy")
        fig = g.get_figure()
        fig.savefig(f"scatter_nr_types_vs_metric_entropy.png", bbox_inches="tight")
        fig.clf()
        
        g = sns.scatterplot(x='type_total_attn', y='metric_entropy', data=df, size=0.05, legend=False, hue='log_nr_outcomes')
        fig = g.get_figure()
        fig.savefig(f"scatter_type_total_attn_vs_metric_entropy.png", bbox_inches="tight")
        fig.clf()

        self._add_average(calc_results, results_dict)
        self._generate_plot(calc_results, "attn_distribution")
        

    def _calc_embed_analysis(self, embedder, results_dict):

        def _perform_calculation(indexes, nr_of_chunks, chunk_start):
            
            W_v = embedder.self_attn.in_proj_weight
            _, D = W_v.shape
            W_v = W_v[2*D:,:].T
            
            # type indices and corresponding type-embeddings
            types = embedder.entity_types[indexes].T            # T_ x E
            t_embeds = embedder.type_embedder.embed(types)      # T_ x E x D
            t_embeds = t_embeds @ W_v

            # calculate padding tensor, and number of types
            padding = types == embedder.PADDING_IDX             # T_ x E
            nr_types = (~padding).sum(0, keepdim=True)          # 1  x E

            # set the padding to zero so sum is only non-padding
            t_embeds[padding] = 0
            # calculate mean type-embedding
            t_sum = t_embeds.sum(dim=0, keepdim=True)           # 1 x E x D
            t_mean = t_sum / nr_types.unsqueeze(-1)             # 1 x E x D
            
            # calculate norm of type embeddings
            t_norms = torch.norm(t_mean, dim=1).squeeze()

            # calculate cosine sim between type-embeddings and their mean
            cos = cos_sim(t_embeds, t_mean, 2)                  # T_ x E x 1
            cos[padding] = 0                                    # T_ x E x 1

            # calculate conicity (average cosine sim)
            conicity = cos.sum(dim=0).div(nr_types).squeeze()   # E

            # calculate cosine sim between entity and avg type
            e_embeds = embedder.base_embedder.embed(indexes)    # E x D
            e_embeds = e_embeds @ W_v
            
            # calcaulte cosine sim between mean type and entity embeddings
            atm = cos_sim(t_mean.squeeze(), e_embeds, 1).squeeze()
            
            # calculate norms of entity embeddings
            e_norms = torch.norm(e_embeds, dim=1)
            
            return dict(
                type_conicity = conicity, 
                entity_type_cosine_sim = atm,
                type_norms = t_norms,
                entity_norms = e_norms,
            )

        self.config.log("\nPerforming analysis of embeddings.")

        calc_results = self._perform_chunked(_perform_calculation)

        if self.trace_entity_level:
            self._add_results(calc_results, results_dict)

        self._add_average(calc_results, results_dict)
        self._generate_plot(calc_results, "embedding_analysis")


    def _evaluate(self):

        with torch.no_grad():

            # create initial trace entry
            self.current_trace["epoch"] = dict(
                type="type_attentive_experiment",
                scope="epoch",
            )

            for slot, embedder in self.embedders.items():

                # results dictionary for this embedder
                embedder_results = dict()

                self._calc_attn_distribution(embedder, embedder_results)
                self._calc_embed_analysis(embedder, embedder_results)

                self.current_trace["epoch"].update({
                    f"{SLOT_STR[slot]}_embedder" : embedder_results
                })



