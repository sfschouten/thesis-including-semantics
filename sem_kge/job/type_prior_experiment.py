import torch
import torch.nn.functional as F

from torch.nn.functional import cosine_similarity as cos_sim
from torch.distributions.kl import kl_divergence

from kge.job import EvaluationJob, Job
from kge.config import Configurable

from sem_kge import TypedDataset
from sem_kge.model import TypePriorEmbedder

import pandas as pd
import seaborn as sns
import numpy as np

import math


SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]


HUMAN_READABLE = {
    "prior_lpdf_diff" : "Log PDF Pos - Neg",
    "prior_lpdf_pos" : "Log PDF Pos",
    "prior_lpdf_neg" : "Log PDF Neg",
    "nr_outcomes" : "Nr. Outcomes",
}

class TypePriorExperimentJob(EvaluationJob):
    """
    """

    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)

        self.dataset = TypedDataset.create(self.dataset)

        embedders = {
            S : model.get_s_embedder(),
            P : model.get_p_embedder(),
            O : model.get_o_embedder()
        }
        is_tpe = lambda x: isinstance(x, TypePriorEmbedder)
        self.embedders = { s : e for s,e in embedders.items() if is_tpe(e) }

        config_key = "type_prior_experiment"
        self.trace_entity_level = self.config.get(config_key + ".trace_entity_level")
        self.chunk_size = self.config.get(config_key + ".chunk_size")
        self.prior_lpdf_nr_trials = self.config.get(config_key + ".prior_lpdf_nr_trials")

        self.jaccard_indices = self.dataset.index("type_jaccards").to(self.device)
    
        #shape = self.jaccard_indices.shape
        #temp = ~(torch.eye(shape[0], device=self.device).bool())
        #print()
        #print( ((self.jaccard_indices > 0.1) & temp).sum() )
        #print( ((self.jaccard_indices > 0.2) & temp).sum() )
        #print( ((self.jaccard_indices > 0.3) & temp).sum() )
        #print( ((self.jaccard_indices > 0.4) & temp).sum() )
        #print( ((self.jaccard_indices > 0.5) & temp).sum() )
        #exit()
        

    def _perform_chunked(self, operation, mode="entities"):
        """ Performs `operation` on entities/types in chunks. """
        if mode == "entities":
            num_elements = self.dataset.num_entities()
        elif mode == "types":
            num_elements = self.dataset.mum_types()
            
        
        # calculate scores in chunks to not have the complete score matrix in memory
        # a chunk here represents a range of entity_values to score against
        chunk_size = self.chunk_size if self.chunk_size > -1 else num_elements
        
        nr_of_chunks = math.ceil(num_elements / chunk_size)
        
        all_results = []
        for chunk_number in range(nr_of_chunks):
            chunk_start = chunk_size * chunk_number
            chunk_end = min(chunk_size * (chunk_number + 1), num_elements)
            
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
            trace_dict[f'{key}_avg'] = value[~value.isnan()].mean().item()
    
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

    

    def _calc_embed_analysis(self, embedder, results_dict):

        def _perform_calculation(indexes, nr_of_chunks, chunk_start):
            
            indexes = indexes.unsqueeze(-1)
            
            NR_SAMPLES = self.prior_lpdf_nr_trials
            
            pos, neg = 0, 0
            for _ in range(NR_SAMPLES):
                pos_, neg_ = embedder._calc_prior_loss(indexes)
                pos += -pos_.squeeze()
                neg += -neg_.squeeze()
                
            pos /= NR_SAMPLES
            neg /= NR_SAMPLES
            
            result = dict(
                prior_lpdf_pos = pos,
                prior_lpdf_neg = neg,
                prior_lpdf_diff = pos - neg
            )
            
            # type indices and corresponding type-distributions
            #types = embedder.entity_types[indexes].T            # T_ x E
            
            
            
            #t_means = embedder.prior_embedder.loc_embedder.embed(types)      # T_ x E x D
            #t_stdvs = embedder.prior_embedder.scale_embedder.embed(types)

            # calculate padding tensor, and number of types
            #padding = types == embedder.PADDING_IDX             # T_ x E
            #nr_types = (~padding).sum(0, keepdim=True)          # 1  x E

            # set the padding to zero so sum is only non-padding
            #t_embeds[padding] = 0
            # calculate mean type-embedding
            #t_sum = t_embeds.sum(dim=0, keepdim=True)           # 1 x E x D
            #t_mean = t_sum / nr_types.unsqueeze(-1)             # 1 x E x D
                
            # calculate norm of type embeddings
            #t_norms = torch.norm(t_mean, dim=1).squeeze()

            # calculate cosine sim between type-embeddings and their mean
            #cos = cos_sim(t_embeds, t_mean, 2)                  # T_ x E x 1
            #cos[padding] = 0                                    # T_ x E x 1

            # calculate conicity (average cosine sim)
            #conicity = cos.sum(dim=0).div(nr_types).squeeze()   # E

            # calculate cosine sim between entity and avg type
            #e_embeds = embedder.base_embedder.embed(indexes)    # E x D
            #e_embeds = e_embeds @ W_v
            
            # calcaulte cosine sim between mean type and entity embeddings
            #atm = cos_sim(t_mean.squeeze(), e_embeds, 1).squeeze()
            
            # calculate norms of entity embeddings
            #e_norms = torch.norm(e_embeds, dim=1)
            
            #result.update(dict(
            #    type_conicity = conicity, 
            #    entity_type_cosine_sim = atm,
            #    type_norms = t_norms,
            #    entity_norms = e_norms,
            #))
            
            return result


        calc_results = self._perform_chunked(_perform_calculation)

        if self.trace_entity_level:
            self._add_results(calc_results, results_dict)

        self._add_average(calc_results, results_dict)
        self._generate_plot(calc_results, "embedding_analysis")


    def _calc_type_analysis(self, type_embedder, results_dict):
        
        T = self.dataset.num_types()
        results = {}
        
        MIN = 0.1
        
        temp = ~torch.eye(T, device=self.device).bool()
        
        #input()
        
        #overlap = self.jaccard_indices[0]
        jaccard = self.jaccard_indices[1]
        union_count = self.jaccard_indices[2]
        intersection_count = self.jaccard_indices[3]
        
        log_union_count = torch.log(union_count)
        
        good_pairs = torch.logical_and(jaccard > MIN, temp)
        good_pairs = torch.logical_and(good_pairs, union_count > 10)
        i = torch.arange(T)[:,None].expand(T,T)
        j = torch.arange(T)[None,:].expand(T,T)
        
        i = i[good_pairs].tolist()
        j = j[good_pairs].tolist()
        
        type_pairs = list(zip(i,j))
        type_pairs_t = torch.tensor(type_pairs, device=self.device)

        self.config.log(f"Proceeding with {len(type_pairs)} datapoints.")

        num_elements = len(type_pairs)
        chunk_size = self.chunk_size if self.chunk_size > -1 else num_elements
        nr_of_chunks = math.ceil(num_elements / chunk_size)
        
        all_results = []
        for chunk_number in range(nr_of_chunks):
            chunk_start = chunk_size * chunk_number
            chunk_end = min(chunk_size * (chunk_number + 1), num_elements)
            
            self.config.log(f"Chunk {chunk_number+1} / {nr_of_chunks}")
            
            t = type_pairs_t[chunk_start:chunk_end]
            t1, t2 = t[:,0], t[:,1]
            
            d1 = type_embedder.dist(t1)
            d2 = type_embedder.dist(t2)
            
            #kl_div1 = kl_divergence(d1,d2).sum(dim=-1)
            #kl_div2 = kl_divergence(d2,d1).sum(dim=-1)
            #kl_div = torch.minimum(kl_div1, kl_div2)
            #kl_div = torch.log(kl_div)
            #all_results.append(kl_div)
            
            diff = d1.mean - d2.mean                # B x D
            S = (d1.variance + d2.variance) / 2     # B x D
            
            diff1 = diff.unsqueeze(1)               # B x 1 x D
            diff2 = diff.unsqueeze(2)               # B x D x 1
            S_ = torch.diag_embed(S, offset=0, dim1=-2, dim2=-1)
            A = torch.bmm(diff1, torch.inverse(S_))
            term1 = torch.bmm(A, diff2).squeeze() / 8
            
            det_S1 = torch.logsumexp(d1.variance, dim=-1)
            det_S2 = torch.logsumexp(d2.variance, dim=-1)
            det_S = torch.logsumexp(S, dim=-1)
            term2 = torch.log( det_S / (det_S1 * det_S2).sqrt() ) / 2
            D_B = term1 + term2
            
            D_B = torch.exp(-D_B)
            all_results.append(D_B)
            
        kl_div = torch.cat(all_results).tolist()

        data = { 
            (x,y) : (
                jaccard[x,y].item(), 
                kl_div[i],
                log_union_count[x,y].item()
            ) 
            for i,(x,y) in enumerate(type_pairs)
        }

        df = pd.DataFrame.from_dict(data, orient="index", columns=['jaccard', 'kl_div', "count"])

        g = sns.jointplot(x="jaccard", y="kl_div", data=df,
              kind="reg", truncate=False, height=7, 
              joint_kws = {'scatter_kws' : dict(alpha=0.1, s=4, c=df["count"], color=None, cmap="YlGnBu")}
        )
        g.set_axis_labels("Jaccard Index", "Bhattacharyya Coefficient")
        
        dataset_name = self.config.get("dataset.name")
        
        fig = g
        fig.savefig(f"{dataset_name}_scatter_typepair_jaccard_vs_bhattacharyya.png", bbox_inches="tight", dpi=1000)


    def _evaluate(self):

        with torch.no_grad():

            # create initial trace entry
            self.current_trace["epoch"] = dict(
                type="type_prior_experiment",
                scope="epoch",
            )

            self.config.log("Analysing entity embeddings")
            
            done = set()
            for slot, embedder in self.embedders.items():

                if id(embedder) in done:
                    self.config.log(f"Skipping {SLOT_STR[slot]}_embedder becuase it is the same as a previous embedder.")
                    continue

                # results dictionary for this embedder
                embedder_results = dict()

                self._calc_embed_analysis(embedder, embedder_results)

                self.current_trace["epoch"].update({
                    f"{SLOT_STR[slot]}_embedder" : embedder_results
                })
                
                done.add(id(embedder))

            self.config.log("Analysing type embeddings")
            # type embedder...
            type_embedder = self.embedders[S].prior_embedder

            embedder_results = dict()
            self._calc_type_analysis(type_embedder, embedder_results)
            
            self.current_trace["epoch"].update({
                "type_embedder" : embedder_results
            })


