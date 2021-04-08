import torch
import torch.nn.functional as F

from kge.job import EvaluationJob, Job

import math
import random

import numpy as np
import seaborn as sns
import pandas as pd

from sklearn import manifold

from sem_kge import TypedDataset

SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]

class TypeClusterTSNEExperimentJob(EvaluationJob):
    """
    Job to analyse entity embeddings by inspecting them using TSNE.
    Specifically to see if entities are being clustered by type.
    """

    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)

        self.embedders = {
            S : model.get_s_embedder(),
            #P : model.get_p_embedder(),   no types for relations
            O : model.get_o_embedder()
        }

        config_key = "type_cluster_tsne_experiment"
        self.chunk_size = self.config.get(f"{config_key}.chunk_size")

        self.dataset = TypedDataset.create(dataset)
        self.type_tensor = dataset.index("entity_type_set").to(self.device)

        # co-occurences of types
        E, T = self.type_tensor.shape
        kwargs = dict(device=self.device, requires_grad=False)
        indices = torch.arange((T), **kwargs).long()

        self.type_co = torch.zeros((T,T), **kwargs).long()
        for types in self.type_tensor:
            self.type_co[types,:] += types.long()

        assert torch.all(self.type_co.T == self.type_co)

        LEVEL = 2
        totals = torch.diagonal(self.type_co)
        self.type_subset = self.type_co == totals
        self.root_types = self.type_subset.sum(dim=0).long() == LEVEL
        config.log(f"Found {self.root_types.sum()} types with exactly {LEVEL-1} superset.")

        FREQUENT = 1000
        self.frequent_types = self.type_tensor.sum(dim=0) >= FREQUENT
        config.log(f"Found {self.frequent_types.sum()} types with at least {FREQUENT} members.")

        # At least half
        self.differentiating_types = self.type_tensor.sum(dim=0) < E/2
        config.log(f"Found {self.differentiating_types.sum()} types with at most half of entities.")

        self.interesting_types = torch.logical_and(self.root_types, self.frequent_types)
        self.interesting_types = torch.logical_and(self.interesting_types, self.differentiating_types)
        config.log(f"Found {self.interesting_types.sum()} types we consider interesting.")

    def _chunked_embed(self, embedder, num_elements, chunk_size):
        nr_of_chunks = math.ceil(num_elements / chunk_size)
        all_embeds = []
        for chunk_number in range(nr_of_chunks):
            chunk_start = chunk_size * chunk_number
            chunk_end = min(chunk_size * (chunk_number + 1), num_elements)

            indexes = torch.arange(chunk_start, chunk_end, device=self.device).long()
            embeds = embedder.embed(indexes)

            all_embeds.append(embeds)
        return torch.cat(all_embeds)

    def _add_type_vectors(self, embedder, dataframe, feat_cols, type_cols):

        # obtain type embeddings
        type_embedder = embedder.type_embedder
        num_types = self.dataset.num_types()
        chunk_size = self.chunk_size if self.chunk_size > -1 else num_types
        type_embeds = self._chunked_embed(type_embedder, num_types, chunk_size)

        # type embeds 'are of their own type'
        types = torch.eye(num_types, device=self.device).bool()

        # filter for interesting types
        # TODO make config option
        #type_embeds = type_embeds[self.interesting_types]
        #types = types[self.interesting_types]

        # apply attention projection
        # TODO make more general? (not all models with type embeddings will also have these kind of projections)
        W_v = embedder.self_attn.in_proj_weight
        _, D = W_v.shape
        W_v = W_v[2*D:,:].T
        type_embeds = type_embeds @ W_v

        # construct dataframe to append
        new_df = pd.DataFrame(type_embeds.tolist(), columns=feat_cols)
        new_df['kind'] = 'type'
        new_df['size'] = 2.

        for i,type_col in enumerate(type_cols):
            new_df[type_col] = types[:,i].tolist()

        return dataframe.append(new_df)

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

                # calculate embeds in chunks
                num_entities = self.dataset.num_entities()
                chunk_size = self.chunk_size if self.chunk_size > -1 else num_entities
                embeds = self._chunked_embed(embedder, num_entities, chunk_size)

                # create dataframe with embeddings
                feat_cols = [ f'dim_{i}' for i in range(embeds.shape[1]) ]
                df = pd.DataFrame(embeds.tolist(), columns=feat_cols)

                # add types to dataframe
                type_cols = [ f'type_{i}' for i in range(self.dataset.num_types()) ]
                for i,type_col in enumerate(type_cols):
                    df[type_col] = self.type_tensor[:,i].bool().tolist()

                df['kind'] = 'entity'
                df['size'] = 0.5

                # if we have type embeddings, add those too
                if hasattr(embedder, 'type_embedder'):
                    df = self._add_type_vectors(embedder, df, feat_cols, type_cols)

                # perform TSNE
                tsne_result = manifold.TSNE(perplexity=30., n_iter=250).fit_transform(df[feat_cols].values)
                df['tsne_1'] = tsne_result[:,0]
                df['tsne_2'] = tsne_result[:,1]

                # plotting
                def plot(hue_col, style_col, name):
                    plot = sns.scatterplot(
                        x="tsne_1", y="tsne_2",
                        hue=hue_col,
                        style=style_col,
                        size='size',
                        data=df,
                        legend="auto",
                        alpha=0.3,
                    )
                    figure = plot.get_figure()
                    figure.set_size_inches(8,8)
                    figure.savefig(name)
                    figure.clf()

                if False: # TODO make config option?
                    df.to_csv('dataframe.csv')

                for i in range(self.dataset.num_types()):
                    if self.interesting_types[i]:
                        plot(f"type_{i}", "kind", f"tsne_{SLOT_STR[slot]}_embeds_type_{i}.png")

                self.current_trace["epoch"].update({
                    f"{SLOT_STR[slot]}_embedder" : embedder_results
                })



