
from torch import Tensor

from kge import Dataset

from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np


class TypedDataset(Dataset):

    @staticmethod
    def create(self: Dataset) -> 'TypedDataset':
        config = self.config

        try:
            self._num_types: Int = config.get("dataset.num_types")
            if self._num_types < 0:
                self._num_types = None
        except KeyError:
            self._num_types: Int = None

        self.__class__ = TypedDataset

        self.create_type_index_functions()

        return self

    def create_type_index_functions(self):
        from sem_kge.indexing import index_entity_types
        from sem_kge.indexing import index_relation_types
        self.index_functions["entity_type_set"] = index_entity_types
        self.index_functions["relation_type_freqs"] = index_relation_types


    def num_types(self) -> int:
        "Return the number of entity types in this dataset."
        if not self._num_types:
            #self._num_types = 
            raise NotImplementedError()
        return self._num_types


    def entity_types(
        self, indexes: Optional[Union[int, Tensor]] = None
    ) -> Union[str, List[str], np.ndarray]:
        """Decode indexes to entity types.

        See `Dataset#map_indexes` for a description of the `indexes` argument.

        """

        map_ = self.load_map(
            "entity_types", as_list=True, ids_key="entity_ids", ignore_duplicates=True
        )

        return self._map_indexes(indexes, map_)
