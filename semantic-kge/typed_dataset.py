

from kge import Dataset


class TypedDataset(Dataset):


    def __init__(self, config, folder=None):
        super().__init__(config, folder)

        try:
            self._num_types: Int = config.get("dataset.num_types")
            if self._num_types < 0:
                self._num_types = None
        except KeyError:
            self._num_types: Int = None


    def num_types(self) -> int:
        "Return the number of entity types in this dataset."
        if not self._num_types:
            #self._num_types = 
            raise NotImplementedError()
        return self._num_types


    def entity_types(
        indexes: Optional[Union[int, Tensor]] = None
    ) -> Union[str, List[str], np.ndarray]:
        """Decode indexes to entity types.

        See `Dataset#map_indexes` for a description of the `indexes` argument.

        """

        map_ = self.load_map(
            "entity_types", as_list=True, ids_key="entity_ids", ignore_duplicates=True
        )

        return self._map_indexes(indexes, map_)
