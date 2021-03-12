
import torch


def index_relation_types(dataset: "TypedDataset"):
    """
    """
    KEY = "relation_type_freqs"
    
    if KEY not in dataset._indexes:
        dataset.config.log("Creating relation_type_freqs tensor.")
        
        types_tensor = dataset.index("entity_type_set")
        
        R = dataset.num_relations()
        T = dataset.num_types()
        # Create structures for relation's common type set. 
        type_head_counts = torch.zeros((R,T))
        type_tail_counts = torch.zeros((R,T))
        type_totals      = torch.zeros((R,1))
    
        # Count types in head/tail and divide by total to get freq.
        triples = dataset.split('train').long()
        for triple in triples:
            h,r,t = triple
            type_head_counts[r] += types_tensor[h]
            type_tail_counts[r] += types_tensor[t]
            type_totals[r] += 1

        relation_type_freqs = torch.stack([
            type_head_counts / type_totals, type_tail_counts / type_totals 
        ])
        
        dataset._indexes[KEY] = relation_type_freqs
        
    return dataset._indexes[KEY]

def index_entity_types(dataset: "TypedDataset"):
    """
    Builds a tensor of each entity's types in one-hot encoded fashion.
    """
    KEY = "entity_type_set"
    
    if KEY not in dataset._indexes:
        # Create structure with each entity's type set.
        entity_types = dataset.entity_types()
        N = dataset.num_entities()
        T = dataset.num_types()

        dataset.config.log("Creating entity type_set one-hot tensor.")

        # We represent each set as a binary vector.
        # A 'True' value means that the type corresponding
        #  to the column is in the set.
        types_tensor = torch.zeros((N,T), dtype=torch.bool)
        for idx, typeset_str in enumerate(entity_types):
            if typeset_str is None:
                continue
            typelist = list(int(t) for t in typeset_str.split(','))
            for t in typelist:
                types_tensor[idx, t] = True
    
        dataset._indexes[KEY] = types_tensor
       
    return dataset._indexes[KEY]




