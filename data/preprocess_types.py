#!/usr/bin/env python3

import argparse
import yaml
import os.path
import numpy as np


def store_map(symbol_map, filename):
    with open(filename, "w") as f:
        for symbol, index in symbol_map.items():
            f.write(f"{index}\t{symbol}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    args = parser.parse_args()

    print(f"Preprocessing type information in {args.folder}...")

    YAML = "dataset.yaml"
    ENTITY_TYPES = "entity_types.txt"
    ENTITY_TYPES_IDS = "entity_types.del"
    
    TRIPLES = ("train.txt", "valid.txt", "test.txt")
   
    print(f"Loading triples to establish which entities are actually used.")
    combined_set = set()
    for triple_file in TRIPLES:
        with open(args.folder + "/" + triple_file, "r") as f:
            combined_set |= frozenset(
                _id for _ids in map(lambda s: s.strip().split("\t"), f.readlines()) for _id in _ids[:1] + _ids[2:]
            )
   
    # -------------------------------- #
    # read entities and collect types  #
    # -------------------------------- #
    print(f"Reading entities and writing type ids.")
    
    types = {}
    used_types = set()
    type_id = 0
    
    nr_types_used = 0
    avg_nr_types = 0
    
    # open raw entity_types for reading
    with open(args.folder + "/" + ENTITY_TYPES, "r") as f1:

        # open processed entity_types for writing
        with open(os.path.join(args.folder, ENTITY_TYPES_IDS), "w") as f2:

            # read the lines into a list of pairs
            raw = list(map(lambda s: s.strip().split("\t"), f1.readlines()))

            # go through line-for-line
            for line in raw:
                entity = line[0]
                ts = line[1:]
                ids = []

                # go through types
                for t in ts:

                    # check for new type 
                    if t not in types:
                        types[t] = type_id
                        type_id += 1

                    ids.append(types[t])
                
                # write entity id with type ids
                f2.write(entity + "\t" + ','.join(str(id_) for id_ in ids) + "\n")
                
                if entity in combined_set:
                    avg_nr_types += len(ids)
                    used_types |= set(ids)
                
    with open(args.folder + "/" + YAML, "r") as f:
        yaml_file = yaml.load(f, Loader=yaml.FullLoader)
                
    nr_entities1 = len(raw)                    # the number of entities in entity_ids.del
    nr_entities2 = len(combined_set)           # the number of entities referenced in the triples

    avg_nr_types /= nr_entities2
    
    print(f"Found {nr_entities1} entities with types in file {ENTITY_TYPES} (versus {nr_entities2} in triples.).")
    print(f"Found total of {len(types)} distinct entity types.")
    print(f"Found {len(used_types)} entity types related to entity in triples.")
    print(f"Average number of types per entity is: {avg_nr_types}.")
    
    print("Writing type map...")
    store_map(types, os.path.join(args.folder, "type_ids.del"))
    print("Done.")

    print("Editing dataset.yaml")
    

    yaml_file["dataset"]["files.entity_types.filename"] = ENTITY_TYPES_IDS
    yaml_file["dataset"]["files.entity_types.type"] = "idmap" 
    yaml_file["dataset"]["num_types"] = len(types)

    old_name = yaml_file["dataset"]["name"]
    suffix = "-typed"
    if not old_name.endswith(suffix):
        yaml_file["dataset"]["name"] = old_name + suffix

    with open(args.folder + "/" + YAML, "w") as f:
        yaml.dump(yaml_file, f)
