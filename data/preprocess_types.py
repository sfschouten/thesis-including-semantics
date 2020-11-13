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

    ENTITY_TYPES = "entity_types.txt"
    ENTITY_TYPES_IDS = "entity_types.del"
    
    # read data and collect types 
    types = {}
    type_id = 0
    
    with open(args.folder + "/" + ENTITY_TYPES, "r") as f1:
        with open(os.path.join(args.folder, ENTITY_TYPES_IDS), "w") as f2:
            raw = list(map(lambda s: s.strip().split("\t"), f1.readlines()))
            for line in raw:
                entity = line[0]
                ts = line[1:]
                ids = []
                for t in ts:
                    if t not in types:
                        types[t] = type_id
                        type_id += 1
                    ids.append(types[t])
                
                f2.write(
                    entity
                    + "\t"
                    + ','.join(str(id_) for id_ in ids)
                    + "\n"
                )

            print(
                f"Found {len(raw)} entities with types in file {ENTITY_TYPES}."
            )

    print(f"{len(types)} distinct entity types")
    print("Writing type map...")
    store_map(types, os.path.join(args.folder, "type_ids.del"))
    print("Done writing.")


    


