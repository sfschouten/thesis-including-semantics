# SemKGE

A python package that extends [LibKGE](https://github.com/uma-pi1/kge) with methods for learning **K**nowledge **G**raph **E**mbeddings that include **Sem**antics.

## Jupyter Notebook

The [Jupyter Notebook](https://github.com/sfschouten/thesis-including-semantics/blob/main/SemKGE.ipynb) also serves as an example for the installation process and how to use the package .


## Implemented Methods
- TransT

## Results
#### FB15K-237 (Freebase)

|                                                                                                       |   MRR | Mean Rank | Hits@1 | Hits@3 | Hits@10 |                                                                                      Config file |
|-------------------------------------------------------------------------------------------------------|------:|----------:|-------:|-------:|--------:|-------------------------------------------------------------------------------------------------:|
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) | 0.313 |  -        |  0.221 |  0.347 |   0.497 |   [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-transe.yaml) |
| TransT-type (ours/best)                                                                               | 0.324 |  159      |  0.231 |  0.357 |   0.517 |   [todo]() |
| TransT-type-only (ours/best)                                                                          |       |           |        |        |         |   [todo]() |

#### WN18RR (Wordnet)

|                                                                                                       |   MRR | Mean Rank | Hits@1 | Hits@3 | Hits@10 |                                                                                 Config file |
|-------------------------------------------------------------------------------------------------------|------:|----------:|-------:|-------:|--------:|--------------------------------------------------------------------------------------------:|
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) | 0.228 |  -        | 0.053  |  0.368 |   0.520 |   [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-transe.yaml) |
| TransT-type (ours/best)                                                                               | 0.191 |  4617     | 0.012  |  0.358 |   0.438 |   [todo]() |
| TransT-type-only (ours/best)                                                                          |       |           |        |        |         |   [todo]() |

#### FB15K (Freebase)

|                                                                                                                   |   MRR | Mean Rank | Hits@1 | Hits@3 | Hits@10 |                                                                                Config file |
|-------------------------------------------------------------------------------------------------------------------|------:|----------:|-------:|-------:|--------:|-------------------------------------------------------------------------------------------:|
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)             | 0.676 |  -        | 0.542  | 0.787  |   0.875 |   [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/fb15k-transe.yaml) |
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) (Ma et al.) | -     |  143      | -      | -      |   0.621 |   -                                                                                        |
| TransT-type (ours/best)                                                                                           |       |           |        |        |         |   [todo]() |
| TransT-type (ours/repro)                                                                                          |       |           |        |        |         |   [todo]() |
| TransT-multiple (ours/repro)                                                                                      |       |           |        |        |         |   [todo]() |
| TransT-multiple-type (ours/repro)                                                                                 |       |           |        |        |         |   [todo]() |
| TransT-type (Ma et al.)                                                                                           | -     |  72       | -      | -      |   0.823 |   -                                                                                        |
| TransT-multiple (Ma et al.)                                                                                       | -     |  62       | -      | -      |   0.836 |   -                                                                                        |
| TransT-multiple-type (Ma et al.)                                                                                  | -     |  46       | -      | -      |   0.854 |   -                                                                                        |

#### WN18 (Wordnet)

|                                                                                                                   |   MRR | Mean Rank | Hits@1 | Hits@3 | Hits@10 |                                                                               Config file |
|-------------------------------------------------------------------------------------------------------------------|------:|----------:|-------:|-------:|--------:|------------------------------------------------------------------------------------------:|
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)             | 0.553 |  -        | 0.315  | 0.764  |   0.924 |   [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wn18-transe.yaml) |
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) (Ma et al.) | -     |  251      | -      | -      |   0.892 |   -                                                                                       |
| TransT-type (ours/repro)                                                                                          |       |           |        |        |         |   [todo]() |
| TransT-type (ours/best)                                                                                           |       |           |        |        |         |   [todo]() |
| TransT-multiple-type (Ma et al.)                                                                                  | -     |  130      | -      | -      |   0.974 |   -                                                                                       |




