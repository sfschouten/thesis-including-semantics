# SemKGE

A python package that extends [LibKGE](https://github.com/uma-pi1/kge) with methods for learning **K**nowledge **G**raph **E**mbeddings that include **Sem**antics.

## Install
To install this package simply clone the repo and run the following.

	pip install --editable .
	
This will automatically install its dependencies as well.

! **While this package is in development, please install [my fork of the LibKGE](https://github.com/sfschouten/kge) package prior to installing this package.** !

After SemKGE and LibKGE are both installed please follow LibKGE's instructions on downloading/preparing the data. 

Then one final data preprocessing step is necessary, which adds the type information to the datasets. This can be done with the following snippet of shell commands.

    cp -rT '<KGE_HOME>/data/wnrr' '<SEMKGE_HOME>/data/wnrr-typed'
	cp -rT '<KGE_HOME>/data/fb15k-237' '<SEMKGE_HOME>/data/fb15k-237-typed'

	cd <SEMKGE_HOME>/data
    mkdir -p ".tmp"
    
	cd <SEMKGE_HOME>/data/.tmp
	!curl -L https://surfdrive.surf.nl/files/index.php/s/N1c8VRH0I6jTJuN/download --output wn18rr.tar.gz
	!curl -L https://surfdrive.surf.nl/files/index.php/s/rGqLTDXRFLPJYg7/download --output fb15k-237.tar.gz
	tar xzf wn18rr.tar.gz
	tar xzf fb15k-237.tar.gz
	cp WN18RR/entity2type.txt /content/sem_kge/data/wnrr-typed/entity_types.txt
	cp FB15k-237/entity2type.txt /content/sem_kge/data/fb15k-237-typed/entity_types.txt
	cd ..
	
	python preprocess_types.py wnrr-typed
	python preprocess_types.py fb15k-237-typed

! **At some point it probably makes more sense simplify this process, or work with LibKGE to just have the type information in their version of the datasets.** !

## Using SemKGE
SemKGE works the same way as LibKGE itself. For example: 

    kge start <SEMKGE_HOME>/runs/fb15k-237-transt.yaml

## Jupyter Notebook
For an example on how to do the entire installation process and use the package you can also look at the [Jupyter Notebook](https://github.com/sfschouten/thesis-including-semantics/blob/main/SemKGE.ipynb).



## Implemented Methods
- TransT

## Results


