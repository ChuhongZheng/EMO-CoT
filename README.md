# EMO-CoT
Part of the code is borrowed from [in-context-learning](https://github.com/dtsip/in-context-learning).

**Epigraph Based Multilevel Optimization (EMO) for Enhancing Chain-of-Thought Reasoning Capabilities**<br />
S. Lu, Y. Ding, L. Horesh, J. Gao and M. Magdon-Ismail<br />
Paper: [https://ieeexplore.ieee.org/abstract/document/10889376](https://ieeexplore.ieee.org/abstract/document/10889376)


## Start up
Follow [in-context-learning](https://github.com/dtsip/in-context-learning)'s instruction by runing
```
conda env create -f environment.yml
conda activate in-context-learning
```

## Training
1. Enter into directory ```src```
2. Run ```train.py``` based on the choosen config file: ```python train.py --config conf/[config_file].yaml```

## Test
1. Enter into directory ```src```
2. Refer to the ```eval.ipynb``` file

