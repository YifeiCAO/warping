# The geometry of map-like representations under dynamic cognitive control

This is a repository of the ongoing project exploring the effect of cognitive control on the learned representation using deep neural network modeling experiments. 

**Abstract**: Extensive work has documented the spatial or “map-like” tuning properties of neurons in both spatial and non-spatial tasks. However, an animal’s objectives often depend on a subset of the features present in the environment, and these may fluctuate depending on changing contexts or goals. To successfully negotiate these situations animals require cognitive control, or the ability to flexibly select or attend stimulus features that are most relevant for achieving the current goal. Classic computational models of cognitive control have focused on settings with categorical feature dimensions rather than spatial or map-like representations. Furthermore, these models have typically emphasized the functional benefits of top-down control processes rather than the effects of control on learning and representational geometry. Here, we integrate these lines of work and explore the relationship between cognitive control and the geometry of map-like representations using neural network models, which we validated against data from an fMRI experiment. Task Our neural network extends classic models of cognitive control to a setting where feature dimensions are continuous rather than categorical, and are not pre-specified but must be learned directly from images. Our model reproduces classic control phenomena including the compression of representations along currently irrelevant feature dimensions, a finding that was also observed in . Additional analyses showed that the model’s representational geometry is affected by the presence of rapidly changing goals in an unexpected way: representations were warped (shared) along a context-invariant axis. The fMRI experiment showed that this warping phenomenon was also present in hippocampal representations, validating the model. Further simulations show that this warped geometry reflects the natural tendency of neural networks to learn a context-invariant value function. Taken together, our results suggest a detailed link between flexible, goal-driven behavior and the geometry of map-like neural representations.


## Summary

This repository contains the following:
- images: contains images of faces
- notebooks: contains all the jupyter notebooks to generate the figures and tsv files
- analyze.py: impelentation of all the analysis
- data.py: classes for impelmenting PyTorch dataset and dataloader
- main.py: train and test the model
- models.py: implementation of all the models 
- test.py: script that tests the model 
- train.py: script that trains the model while having periodic validation, testing, and analyzing results
- utils.py: contains helper function


### Requirements
- Python
- PyTorch
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org/get-started/locally/) to make sure of this.
- create a conda environment with the name 'warping': 
    - conda create -n warping
- install pip using 'conda install pip'
- activate the environment you created: 
    - conda activate warping
- install requirements by `pip install -r requirements.txt` or  install the packages separately, for example: 
    - ipykernel: `pip install ipykernel`
	- statsmodels: `pip install statsmodels`
	- sklearn: `pip install -U scikit-learn`
    - saeborn: `pip install seaborn==0.11.1`
- add the conda enviroment (called 'warping') to the jupyter kernel by using:
    - python -m ipykernel install --user --name warping --display-name "warping"

## Train and test the model
To train and test the model with the default parameters:

```bash
./run_all_experiments.sh
```
This will train and test the model, create `results` and `figures` folder. The results of the saved models will be stored in the `results` folder. 



## Figures and tsv files
To generate the figures, run each of the jupyter notebooks under `notebooks` folder. The generated figures and tsv files will be saved under `figures` and `results/tsv` folders, respectively. 
