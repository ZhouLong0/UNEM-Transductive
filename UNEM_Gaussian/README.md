# UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning


##  Introduction
This folder contains the code for our paper "UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning" to reproduce the visual only methods.


## 1. Getting started

### 1.1. Requirements

- python 3.10
- pytorch=2.3
- loguru=0.5.3
- matplotlib
- pyyalm
- scikit-learn
- torchvision
- tqdm
- cvxpy

### 1.2 Download datasets and models
Our framework was developped for the datasets _mini-imagenet_, _tiered-imagenet_ and _CUB_. We used pre-trained models. 

The downloaded datasets should be placed in the folder data/ the following way:

    .
    ├── ...
    ├── data                    
    │   ├── mini       
    │   ├── tiered        
    │   └── cub               
    └── ...

The downloaded models should be placed in the folder checkpoints/ the following way:

    .
    ├── ...
    ├── checkpoints                    
    │   ├── mini          
    │   ├── tiered        
    │   └── cub               
    └── ...


For the datasets and pre-trained models, please download it by following this [repo](https://github.com/imtiazziko/LaplacianShot).



## 3. Reproducing the few-shot results

You can reproduce the main results displayed in the paper by using the ```config/main_config.yaml``` file. Small variations in the results may be observed due to the randomization of the tasks.


Methods ($\alpha$-TIM, PADDLE, Laplacian Shot, BDCPSN) having a hyper-parameter have be previously tuned on the validation set as [PADDLE](https://github.com/SegoleneMartin/PADDLE).

For example, to run the method PADDLE on Mini-imagenet with Resnet18 as backbone on 1000 realistic transductive tasks: 
```python
python main.py --opts dataset mini arch resnet18 method paddle train_unrolling False
```

The learned parameters of UNEM-Gaussian are in the results/u_params folder under .pt extension files.

For example, to run the method UNEM-Gaussian on Mini-imagenet with Resnet18 as backbone on 1000 realistic tranductive tasks: 
```python
python main.py --opts dataset mini arch resnet18 method u_paddle_embt_l train_unrolling  False
```

To train again the parameters of UNEM-Gaussian:
```python
python main.py --opts dataset <dataset> arch <resnet18/wideres> method u_paddle_embt_l u_train True train_unrolling True train_method U_PADDLE_EMBT_L
```

# Acknowledgement
This repository was inspired by the publicly available code from the paper [Towards Practical Few-Shot Query Sets: Transductive Minimum Description Length Inference](https://github.com/SegoleneMartin/PADDLE).
