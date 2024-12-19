# UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning


##  Introduction
This folder contains the code for our paper "UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning" for the VLM methods.

## 1. Getting started

### 1.1. Requirements

- torch 1.13.1 (or later)
- torchvision 
- tqdm
- numpy
- pillow
- pyyaml
- scipy
- [clip](https://github.com/openai/CLIP)


### 1.2 Download datasets and splits
For downloading the datasets and splits, we follow the instructions given in the Github repository of [TIP-Adapter](https://github.com/gaopengcuhk/Tip-Adapter). We use train/val/test splits from [CoOp's Github](https://github.com/KaiyangZhou/CoOp) for all datasets.

The downloaded datasets should be placed in the folder data/ the following way:

    .
    ├── ...
    ├── data           
    │   ├── food101       
    │   ├── eurosat       
    │   ├── dtd       
    │   ├── oxfordpets       
    │   ├── flowers101     
    │   ├── caltech101      
    │   ├── ucf101       
    │   ├── fgvcaircraft                
    │   ├── stanfordcars      
    │   ├── sun397        
    │                 
    └── ...


## 2. Reproducing the few-shot results

You can reproduce the results displayed in Table 2 in the paper by using the ```config/main_config.yaml``` file. Small variations in the results may be observed due to the randomization of the tasks.

The methods are EM-Dirichlet (```em_dirichlet```), Hard EM-Dirichlet (```hard_em_dirichlet```), $\alpha$-TIM (```alpha_tim```), PADDLE (```paddle```), Laplacian Shot (```laplacian_shot```), BDSCPN (```bdcpsn```), UNEM-Dirichlet (```u_em_dirichlet_embt_l```).

Methods ($\alpha$-TIM, PADDLE, Laplacian Shot, BDCPSN) having a hyper-parameter have be previously tuned on the validation set as [Transductive-clip](https://github.com/SegoleneMartin/transductive-CLIP).

For example, to run the method EM-Dirichlet on Caltech101 on 1000 realistic tranductive 4-shot tasks: 
```python
python main.py --opts shots 4 dataset caltech101 method em_dirichlet batch_size 100 number_tasks 1000 use_softmax_feature True
```

The learned parameters of UNEM-Dirichlet on the validation set are in the u_params folder under .pt extension files.

For example, to run the method UNEM-Dirichlet on Caltech101 on 1000 realistic tranductive 4-shot tasks: 
```python
python main.py --opts shots 4 dataset caltech101 batch_size 100 number_tasks 1000 method u_em_dirichlet_embt_l u_train False
```

To train again the parameters of UNEM-Dirichlet:
```python
python main.py --opts shots 4 dataset <dataset> method u_em_dirichlet_embt_l u_train True
```



## Aknowlegments
This repository was inspired by the publicly available code from the paper [Realistic evaluation of transductive few-shot learning](https://github.com/oveilleux/Realistic_Transductive_Few_Shot), [TIP-Adapter](https://github.com/gaopengcuhk/Tip-Adapter) and [Transductive-clip](https://github.com/SegoleneMartin/transductive-CLIP).





