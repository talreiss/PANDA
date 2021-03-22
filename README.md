# PANDA
Official PyTorch implementation of the paper: “PANDA: Adapting Pretrained Features for Anomaly Detection and Segmentation”.

## Virtual Environment
Use the following commands:
```
cd path-to-PANDA-directory
virtualenv venv --python python3
source venv/bin/activate
pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html
```

## Data Preparation
Download:
* [80M Tiny Images - OE](https://drive.google.com/file/d/16c8-ofOnN5l7hmWp--WBCx3LIKXwHHuf/view?usp=sharing)
* [Fisher Information Matrix Diagonal](https://drive.google.com/file/d/12PTw4yNqp6bgCHj94vcowwb37m81rvpY/view?usp=sharing)

Extract these files in `data` folder.

## Experiments
To replicate the results on CIFAR10, FMNIST for a specific normal class with EWC:
```
python panda.py --dataset=cifar10 --label=n --ewc
python panda.py --dataset=fashion --label=n --ewc
```
To replicate the results on CIFAR10, FMNIST for a specific normal class with early stopping:
```
python panda.py --dataset=cifar10 --label=n
python panda.py --dataset=fashion --label=n
```
Where n indicates the id of the normal class.

To run experiments on different datasets, please set the path in utils.py to the desired dataset.

## OE Experiments
To replicate the results on CIFAR10 for a specific normal class:
```
python outlier_exposure.py --dataset=cifar10 --label=n
```
Where n indicates the id of the normal class.


## Citation
If you find this useful, please cite our paper:
```

```
