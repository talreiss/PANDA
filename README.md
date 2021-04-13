# PANDA
Official PyTorch implementation of [**“PANDA: Adapting Pretrained Features for Anomaly Detection and Segmentation”**](https://arxiv.org/pdf/2010.05903.pdf) (CVPR 2021).

## Virtual Environment
Use the following commands:
```
cd path-to-PANDA-directory
virtualenv venv --python python3
source venv/bin/activate
pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html
```

## Data Preparation
Use the following commands:
```
cd path-to-PANDA-directory
mkdir data
```

Download:
* [80M Tiny Images - OE](https://drive.google.com/file/d/16c8-ofOnN5l7hmWp--WBCx3LIKXwHHuf/view?usp=sharing)
* [Fisher Information Matrix Diagonal](https://drive.google.com/file/d/12PTw4yNqp6bgCHj94vcowwb37m81rvpY/view?usp=sharing)

Extract these files into `path-to-PANDA-directory/data` and unzip tiny.zip

## Experiments
To replicate the results on CIFAR10, FMNIST for a specific normal class with EWC:
```
python panda.py --dataset=cifar10 --label=n --ewc --epochs=50
python panda.py --dataset=fashion --label=n --ewc --epochs=50
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
@article{reiss2020panda,
  title={PANDA--Adapting Pretrained Features for Anomaly Detection},
  author={Reiss, Tal and Cohen, Niv and Bergman, Liron and Hoshen, Yedid},
  journal={arXiv preprint arXiv:2010.05903},
  year={2020}
}
```
