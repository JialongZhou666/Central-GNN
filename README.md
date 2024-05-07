# Central-GNN
The code is based on a Pytorch adversarial repository, DeepRobust [(https://github.com/DSE-MSU/DeepRobust)](https://github.com/DSE-MSU/DeepRobust)

## Requirements
See that in https://github.com/DSE-MSU/DeepRobust/blob/master/requirements.txt
```
matplotlib==3.1.1
numpy==1.17.1
torch==1.2.0
scipy==1.3.1
torchvision==0.4.0
texttable==1.6.2
networkx==2.4
numba==0.48.0
Pillow==7.0.0
scikit_learn==0.22.1
skimage==0.0
tensorboardX==2.0
```

## Run the code
```
cd Central-GNN
python train.py --dataset polblogs --attack meta --ptb_rate 0.1 --epochs 1000
```
