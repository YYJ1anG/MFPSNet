## MFPSNet

This repository contains the code for the blind face restoration paper "Multi-Prior Learning via Neural Architecture Search for Blind Face Restoration." This paper searches the optimal network for blind face restoration and utilizes multiple facial priors in one network by neural network architecure. The code for searching, retraining, and testing is included in this repo. We also embed the searched architectures in the code for more convenience.


## Requirements

### Environment

1. Python 3.6.*
2. CUDA 10.0n
3. PyTorch >= 1.1.0



###  LQ Image Preparation
You can use `degrade.py` to generate LQ images for training. Please modify the degradation type and source image directory before applying it.

```shell
python degrade.py
```

### Run demo

You can directly run a trained model using `demo.sh` for demo images in `./data/img/`.
```shell
sh demo.sh
```


### Architecture Search 
two source files for the architecture search are also presented: 

#### 1. Search 
The `search.py` is the code for architecuture seach. Modify  `./config_utils/search_args.py` before searching.
```
python search.py
```
#### 2. Retrain 
The `train.py` is the code for retraining MFPSNet. Modify  `./config_utils/train_args.py` before retraining. Notably, the search architectures are already embeded in the source code for convenience. Thus, you can directly retrain the MFPSNet without searching the whole architecture first.
```shell
python train.py
```

