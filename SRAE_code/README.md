# Self-recoverable Adversarial Examples: A New Effective Protection Mechanism in Social Networks (TCSVT 2022)

Pytorch implementation of paper "Self-recoverable Adversarial 
Examples: A New Effective Protection Mechanism in Social Networks
" by Jiawei Zhang, Jinwei Wang, Hao Wang, Xiangyang Luo.

Published on IEEE Transactions on Circuits and Systems for Video Technology (TCSVT 2022).

## Requirements

You need [Pytorch](https://pytorch.org/) with TorchVision to run this.
The code has been tested with Python 3.7 and runs on Windows 10.

## Data

[Caltech-256](https://authors.library.caltech.edu/7694/) 
is used to train and evaluate the performance.The default path to data is D:/caltech256/256_ObjectCategories/...

## Running

You will need to install the requirements, then run 
```
python main.py
```
for the training.

You can run
```
python evaluate.py
```
for the evaluation.

You can run
```
python robust test.py
```
for the robustness evaluation.

You can run
```
python train_target_model.py
```
for alternative target model.

## Citation
If you find this repository helpful, you may cite:

```tex
@ARTICLE{3207008,
  author={Zhang, Jiawei and Wang, Jinwei and Wang, Hao and Luo, Xiangyang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Self-recoverable Adversarial Examples: A New Effective Protection Mechanism in Social Networks}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2022.3207008}}
```
