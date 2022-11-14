# D-GSM
## Introduction
This is the official code of the paper *"Continual Interactive Behavior Learning With Scenarios Divergence Measurement: A Dynamic Gradient Scenario Memory Approach",Yunlong Lin, Zirui Li, Cheng Gong, Chao Lu, Xinwei Wang, Jianwei Gong*. The paper is submitted to *IEEE Transanctions on Intelligent Transportation Systems*.

## Enviroment and set up
Codes are implemented in Ubuntu 18.04. The detailed requirement of setup is shown in *requirement.txt*.
You can have everything set up by running:
```
pip install -r requirements.txt
```

## Dataset
Dataset used in this paper is the publicly available *INTERACTION Dataset: An INTERnational, Adversarial and Cooperative moTION Dataset in Interactive Driving Scenarios with Semantic Maps.* Subsets denoted as MA, FT, ZS, EP, SR are used in experiments to construct continuous scenarios. [INTERACTION](https://interaction-dataset.com/)

## Plug-and-play quality.
The proposed approach is plug-and-play. As an example, the Social-STGCNN model proposed in paper *Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction* is adopted as base model in this work. The original base model is availble in [base model](https://github.com/abduallahmohamed/Social-STGCNN).

## Codes and Usage
The file fold named "D-GSM-training" contains the implementation of the proposed dynamic-GSM. Another fold named "GSM-training" is the implementation of the proposed method without dynamic memory mechanism. And in these two folds, there are shell files to directly run the codes (if you prepare the right file path and processed data). In the shell file, the default setting is to train three continual scenarios. You can change the tasks number and datasets to implement other continual scenarios. The detailed parameter or variable names are described in annotations of .py files, please check them and make your own experiments. 

File folds named "Testing" and "Traffic divergence measuring" are codes for model evaluation and the implementation of the proposed methods for measuring divergence between different scenarios.
