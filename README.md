# D-GSM
## Introduction
This is the official code of the paper *"Continual Interactive Behavior Learning With Scenarios Divergence Measurement: A Dynamic Gradient Scenario Memory Approach",Yunlong Lin, Zirui Li, Cheng Gong, Chao Lu, Xinwei Wang, Jianwei Gong*. The paper is submitted to *IEEE Transanctions on Intelligent Transportation Systems*.

## Enviroment
Codes are implemented in Ubuntu 18.04. The detailed requirement of setup is shown in *requirement.txt*.

## Dataset
Dataset used in this paper is the publicly available *INTERACTION Dataset: An INTERnational, Adversarial and Cooperative moTION Dataset in Interactive Driving Scenarios with Semantic Maps.* Subsets denoted as MA, FT, ZS, EP, SR are used in experiments to construct continuous scenarios. [INTERACTION](https://interaction-dataset.com/)

## Plug-and-play quality.
The proposed approach is plug-and-play. As an example, the Social-STGCNN model proposed in paper *Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction* is adopted as base model in this work. The original base model is availble in [base model](https://github.com/abduallahmohamed/Social-STGCNN).
