# Dynamic Gradient Scenario Memory (D-GSM)
## Introduction
This is the official code of the paper *"Continual Interactive Behavior Learning With Scenarios Divergence Measurement: A Dynamic Gradient Scenario Memory Approach",Yunlong Lin, Zirui Li, Cheng Gong, Chao Lu, Xinwei Wang, Jianwei Gong*. The paper is submitted to *IEEE Transanctions on Intelligent Transportation Systems (IEEE T-ITS)*.

## Enviroment and set up
Codes are implemented in Ubuntu 18.04. 

#### Python = 3.6
#### Pytorch $\geq$ 1.09

The detailed requirement of setup is shown in *requirement.txt*.
You can have everything set up by running:
```
pip install -r requirements.txt
```

## Dataset
Dataset used in this paper is the publicly available *INTERACTION Dataset: An INTERnational, Adversarial and Cooperative moTION Dataset in Interactive Driving Scenarios with Semantic Maps.* Subsets denoted as MA, FT, ZS, EP, SR are used in experiments to construct continuous scenarios. [INTERACTION](https://interaction-dataset.com/)

## Plug-and-play quality.
The proposed approach is plug-and-play. As an example, the Social-STGCNN model proposed in paper *Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction* is adopted as base model in this work. The original base model is availble in [base model](https://github.com/abduallahmohamed/Social-STGCNN).

## General Introduction of Codes
The file fold named "D-GSM-training" contains the implementation of the proposed dynamic-GSM. Another fold named "GSM-training" is the implementation of the proposed method without dynamic memory mechanism. And in these two folds, there are shell files to directly run the codes (if you prepare the right file path and processed data). In the shell file, the default setting is to train three continuous scenarios (corresponding to the experiment in **Section IV** of the paper). You can change the tasks number and datasets to implement other continual scenarios. The detailed parameter or variable names are described in annotations of .py files, please check them and make your own experiments. 

File folds named "Testing" and "Traffic Divergence Measuring" are codes for model evaluation (corresponding to **Section IV-C** of the paper) and the implementation of the proposed methods for measuring divergence between different scenarios (corresponding to experiments in **Appendix B**).

## Usage of Codes: Traffic Divergence Measuring
These codes are the implementation of experiments described in **Appendix B** of the paper, which explores the influence of data amount on conditional Kullback-Leibler divergence (CKLD) calculations. Under the direction "./Traffic Divergence Measuring", the file fold "data" is to contain processed cases, and "mdn_model" contains models of Mixture Density Networks (MDN) used in the paper.
### Running
*STEP 1:* Preprocess the raw data by using "preprocess_raw_data.m" (Please preprocess and save different scenarios seperately. The default setting is for scenario "MA".) The saved file is named "Scenario-XX.mat" (XX is the markers of a specific scenairo, e.g. "Scenario-MA.mat"). Then put the preprocessed file into "./Traffic Divergence Measuring/data" file fold. 

*STEP 2:* Use "data_extraction.py" to futher process "Scenario-XX.mat" to obtain the final processed cases. The processed file will be named "Scenario-XX-vi-j-n%.mat" ("XX" is scenario marker, and "i" is the considered number of surrounding vehicles for each case, and "j" is the considered number of eigenvectors. "n%" corresponds to the ratio of data usage. e.g."Scenario-MA-v5-3-100%.mat").
```
python3 data_extraction.py
```

*STEP 3:* Use "main_cot_lap_kld.py" to estimate GMMs of scenario cases, and then, to calculate CKLDs between scenarios. We have provided a group of processed files "Scenario-XX-v5-3-100%.mat" as examples in the "data" fold. Thus, you can directly run the following codes in the terminal to start the calculation.
```
python3 main_cot_lap_kld.py
```

## Usage of Codes: D-GSM-training
These codes are implementations of experiments represented in **Section IV**, where base models are applied with the proposed CL approach to continually learn interactive behaviors in continuous scenarios.

### Running
*STEP 1:* Extract data features from raw data for trajectory prediction by using "./D-GSM-training/datasets/preprocess_for_learning.py". Please edit the right file direction in the script, then you can run:
```
python3 preprocess_for_learning.py
```
If you want to use the default codes without changing the file directions, please remember to save different datasets representing different scenarios, sperately. (Detailed setting can be found in "./D-GSM-training/datasets/readme.md")

*STEP 2:* Train the base model in one single scenario (CL is not needed in this situation). We have provided trained model in the single scenario MA in file fold "./D-GSM-training/checkpoint/social-stgcnn-MA" for further training in continuous scenarios.

*STEP 3:* Train the model in contiuous scenarios by directly running .sh files: "./D-GSM-training/train_DGSM_two_mem3500.sh" and "./D-GSM-training/train_DGSM_two_mem3500.sh" are for model with D-GSM and GSM (see *Experimental settings* in the paper **Section IV-B**).

```
bash train_DGSM_two_mem3500.sh
```

```
bash train_GSM_two_mem3500.sh
```
It should be noted that, as examples, the .sh files described above are set for two continuous scenarios (MA->FT) with 3,500 samples of memory data. If you want to train model further (eg. 3/4/5 continuous scenarios), please change the hyper paramters in shell files based on understanding python stripts train_DGSM.py and train_GSM.py. Examples have also been provided as notes in the .sh files now. Make sure the file direction of your saved weights is right.


## Usage of Codes: Testing
These codes are used to evaluate the trained models.

### Running
*STEP 1:* Set the model to be tested with the scenario data for the testing. Details for DIY settings can be found in readme.md in folds "./D-GSM/Testing/datasets" and "./D-GSM/Testing/checkpoint". We have provided an example in these folds-- the evaluation for model being contiually trained in five continuous scenarios (MA->FT->ZS->EP->SR). The performance (measured by ADE and FDE) on the first observed scenario "MA" can be shown by running:

```
python3 test.py
```
If you want to test for other model or scenario using default direction settings, please set the models and the testing data in the right direction.
