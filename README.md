# PlantDiseaseRecongnition
The project aimed at developing an automated plant disease classification system that can assist farmers, researchers, and other stakeholders in the agricultural sector. The project leverages digital imaging technology and deep learning algorithms to accurately detect and classify different types of plant diseases using leaf images.

We have used deep learning algorithms such as MobileNet, ShuffleNet, and Resnet18 to learn the visual features of healthy and diseased leaves and classify them into different disease categories.

Specifically, the models achieved accuracy rates ranging from 92.64% to 99.19% on various datasets of diseased and healthy plant leaves. These results demonstrate the potential of deep learning-based solutions for accurately identifying and classifying plant diseases, which could ultimately reduce the need for manual inspection by experts and improve the efficiency and cost-effectiveness of plant disease diagnosis.

## Getting started
There are three main directory dedicated to each model. 
- [MobileNet-v2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)
- [ResNet18](https://pytorch.org/hub/pytorch_vision_resnet/)
- [ShuffleNet](https://pytorch.org/hub/pytorch_vision_shufflenet_v2/)

In each of above directory we have three subdirectory containing jupyter notebook for each Dataset.
- [Dataset 1](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset)
- [Dataset 2](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [Dataset 3](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## Hyperparameter Tuning
The directory HyperParameterTuning contains different notebook run on different hyper-parameters.

## TNSE
The directory TNSE contains notebook showing TNSE analysis of datasets

## Transfer Learning
The directory transfer learning contains notebooks in which model are run on pretained weights such as [ Weights.IMAGENET1K_V1](https://pytorch.org/vision/stable/models.html#general-information-on-pre-trained-weights)


## How to run
Import the notebook at [google collab](https://colab.research.google.com/) and run in GPU.

## Testing
Each model directory has saved trained model .pt file. You can import the saved model and do inference on your custom dataset to detect plant disease.
