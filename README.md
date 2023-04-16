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

Instruction to obtain the Datasets:
    Dataset can be downloaded directly from the Kaggle. In the included notebooks, we just have to define the Username and the password for Kaggle. And the dataset can be downloaded through the implementation it self.
  
  Eg: 
  
    import os
    os.environ['KAGGLE_USERNAME'] ='sampleUserName'
    os.environ['KAGGLE_KEY'] = 'samplePassword'
    
   By running the below code in the implementation notebooks, the dataset will be downlaoded to the colab environment.
   
      !kaggle datasets download -d rashikrahmanpritom/plant-disease-recognition-dataset


Floder Structure:

  HyperParameterTuning
  
    Contains the notebooks which is used for the hyper parameter tuning.
      Dataset1
        There several notebooks which uses the dataset 1 with shufflenet.
          shufflenet1 _epoch10_batch62.ipynb
          shufflenet1 _epoch20_batch32.ipynb
          shufflenet1_epoch10_batch128.ipynb
          shufflenet1_epoch_1.ipynb
          shufflenet1_epoch_20_batch62.ipynb
      Dataset2
        There 4 notebooks which uses the dataset 2 with shufflenet for the batch sizes 32, 64, 128 and 256.
          shufflenet2new_batch_size_128.ipynb
          shufflenet2new_batch_size_256.ipynb
          shufflenet2new_batch_size_32.ipynb
          shufflenet2new_batch_size_64.ipynb
          
  MobileNet-v2
  
    There are 3 models of MobileNet-v2 with 3 datasets
      Model1/MobileNetv2_dataset1.ipynb
      Model2/MobileNetv2_dataset2.ipynb
      Model3/MobileNetv2_dataset3.ipynb

  ResNet18
  
    There are 3 models of ResNet18 with 3 datasets and TSNE model along with the saved models.
      Dataset1
        Final_Dataset1.ipynb
        Model.pt
        Model.pth
      Dataset2
        Final_Dataset2.ipynb
        Model.pt
      Dataset3
        Final_Dataset3.ipynb
        Model.pt
      TSNE
        TSNE_Dataset3.ipynb
  
  ShuffleNet
  
    There are 3 models of ShuffleNet with 3 datasets and TSNE model along with the saved models.
      ShuffleNet_Dataset_1
        model.pt
        model.pth
        shufflenet1.ipynb
      ShuffleNet_Dataset_2
        model.pt
        shufflenet2.ipynb
      ShuffleNet_Dataset_3
        model.pt
        shufflenet3.ipynb

  TSNE
  
    There are 3 TSNE models using 3 datasets along with MobileNet-v2 and ShuffleNet
      Dataset1
        TSNEMobileNetv2_model1.ipynb
        TSNEshufflenet1.ipynb
      Dataset2
        TSNE_MobileNetv2_model2.ipynb
        TSNE_shufflenet2.ipynb
      Dataset3
        TSNE_shufflenet2.ipynb
  
  TransferLearning
  
    There are 2 models with Transfer learning using MobileNet-v2 and ShuffleNet
    Model1
      MobileNetv2_model2.ipynb
    Model2_shuffleNet
      best_model.pt
      transferlearning-shufflenetv2.ipynb

Requirements: 
  The following libraries have been used for the implementation of these 9 models.
    
    * os
    * shutil
    * random
    * os
    * torch
    * torchvision.transforms
    * torchvision.datasets
    * torch.nn
    * torch.optim
    * torchvision.models
    * ImageFolder
    * glob
    * matplotlib.pyplot
    * tqdm
    * numpy
    * drive (google.colab)
    * sklearn.metrics
    * seaborn


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
