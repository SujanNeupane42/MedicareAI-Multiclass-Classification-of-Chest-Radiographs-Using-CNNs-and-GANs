# MedicareAI-Multiclass-Classification-of-Chest-Radiographs-Using-CNNs-and-GANs

The title of my undergraduate final year project is "MedicareAI". It involves a multiclass classification problem using the Kaggle radiography dataset which has three classes: Normal, Viral Pneumonia, and Covid. However, the dataset suffers from a severe class imbalance. To address this, I trained three classifiers - ResNet50, GoogLeNet, and EfficientNet - and compared their performance across a wide range of evaluation metrics such as precision, recall, F1 score, accuracy, and AUC. In addition to training these models on traditional data augmentation techniques such as random rotation, horizontal flip, and vertical flip, I also used three GAN architectures (DCGAN, WGAN, and WGAN-GP) to augment the data. The quality of each GAN was assessed using Fretchet Inception Distance.

Furthermore, I also trained each CNN model using data augmented with the three GAN architectures. Finally, the performance of each CNN was evaluated on imbalanced data, data with traditional augmentation, and data with DCGAN, WGAN, and WGAN-GP based augmentation. The results indicated that the GoogLeNet model performed the best when trained with WGAN-based augmented data.

Following technologies and tools have been used in this project.


![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-%23013243.svg?style=flat&logo=MySQL&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-%23150458.svg?style=flat&logo=Jupyter&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-%23150458.svg?style=flat&logo=Anaconda&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-%23150458.svg?style=flat&logo=Seaborn&logoColor=white)


Following GIFs showcase image generated at every epoch for 3 GAN architeres. Since each GAN was trained for 1500 epochs, a total of 150 images were stiched together to create GIFs. 

## DCGAN
![DCGAN](./Finalized%20Visualizations/gcgan.gif)

## WGAN
![WGAN](./Finalized%20Visualizations/wgan.gif)

## WGAN-GP
![WGAN-GP](./Finalized%20Visualizations/wgangp.gif)


Now, The Fretchet Inception Distance(FID) for each GAN architeteceture has also been calculated, which is as follows.

## FID Score
![FID](./Finalized%20Visualizations/FID_GANS.jpg)

WGAN performed better than all GANs. Similarly, DCGAN performed worse due to mode collapse and vanishing gradient problem.

After training GANs, new images for imbalanced classes were augmented using each GAN and 3 CNN architectures were trained and their performance was calcualted on the test dataset.

# ResNet50
## ResNet50 Training 
![ResNet50](./Finalized%20Visualizations/ResNet50s_Training_And_Validation_Performance_Variation_Accross_Epochs.jpg)

## ResNet50 performance metrics 
![ResNet50](./Finalized%20Visualizations/ResNet50_Performance_Metrics.jpg)

## ResNet50 Confusion Matrix 
![ResNet50](./Finalized%20Visualizations/ResNet50_Combined_ConfusionMatrix.jpg)

## ResNet50 ROC Curve 
![ResNet50](./Finalized%20Visualizations/ResNet50_Combined_ROC_Curve.jpg)


Similarly, the performance of EfficientNet is also given below.

# EfficientNet
## EfficientNet Training 
![EfficientNet](./Finalized%20Visualizations/EfficientNets_Training_And_Validation_Performance_Variation_Accross_Epochs.jpg)

## EfficientNet performance metrics 
![EfficientNet](./Finalized%20Visualizations/EfficientNet_Performance_Metrics.jpg)

## EfficientNet Confusion Matrix 
![EfficientNet](./Finalized%20Visualizations/EfficientNet_Combined_ConfusionMatrix.jpg)

## EfficientNet ROC Curve 
![EfficientNet](./Finalized%20Visualizations/EfficientNet_Combined_ROC_Curve.jpg)


lets look at the performance of GoogLeNet. Two variants of GoogLeNet was implemented: With and Without Auxiliary classifiers.

# GoogLeNet (Without Auxiliary Classifiers)
## GoogLeNet (Without Auxiliary Classifiers) Training 
![GoogLeNet](./Finalized%20Visualizations/GoogLeNets_Training_And_Validation_Performance_Variation_Accross_Epochs.jpg)

## GoogLeNet (Without Auxiliary Classifiers) performance metrics 
![GoogLeNet](./Finalized%20Visualizations/GoogLeNet_Performance_Metrics.jpg)

## GoogLeNet (Without Auxiliary Classifiers) Confusion Matrix 
![GoogLeNet](./Finalized%20Visualizations/GoogLeNet_Combined_ConfusionMatrix.jpg)

## GoogLeNet (Without Auxiliary Classifiers) ROC Curve 
![GoogLeNet](./Finalized%20Visualizations/GoogLeNets_Combined_ROC_Curve.jpg)

Additionally, GoogLeNet classifier was also trained with auxiliary classifiers enabled during training.


# GoogLeNet (With Auxiliary Classifiers)
## GoogLeNet (With Auxiliary Classifiers) Training 
![GoogLeNet](./Finalized%20Visualizations/GoogLeNetUpdated_Training_And_Validation_Performance_Variation_Accross_Epochs.jpg)

## GoogLeNet (With Auxiliary Classifiers) performance metrics 
![GoogLeNet](./Finalized%20Visualizations/GoogLeNetUpdated_Performance_Metrics.jpg)

## GoogLeNet (With Auxiliary Classifiers) Confusion Matrix 
![GoogLeNet](./Finalized%20Visualizations/GoogLeNetUpdated_Combined_ConfusionMatrix.jpg)

## GoogLeNet (With Auxiliary Classifiers) ROC Curve 
![GoogLeNet](./Finalized%20Visualizations/GoogLeNetUpdated_Combined_ROC_Curve.jpg)

GoogLeNet, without auxiliary classifiers, performed the best as compared to other models. The following table summarizes the performance of GoogLeNet on the validation and test set.

| Models | Precision (Val) | Precision (Test) | Recall (Val) | Recall (Test) | F1 Score (Val) | F1 Score (Test) | AUC (Val) | AUC (Test) | Accuracy (Val) | Accuracy (Test) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GoogLeNet-Imbalanced | 0.9354 | 0.9363 | 0.9333 | 0.9325 | 0.9337 | 0.9325 | 0.9878 | 0.9881 | 93.33 | 93.25 |
| GoogLeNet-Traditional | 0.9308 | 0.9485 | 0.9291 | 0.9458 | 0.9295 | 0.9459 | 0.9883 | 0.9922 | 92.916 | 94.5833 |
| GoogLeNet-DCGAN | 0.9406 | 0.9375 | 0.9383 | 0.9333 | 0.9387 | 0.9331 | 0.9881 | 0.9919 | 93.83 | 93.33 |
| GoogLeNet-WGAN | 0.9378 | 0.9604 | 0.9375 | 0.9591 | 0.9376 | 0.9592 | 0.99 | 0.9947 | 93.75 | 95.92 |
| GoogLeNet-WGANGP | 0.9422 | 0.9584 | 0.9408 | 0.9575 | 0.9411 | 0.9575 | 0.9901 | 0.9933 | 94.08 | 95.75 |

Finally, GoogLeNet model trained with WGAN augmented data was deployed in a web application using Django. The following screenshots contain the predictions make by the system.

## LOGIN
![LOGIN](./Finalized%20Visualizations/Login.jpg)

## Signup
![Signup](./Finalized%20Visualizations/Signup.jpg)

## Prediction
![Prediction](./Finalized%20Visualizations/prediction.jpg)



