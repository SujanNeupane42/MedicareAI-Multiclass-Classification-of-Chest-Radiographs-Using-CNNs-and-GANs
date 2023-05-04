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
