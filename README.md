# MedicareAI-Multiclass-Classification-of-Chest-Radiographs-Using-CNNs-and-GANs
Name: Sujan Neupane  ID: 2058939 Supervisor: Dinesh Saud


The title of my undergraduate final year project is "MedicareAI". It involves a multiclass classification problem using the Kaggle radiography dataset which has three classes: Normal, Viral Pneumonia, and Covid. However, the dataset suffers from a severe class imbalance. To address this, I trained three classifiers - ResNet50, GoogLeNet, and EfficientNet - and compared their performance across a wide range of evaluation metrics such as precision, recall, F1 score, accuracy, and AUC. In addition to training these models on traditional data augmentation techniques such as random rotation, horizontal flip, and vertical flip, I also used three GAN architectures (DCGAN, WGAN, and WGAN-GP) to augment the data. The quality of each GAN was assessed using Fretchet Inception Distance.

Furthermore, I also trained each CNN model using data augmented with the three GAN architectures. Finally, the performance of each CNN was evaluated on imbalanced data, data with traditional augmentation, and data with DCGAN, WGAN, and WGAN-GP based augmentation. The results indicated that the GoogLeNet model performed the best when trained with WGAN-based augmented data.
