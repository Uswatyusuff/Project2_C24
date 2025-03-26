# Project2_C24
This project focuses on crowd counting using various deep learning models and datasets. The project follows a structured approach to data collection, preprocessing, model training, and evaluation.

# Problem Statement
Accurate crowd counting is crucial for public safety, urban planning, and event management. The objective of this project is to develop a robust crowd counting model that can generalize across different crowd densities and environmental conditions. The models trained in this project aim to provide high accuracy while maintaining computational efficiency.

The CRISP-DM methodology was followed for the successful completion of this project, ensuring a structured approach to data understanding, preparation, modeling, evaluation, and deployment.

# Functional And Non-Functional Requirements
![image alt](https://github.com/Uswatyusuff/Project2_C24/blob/0fe9dee3f500213cd8e8a95a2557026e6fd01b05/Functional%20Requirements.png) ![image alt](https://github.com/Uswatyusuff/Project2_C24/blob/0fe9dee3f500213cd8e8a95a2557026e6fd01b05/Non%20Funtional%20Requirements.png)

# ML Pipeline
![image alt](https://github.com/Uswatyusuff/Project2_C24/blob/ee9ffb2cca0f06527c5efa1991247e31dabca80b/ML%20Pipeline.png)

# Datasets Used
The project utilizes multiple crowd counting datasets to ensure robustness:

JHU-CROWD++: A large-scale crowd counting dataset with diverse crowd densities.

ShanghaiTech (Part A & B): Part A contains highly dense crowd images, while Part B consists of relatively sparse crowd images.

UCF-QNRF: A challenging dataset with high-resolution images and varying crowd densities.

Custom datasets: Includes data collected from the Giraffe event and graduation event, providing real-world scenarios for model fine-tuning.

# Data Preprocessing

Cleaning & Annotation

Resizing images for model compatibility.

Generating density maps using Gaussian kernels.

Normalizing and augmenting datasets for improved generalization.

# Crowd Counting Models Implemented

1. CSRNet (Convolutional Neural Network for Crowd Counting)

A deep learning-based approach leveraging dilated convolutions for density estimation.

Trained from scratch and fine-tuned using JHU-CROWD++ and custom datasets.

2. DM-Count (Differentiable Mean Counting Network)

A density-based model using optimal transport for improved accuracy.

Evaluated on JHU-CROWD++, UCF-QNRF, and ShanghaiTech datasets.

3. YOLO-based Crowd Counting

Utilizes object detection techniques to estimate the number of people in an image.

Trained on ShanghaiTech Part B for detecting sparse crowds.

4. Traditional Machine Learning Approaches

Implemented regression-based models for baseline comparisons.

Extracted handcrafted features and applied Random Forest and SVM for initial tests.

# Model Training & Evaluation
Models were trained using different train-test splits (80-20, 70-30) to assess generalization.

Performance was evaluated using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Density Map Visualization.

Hyperparameter tuning was performed using Grid Search and Bayesian Optimization.

# Project Management

The project was managed using Agile methodologies, with regular updates and iterative improvements. The following project management resources were maintained:

Ethical Mind Maps: Addressing ethical concerns in crowd counting applications.

Change Log: Documenting modifications and improvements.

Functional & Non-functional Requirements: Defining project scope.

ML Pipeline: Outlining data flow from preprocessing to deployment.

Risk Assessment: Identifying potential risks and mitigation strategies.

# Project Outcome

Developed a high-accuracy crowd counting model.

Created a user-friendly interface for real-time crowd estimation.

Provided insights for public event management based on real-world data.

# Areas for Improvement

Exploring transformer-based models for enhanced accuracy.

Incorporating temporal analysis for video-based crowd estimation.

Expanding dataset collection for improved model generalization.


