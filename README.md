# Machine_Learning

This repository contains two comprehensive machine learning projects implemented using Python and the scikit-learn library. The projects involve creating synthetic datasets, training multiple models, combining models using ensemble methods, and comparing linear and polynomial regression models.

## Project Overview

### Ensemble Learning with Voting Classifier
- Created a synthetic dataset using the make_moons function.
- Split the dataset into training and testing sets.
- Trained three classifiers: Logistic Regression, SVM, and Random Forest.
- Combined these classifiers into a Voting Classifier.
- Evaluated the ensemble model and visualized the decision boundary.

### Comparison of Linear and Polynomial Regression Models
- Loaded datasets from text files.
- Split the datasets into training and testing sets.
- Applied a linear regression model.
- Applied a polynomial regression model.
- Compared the performance of both models using R² score.
- Visualized the regression lines and the data points.

## Results
### Ensemble Learning with Voting Classifier
The Voting Classifier combined the strengths of Logistic Regression, SVM, and Random Forest classifiers, resulting in improved accuracy. The decision boundary for the Voting Classifier was visualized to show how it differentiates between classes.

![voting_classifier](https://github.com/user-attachments/assets/6fc343b7-ae36-4c66-a8fd-6ceb9a9033b0)

### Comparison of Linear and Polynomial Regression Models
The comparison between linear and polynomial regression models showed that the polynomial model often fit the data better, resulting in higher R² scores. The visualizations demonstrated the difference in model fitting for two different datasets.

![porownanie_modeli](https://github.com/user-attachments/assets/9ce3852e-2dc0-4cbb-8339-110ce62b9ce9)

## What I Learned
- scikit-learn
- NumPy
- Matplotlib
- How to create and split synthetic datasets.
- The implementation and benefits of ensemble learning methods.
- The process of training and comparing linear and polynomial regression models.
- Visualization techniques for understanding model performance and decision boundaries.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
