# Employee Attrition Prediction using Decision Tree and Random Forest Classification

## Objective

The objective of this project is to predict employee attrition using machine learning classification algorithms. Decision Tree and Random Forest models are developed, trained, and evaluated using the IBM HR Analytics Employee Attrition dataset. The performance of both models is compared using multiple evaluation metrics.

---

## Dataset Link

IBM HR Analytics Employee Attrition & Performance Dataset

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

---

## Libraries Used

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Methodology

1. Loaded the dataset using Pandas.
2. Explored the dataset and identified numerical and categorical features.
3. Checked for missing values.
4. Removed unnecessary columns.
5. Encoded categorical variables using Label Encoding.
6. Split the dataset into 80% training and 20% testing sets.
7. Built a Decision Tree Classifier.
8. Built a Random Forest Classifier with 100 estimators.
9. Evaluated both models using Accuracy, Precision, Recall, and F1-Score.
10. Generated Confusion Matrices for both models.
11. Visualized the Feature Importance of the Random Forest model.
12. Compared the performance of both models.

---

## Results

Both machine learning models successfully predicted employee attrition. Performance was evaluated using Accuracy, Precision, Recall, and F1-Score. Confusion Matrices were generated for both models, and the Random Forest model also provided Feature Importance scores for identifying the most influential employee attributes.

---

## Model Comparison

* Decision Tree is simple, easy to interpret, and fast to train.
* Random Forest combines multiple Decision Trees to improve prediction performance.
* Random Forest generally achieved higher Accuracy, Precision, Recall, and F1-Score than the Decision Tree model.
* Random Forest is more robust and less prone to overfitting.

---

## Conclusion

This project demonstrated the use of Decision Tree and Random Forest classifiers for employee attrition prediction. Both models produced satisfactory results; however, the Random Forest classifier performed better due to its ensemble learning approach, which improves generalization and reduces overfitting. Decision Trees are easier to interpret but may overfit the training data, while Random Forest models require more computational resources and are less interpretable. Overall, Random Forest proved to be the more reliable model for predicting employee attrition.

---

## Bonus Challenge (Hyperparameter Tuning)

A simple hyperparameter tuning experiment was performed by modifying the **max_depth** parameter of the Decision Tree classifier.

Example:

```python
DecisionTreeClassifier(max_depth=5, random_state=42)
```

### Observation

Reducing the maximum depth helped control overfitting and produced a model that generalized better on the testing dataset. Very small values of `max_depth` may lead to underfitting, while very large values may increase overfitting.
