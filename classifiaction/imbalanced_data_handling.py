'''
Why is Imbalance a Problem?

Classifier is biased toward the majority class.

Minority class (often more important, like fraud/disease) is ignored.

Metrics like accuracy become misleading → better to use Precision, Recall, F1, AUc 



'''

#1. Resampling Methods
# Oversampling minority class → e.g., SMOTE (Synthetic Minority Oversampling Technique) generates synthetic examples.

# Undersampling majority class → randomly remove some majority samples to balance.
from imblearn.over_sampling import SMOTE



smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

'''
2. Change the Performance Metric

Instead of accuracy, use:
✅ Precision
✅ Recall
✅ F1 Score
✅ ROC-AUC

3. Use Class Weights in Models

Most algorithms (Logistic Regression, Random Forest, XGBoost, etc.) allow a class_weight parameter.
This penalizes misclassification of the minority class more heavily.'''
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')


4.Ensemble Methods

Use Bagging & Boosting (Random Forest, XGBoost, LightGBM) with class weights.

Boosting (like XGBoost) often works well with imbalanced data.