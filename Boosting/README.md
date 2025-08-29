ensemble technique in ML  
- combine multiple weak learners 
- highly accurate model 

it is a sequential algo 
 reduces bias 

 TYPES :
1. AdaBoost (Adaptive Boosting)
First boosting algorithm.
Works by assigning weights to each data point.
Misclassified points get higher weight so the next learner focuses on them.

2. Gradient Boosting (GBM)
Instead of adjusting weights, it reduces errors using gradient descent.
Each new model is trained on the residual errors of the previous model.

3.XGBoost
“Extreme Gradient Boosting” – faster & regularized version of GBM.
Uses regularization to prevent overfitting.

4.LightGBM & CatBoost
More advanced, faster for big data.
LightGBM uses leaf-wise tree growth.
CatBoost is great for categorical data

