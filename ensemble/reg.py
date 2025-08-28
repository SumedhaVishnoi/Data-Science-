from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error

# Dataset
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
lr = LinearRegression()
dt = DecisionTreeRegressor()

# Voting Regressor (Averaging)
voting_reg = VotingRegressor([('lr', lr), ('dt', dt)])
voting_reg.fit(X_train, y_train)

y_pred = voting_reg.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
