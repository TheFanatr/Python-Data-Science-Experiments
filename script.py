import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime

train_data = pd.read_csv('train.csv')
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

train_X, val_X, train_y, val_y = train_test_split(train_data[features], train_data.SalePrice, random_state=1)

print(train_X.to_string)
print(val_X.to_string)
print(train_y.to_string)
print(val_y.to_string)

# Decision tree regression with default value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Decision tree regression with best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Random forest regression with random_state as 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# Test data prediction generation for submission
test_data = pd.read_csv('test.csv')
rf_model_test_predictions = rf_model.predict(test_data[features])
formatted_prediction_data = pd.DataFrame({'Id': test_data.Id, 'SalePrice': rf_model_test_predictions})

print(rf_model.feature_importances_)

print(test_data.to_string)
print(formatted_prediction_data.to_string)

formatted_prediction_data.to_csv('results\\prediction-{}-{:.0f}.csv'.format(datetime.now().timestamp(), rf_val_mae), index=False)