import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#read data
df = pd.read_csv("melb_data.csv")

#prediction target
y = df.Price

#training data, validation data
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = df[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

#build model
my_model = DecisionTreeRegressor()
my_model.fit(train_X, train_y)
model_prediction = my_model.predict(val_X)

#split data
temp_df = df.loc[:100, 'Address': 'Bathroom']
temp_df.to_csv('test.csv')

#calculate MAE
mae = mean_absolute_error(val_y, model_prediction)
print("Mean absolute error with default max leaf nodes: ", mae)

#find best max_leaf_nodes
def get_mae(mln, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=mln, random_state=1)
    model.fit(train_X, train_y)
    pre = model.predict(val_X)
    MAE = mean_absolute_error(val_y, pre)
    return MAE

for i in [5, 50, 500, 5000]:
    mae = get_mae(i, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %s\t Mean absolute error: %s" %(i, mae))

result = {i: get_mae(i, train_X, val_X, train_y, val_y) for i in [5, 50, 500, 5000]} 
best_tree_size = min(result, key=result.get)
print("This is the best tree size: %s with MAE: %s" %(best_tree_size, get_mae(best_tree_size, train_X, val_X, train_y, val_y)))