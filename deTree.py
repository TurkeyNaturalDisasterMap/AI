import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("data.csv")
data = data.drop(["Date", "Time‡", "Place"], axis=1)

X = data.drop("Deaths", axis=1)
y = data["Deaths"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler() # 1 ile 0 arasına verileri çeker
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a DecisionTreeRegressor
tree_model = DecisionTreeRegressor(random_state=42)

# Define hyperparameters to search
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(tree_model, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_tree_model = grid_search.best_estimator_

# Predict using the best model
y_pred = best_tree_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Ortalama Kare Hata (MSE): {mse}")
print(f"R^2 Skoru: {r2}")
print("En iyi parametreler:", grid_search.best_params_)

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Gerçek Değerler', linestyle='-', marker='o', markersize=5)
plt.plot(y_pred, label='Tahmin Edilen Değerler', linestyle='-', marker='o', markersize=5)
plt.xlabel('Örnekler')
plt.ylabel('Değerler')
plt.title('Decision Tree Regresyon - Gerçek vs. Tahmin Edilen Değerler')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("DecisionTreeRegAcc.png")
plt.show()
