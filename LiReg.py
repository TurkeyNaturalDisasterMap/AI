import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv("data.csv")
data=data.drop(["Date","Time‡","Place"],axis=1)

X = data.drop("Deaths", axis=1)
y = data["Deaths"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Ortalama Kare Hata (MSE): {mse}")
print(f"R^2 Skoru: {r2}")

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Gerçek Değerler', linestyle='-', marker='o', markersize=5)
plt.plot(y_pred, label='Tahmin Edilen Değerler', linestyle='-', marker='o', markersize=5)
plt.xlabel('Örnekler')
plt.ylabel('Değerler')
plt.title('Lineer Regresyon - Gerçek vs. Tahmin Edilen Değerler')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("LineerRegAcc.png")
plt.show()