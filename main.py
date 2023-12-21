import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv("data.csv")
data=data.drop(["Date","Time‡","Place"],axis=1)

X = data.drop("Mag", axis=1)
y = data["Mag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(units=256, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=128,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=64,activation="relu"))
model.add(Dense(units=32,activation="relu"))
model.add(Dense(units=16,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=8,activation="relu"))
model.add(Dense(units=4,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=2,activation="relu"))
model.add(Dense(units=1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=128, batch_size=32, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Ortalama Kare Hata (MSE): {mse}")
print(f"R^2 Skoru: {r2}")


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, c='blue', marker='o', label='Gerçek vs. Tahmin')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek Değerler vs. Tahmin Edilen Değerler')
plt.legend(loc='best')

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())
plt.plot([min_value, max_value], [min_value, max_value], c='red', linestyle='--', label='45 Derece Doğru')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("acc.png")
plt.show()






