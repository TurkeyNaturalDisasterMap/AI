import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Veriyi yükle
data = pd.read_csv("data.csv")
data = data.drop(["Date", "Time‡", "Place"], axis=1)

# Bağımsız ve bağımlı değişkenleri ayır
X = data.drop("Deaths", axis=1)
y = data["Deaths"]

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi standartlaştır
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Lineer regresyon modelini eğit
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Kullanıcıdan girdi al
user_input = []
for feature in X.columns:
    value = float(input(f"Enter value for {feature}: "))
    user_input.append(value)

# Girdileri standartlaştır
user_input = scaler.transform([user_input])

# Modeli kullanarak tahmin yap
user_prediction = linear_model.predict(user_input)

# Tahmin sonucunu yazdır
print(f"Tahmin Edilen Ölüm Sayısı: {user_prediction[0]}")
