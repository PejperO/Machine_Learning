import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Wczytanie danych z pliku
with open("dane1.txt", "r") as file:
    data1 = file.readlines()

with open("dane2.txt", "r") as file:
    data2 = file.readlines()

# Przetwarzanie danych do postaci potrzebnej do dopasowania modelu
x_data1, y_data1 = [], []
for line in data1:
    x, y = map(float, line.split())  # Zakładamy, że dane są oddzielone spacją
    x_data1.append(x)
    y_data1.append(y)

x_data2, y_data2 = [], []
for line in data2:
    x, y = map(float, line.split())  # Zakładamy, że dane są oddzielone spacją
    x_data2.append(x)
    y_data2.append(y)

# Konwersja do numpy array
x_data1, y_data1 = np.array(x_data1), np.array(y_data1)
x_data2, y_data2 = np.array(x_data2), np.array(y_data2)

# Podział danych na dane treningowe i testowe
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data1, y_data1, test_size=0.2, random_state=42)
x_train2, x_test2, y_train2, y_test2 = train_test_split(x_data2, y_data2, test_size=0.2, random_state=42)

# Dopasowanie modelu liniowego do danych treningowych
X_train1 = x_train1.reshape(-1, 1)
X_train2 = x_train2.reshape(-1, 1)

# Użycie modelu LinearRegression
model1 = LinearRegression()
model1.fit(X_train1, y_train1)
a1 = model1.coef_[0]
b1 = model1.intercept_

model2 = LinearRegression()
model2.fit(X_train2, y_train2)
a2 = model2.coef_[0]
b2 = model2.intercept_

# Ocena modelu na danych testowych
X_test1 = x_test1.reshape(-1, 1)
X_test2 = x_test2.reshape(-1, 1)

y_pred1 = model1.predict(X_test1)
precision1 = r2_score(y_test1, y_pred1)

y_pred2 = model2.predict(X_test2)
precision2 = r2_score(y_test2, y_pred2)

# Model wielomianowy stopnia 2
poly_features = PolynomialFeatures(degree=2)
X_train_poly1 = poly_features.fit_transform(X_train1)
X_train_poly2 = poly_features.fit_transform(X_train2)

X_test_poly1 = poly_features.transform(X_test1)
X_test_poly2 = poly_features.transform(X_test2)

model_poly1 = LinearRegression()
model_poly1.fit(X_train_poly1, y_train1)

model_poly2 = LinearRegression()
model_poly2.fit(X_train_poly2, y_train2)

# Ocena modelu wielomianowego na danych testowych
y_pred_poly1 = model_poly1.predict(X_test_poly1)
precision_poly1 = r2_score(y_test1, y_pred_poly1)

y_pred_poly2 = model_poly2.predict(X_test_poly2)
precision_poly2 = r2_score(y_test2, y_pred_poly2)

# Wykres prezentujący punkty z danych i dopasowane modele
plt.figure(figsize=(14, 7))

# Wykres dla pierwszego zbioru danych
plt.subplot(1, 2, 1)
plt.scatter(x_train1, y_train1, color='blue', label='Dane treningowe')
plt.scatter(x_test1, y_test1, color='green', label='Dane testowe')
x_range1 = np.linspace(min(x_train1), max(x_train1), 100).reshape(-1, 1)
plt.plot(x_range1, model1.predict(x_range1), color='red', label=f'Liniowy: y = {a1:.2f}x + {b1:.2f}, r^2={precision1:.2f}')
plt.plot(x_range1, model_poly1.predict(poly_features.transform(x_range1)), color='orange', label=f'Wielomianowy: r^2={precision_poly1:.2f}')
plt.xlabel('Wartość X')
plt.ylabel('Wartość Y')
plt.title('Porównanie modeli - Dane 1')
plt.legend()

# Wykres dla drugiego zbioru danych
plt.subplot(1, 2, 2)
plt.scatter(x_train2, y_train2, color='blue', label='Dane treningowe')
plt.scatter(x_test2, y_test2, color='green', label='Dane testowe')
x_range2 = np.linspace(min(x_train2), max(x_train2), 100).reshape(-1, 1)
plt.plot(x_range2, model2.predict(x_range2), color='red', label=f'Liniowy: y = {a2:.2f}x + {b2:.2f}, r^2={precision2:.2f}')
plt.plot(x_range2, model_poly2.predict(poly_features.transform(x_range2)), color='orange', label=f'Wielomianowy: r^2={precision_poly2:.2f}')
plt.xlabel('Wartość X')
plt.ylabel('Wartość Y')
plt.title('Porównanie modeli - Dane 2')
plt.legend()

plt.tight_layout()
plt.savefig('porownanie_modeli.png')
plt.show()
