import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Завантаження основного датасету
dataset = pd.read_csv('D:/Git/Kursova/data/games.csv')
dataset = dataset[['BGGId', 'Name', 'AvgRating', 'YearPublished', 'GameWeight', 'Cat_Thematic', 'Cat_Strategy', 'Cat_War',
             'Cat_Family', 'Cat_CGS', 'Cat_Abstract', 'Cat_Party']]

# Аналіз данних
print("            Інформація")
print(dataset.info())
print()
print("    Опис")
print(dataset.describe())
print()
print("    Перші 10 рядків")
print(dataset.head(10))

year_info = dataset['YearPublished'].value_counts().sort_index()
plt.plot(year_info.index, year_info.values, marker='o')
plt.xlabel('Роки')
plt.ylabel('Кількість ігор')
plt.title('Кількість ігор за рік')
plt.grid(True)
plt.show()

# Підготовка даних
dataset = dataset[dataset['YearPublished'] >= 1900]

year_info = dataset['YearPublished'].value_counts().sort_index()
plt.plot(year_info.index, year_info.values, marker='o')
plt.xlabel('Роки')
plt.ylabel('Кількість ігор')
plt.title('Кількість ігор за рік')
plt.grid(True)
plt.show()

dataset = dataset.drop_duplicates(subset=['Name', 'BGGId'])

# Розділення даних на вхідні ознаки (X) та цільову змінну (y)
X = dataset[['YearPublished', 'GameWeight', 'Cat_Thematic', 'Cat_Strategy', 'Cat_War',
             'Cat_Family', 'Cat_CGS', 'Cat_Abstract', 'Cat_Party']]
y = dataset['AvgRating']

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Розділення даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

#-------------------------------------------Лінійна регресія----------------------------------------------
# Ініціалізація моделі лінійної регресії
lin = LinearRegression()

# Навчання моделі на тренувальних даних
lin.fit(X_train, y_train)

# Прогнозування на тестовому наборі
y_pred_lin = lin.predict(X_test)

# Оцінка моделі
mse = mean_squared_error(y_test, y_pred_lin)
r2 = r2_score(y_test, y_pred_lin)
print('Лінійна регресія')
print('Mean Squared Error:', mse)
print('R^2 Score:', r2)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_lin)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Справжні значення')
ax.set_ylabel('Передбачені значення')
plt.show()

#-------------------------------------------Decision Tree Regression----------------------------------------------
# Ініціалізація моделі лінійної регресії
DRG = DecisionTreeRegressor()

# Навчання моделі на тренувальних даних
DRG.fit(X_train, y_train)

# Прогнозування на тестовому наборі
y_pred_drg = DRG.predict(X_test)

# Оцінка моделі
mse = mean_squared_error(y_test, y_pred_drg)
r2 = r2_score(y_test, y_pred_drg)
print('Decision Tree Regression')
print('Mean Squared Error:', mse)
print('R^2 Score:', r2)

# Вивід графіку
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_drg)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Справжні значення')
ax.set_ylabel('Передбачені значення')
plt.show()

#-------------------------------------------Random Forest Regression----------------------------------------------
# Ініціалізація моделі лінійної регресії
RFR = RandomForestRegressor(random_state=20)

# Навчання моделі на тренувальних даних
RFR.fit(X_train, y_train)

# Прогнозування на тестовому наборі
y_pred_RFR = RFR.predict(X_test)

# Оцінка моделі
mse = mean_squared_error(y_test, y_pred_RFR)
r2 = r2_score(y_test, y_pred_RFR)
print('Random Forest Regression')
print('Mean Squared Error:', mse)
print('R^2 Score:', r2)

# Вивід графіку
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_RFR)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Справжні значення')
ax.set_ylabel('Передбачені значення')
plt.show()

#-------------------------------------------Загальні результати----------------------------------------------
print("Порівняння R2 та MSE")
print("R2: \n\tLinearRegression: {0}\n\tDecision Tree Regression: {1}\n\tRandomForestRegressor: {2}"
      .format(r2_score(y_test, y_pred_lin), r2_score(y_test, y_pred_drg), r2_score(y_test, y_pred_RFR)))
print("MSE: \n\tLinearRegression: {0}\n\tDecision Tree Regression: {1}\n\tRandomForestRegressor: {2}"
      .format(mean_squared_error(y_test, y_pred_lin), mean_squared_error(y_test, y_pred_drg), mean_squared_error(y_test, y_pred_RFR)))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].scatter(y_test, y_pred_lin)
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax[0].set_xlabel('Справжні значення')
ax[0].set_ylabel('Передбачені значення')
ax[0].set_title('LinearRegression')

ax[1].scatter(y_test, y_pred_drg)
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax[1].set_xlabel('Справжні значення')
ax[1].set_ylabel('Передбачені значення')
ax[1].set_title('DecisionTreeRegressor')

ax[2].scatter(y_test, y_pred_RFR)
ax[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax[2].set_xlabel('Справжні значення')
ax[2].set_ylabel('Передбачені значення')
ax[2].set_title('RandomForestRegressor')

plt.tight_layout()
plt.show()
