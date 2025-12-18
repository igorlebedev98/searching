#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error


# # Практическая работа №4. Регрессионный анализ

# ## 1. Описательный анализ
# Очищенный набор данных содержит информацию о цене, трансмиссии, пробеге, типе топлива, дорожном налоге, милях на галлон (миль на галлон) и объеме двигателя.
# В моем варианте данные содержат информацию об автомобилях hyundai.
# 
# ### Показатели:
# **model** - модель  
# **year** - год сборки  
# **price** - цена  
# **transmission** - тип коробки передач  
# **mileage** - пробег  
# **fuelType** - тип топлива  
# **tax** - налог на авто  
# **mpg** - расход топлива  
# **engineSize** - объем двигателя  
# 

# In[2]:


data = pd.read_csv("hyundi.xls")
data


# In[3]:


has_missing_values = data.isnull().values.any()

if has_missing_values:
    print("В данных есть пропуски.")
else:
    print("В данных нет пропусков.")


# In[4]:


sns.pairplot(data)


# In[5]:


plt.figure(figsize = (20, 4))
sns.countplot(x = "model", data = data)
plt.figure(figsize = (20, 4))
sns.boxplot(x = "model", y = "price", data = data)


# In[6]:


plt.figure(figsize = (20, 4))
sns.countplot(x = "transmission", data = data)
plt.figure(figsize = (20, 4))
sns.boxplot(x = "transmission", y = "price", data = data)


# In[7]:


plt.figure(figsize = (20, 4))
sns.countplot(x = "fuelType", data = data)
plt.figure(figsize = (20, 4))
sns.boxplot(x = "fuelType", y = "price", data = data)


# In[8]:


# Отбор только числовых столбцов
numeric_data = data.select_dtypes(include=['number'])

# Коэффициенты корреляции
plt.figure(figsize=(20, 10))
sns.heatmap(numeric_data.corr(method='kendall'), annot=True, cmap='coolwarm')
plt.show()


# In[9]:


## С помощью метода квантилей, найдем и уберем аномалии
q1 = numeric_data.quantile(0.25)
q3 = numeric_data.quantile(0.75)
iqr = q3 - q1

outliers = numeric_data[((numeric_data < (q1 - 1.5 * iqr)) | (numeric_data > (q3 + 1.5 * iqr))).any(axis=1)]
print(outliers)


# ## 2. Построение базовых регрессионых моделей.

# In[10]:


df = pd.DataFrame(data)
X = df.select_dtypes(include=['number'])  # Признаки (оставляем только числа)
y = df['price']  # Целевая переменная

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[11]:


# Constant
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
mean_value = y_train.mean()
y_pred = linear_model.predict(X_test)

print("Предсказанное среднее значение:", mean_value)
print("Ошибка модели на тестовых данных:", metrics.mean_squared_error(y_test, y_pred, squared=False))
print("Оценка качества модели на тестовых данных:", metrics.r2_score(y_test, y_pred))


# In[12]:


# ConstantByGroup

# Расчет среднего значения цены по модели автомобиля
mean_price_by_model = df.groupby('model')['price'].mean().reset_index()

print(mean_price_by_model)


# In[13]:


# OneParamModel
# Выбор показателя для модели
X = df[['mileage']]  # 'mileage' - единственный признак
y = df['price']      # Целевая переменная

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Ошибка модели на тренировочных данных:", metrics.mean_squared_error(y_train, y_pred_train, squared=False))
print("Ошибка модели на тестовых данных:", metrics.mean_squared_error(y_test, y_pred_test, squared=False))
print("Оценка качества модели на тренировочных данных:", metrics.r2_score(y_train, y_pred_train))
print("Оценка качества модели на тестовых данных:", metrics.r2_score(y_test, y_pred_test))

# Вывод коэффициентов модели
print("Коэффициенты модели:", model.coef_)
print("Свободный член (intercept):", model.intercept_)


# ## 3. Построение линейных регрессионых моделей для прогноза целевого показателя

# In[14]:


# Без нормализации

X = df[['mileage', 'engineSize', 'mpg']] 
y = df['price'] # Целевая переменная 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Ошибка модели на тренировочных данных:", metrics.mean_squared_error(y_train, y_pred_train, squared=False))
print("Ошибка модели на тестовых данных:", metrics.mean_squared_error(y_test, y_pred_test, squared=False))
print("Коэффициент детерминации на тренировочных данных:", metrics.r2_score(y_train, y_pred_train))
print("Коэффициент детерминации на тестовых данных:", metrics.r2_score(y_test, y_pred_test))


# In[15]:


# Оценка важности показателей для линейной регриссионой модели

coefficients = model.coef_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
print(importance_df)


# In[16]:


# С нормолизацей

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_norm, y_train)

y_pred_train = model.predict(X_train_norm)
y_pred_test = model.predict(X_test_norm)

# Оценка модели
print("Ошибка модели на тренировочных данных:", metrics.mean_squared_error(y_train, y_pred_train, squared=False))
print("Ошибка модели на тестовых данных:", metrics.mean_squared_error(y_test, y_pred_test, squared=False))
print("Коэффициент детерминации на тренировочных данных:", metrics.r2_score(y_train, y_pred_train))
print("Коэффициент детерминации на тестовых данных:", metrics.r2_score(y_test, y_pred_test))


# In[ ]:


# Без регуляризации \ с регуляризацией

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель без регуляризации
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)


# Модель с Lasso регуляризацией
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)
y_pred_lasso = lasso_reg.predict(X_test)


# Модель с Ridge регуляризацией
ridge_reg = Ridge(alpha=0.1)
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_test)


print(f'Ошибка модели без регуляризации: {mean_squared_error(y_test, y_pred)}')
print(f'Ошибка модели с Lasso регуляризацией: {mean_squared_error(y_test, y_pred_lasso)}')
print(f'Ошибка модели с Ridge регуляризацией: {mean_squared_error(y_test, y_pred_ridge)}')


# In[18]:


# Только количесвтенные показатели
df = pd.DataFrame(data)

X = df[['mileage', 'engineSize', 'mpg']]  # Количественные показатели
y = df['price']  # Целевая переменная

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_train = lin_reg.predict(X_train)
y_pred_test = lin_reg.predict(X_test)

print(f"Ошибка на тренировочных данных: {mean_squared_error(y_train, y_pred_train, squared=False)}")
print(f"Ошибка на тестовых данных: {mean_squared_error(y_test, y_pred_test, squared=False)}\n")


# In[ ]:


# Только качественные показатели

df = pd.DataFrame(data)

# Преобразование категориальных переменных в числовые
df = pd.get_dummies(df, columns=['transmission', 'fuelType'], drop_first=True)


X = df[['transmission_Manual', 'transmission_Other', 'transmission_Semi-Auto',
       'fuelType_Hybrid', 'fuelType_Other', 'fuelType_Petrol']]
y = df['price']  # Целевая переменная

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_train = lin_reg.predict(X_train)
y_pred_test = lin_reg.predict(X_test)

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)
y_pred_lasso_train = lasso_reg.predict(X_train)
y_pred_lasso_test = lasso_reg.predict(X_test)

print("Модель без регуляризации:")
print(f"Ошибка на тренировочных данных: {mean_squared_error(y_train, y_pred_train, squared=False)}")
print(f"Ошибка на тестовых данных: {mean_squared_error(y_test, y_pred_test, squared=False)}\n")

print("Модель с Lasso регуляризацией:")
print(f"Ошибка на тренировочных данных: {mean_squared_error(y_train, y_pred_lasso_train, squared=False)}")
print(f"Ошибка на тестовых данных: {mean_squared_error(y_test, y_pred_lasso_test, squared=False)}\n")


# ## 4. Применение других регрессионных моделей

# In[ ]:


# Полиномиальная регриссионная модель

poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Модель без регуляризации
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_train = poly_reg.predict(X_train_poly)
y_pred_test = poly_reg.predict(X_test_poly)

# Оценка модели без регуляризации
mse_no_reg_train = mean_squared_error(y_train, y_pred_train, squared=False)
mse_no_reg_test = mean_squared_error(y_test, y_pred_test, squared=False)

# Модель с Ridge регуляризацией
ridge_reg = Ridge(alpha=0.1)
ridge_reg.fit(X_train_poly, y_train)
y_pred_ridge_train = ridge_reg.predict(X_train_poly)
y_pred_ridge_test = ridge_reg.predict(X_test_poly)

mse_reg_train = mean_squared_error(y_train, y_pred_ridge_train, squared=False)
mse_reg_test = mean_squared_error(y_test, y_pred_ridge_test, squared=False)

print("Полиномиальная модель без регуляризации:")
print(f"Ошибка на тренировочных данных: {mse_no_reg_train}")
print(f"Ошибка на тестовых данных: {mse_no_reg_test}\n")

print("Полиномиальная модель с Ridge регуляризацией:")
print(f"Ошибка на тренировочных данных: {mse_reg_train}")
print(f"Ошибка на тестовых данных: {mse_reg_test}")


# In[21]:


# Модель на основе дерева решений
df = pd.DataFrame(data)

X = df[['mileage', 'mpg', 'engineSize']] 
y = df['price'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели дерева решений
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

y_pred_train = tree_reg.predict(X_train)
y_pred_test = tree_reg.predict(X_test)

print(f"Ошибка на тренировочных данных: {mean_squared_error(y_train, y_pred_train, squared=False)}")
print(f"Ошибка на тестовых данных: {mean_squared_error(y_test, y_pred_test, squared=False)}")


# In[25]:


# Оценка важности показателей для модели на основе дерева решений
importances = tree_reg.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)

plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.show()


# In[23]:


# Создание модели случайного леса
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

y_pred_train = rf_reg.predict(X_train)
y_pred_test = rf_reg.predict(X_test)

print(f"Ошибка на тренировочных данных: {mean_squared_error(y_train, y_pred_train, squared=False)}")
print(f"Ошибка на тестовых данных: {mean_squared_error(y_test, y_pred_test, squared=False)}")


# In[ ]:


# Оценка важности показателей для модели случайного леса

importances = rf_reg.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.show()


# По важности показателем, сразу заметен engine size. В линейной регрессии он имеет очень большое значение относительно других признаках. В других моделях он сохраняет самую большую важность, однако уже не так сильно доминирует над друними признаками. Также в линейной регресии признаки mpg и milages имеют отрицаительное значение, что логично, так как в основном большой пробег и расход являются негативным фактором. В модели случайного леса и дерева решений показывается лишь их важность.
# 
# Нормализация данных никак не повлияла на увеличение точности модели. 
# Регуляризация немного улучшила точность при использовании полиномиальной регриссионной модель, но только в диапазоне 1%.
