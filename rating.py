#!/usr/bin/env python
# coding: utf-8

# In[2]:


print(1 + (ord('I') + ord('L')) % 6)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Работа №4. Алгоритмы снижения размерности.

# Был выбран набор данных о семи видах сухих бобов ("Dry_Bean_Dataset.xlsx"). В наборе есть 16 характеристик для 7 различных зарегистрированных сухих бобов.

# In[3]:


data = pd.read_excel("Dry_Bean_Dataset.xlsx").dropna()
# Удаление признака класса
X = data.drop(columns=['Class'])
y = data['Class']
# Стандартизация данных
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ## Задача 1

# In[4]:


# диаграмма распределения объектов по классам
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='Class', palette='Paired')
plt.title('Диаграмма распределение объектов по классам')
plt.xlabel('Класс бобов')
plt.ylabel('Кол-во объектов')
plt.xticks(rotation=45)
plt.show()


# ## Задача 2

# In[ ]:


from minisom import MiniSom
# модель Кохонена с 100 нейронами (10x10)
som_size = 10  # Размер карты 10x10
som = MiniSom(x=som_size, y=som_size, input_len=X_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.train(X_scaled, num_iteration=1000)  # Обучение на 100 итераций


# In[13]:


plt.figure(figsize=(10, 8))
plt.title('Распределение объектов по ячейкам карты')
sns.heatmap(som.distance_map().T, cmap='Paired', cbar=True)
plt.show()


# In[ ]:


# создадим отображения меток классов
class_labels = y.unique()
class_mapping = {label: idx for idx, label in enumerate(class_labels)}


# In[21]:


# Распределение объектов по классам для каждой ячейки
win_map = som.win_map(X_scaled)
class_distribution = {i: {} for i in range(som_size * som_size)}  # 100 ячеек
y = data['Class'].reset_index(drop=True) 
# Для всех данных получаем индекс ячейки на карте 
for idx, x in enumerate(X_scaled):
    w = som.winner(x)
     # переводим координаты в индекс и получаем метки классов
    cell_index = w[0] * som_size + w[1] 
    label = y.iloc[idx]
    if label not in class_distribution[cell_index]:
        class_distribution[cell_index][label] = 0
    class_distribution[cell_index][label] += 1 

# Создаем массив для хранения кол-во объектов каждого класса для каждой ячейки карты.
num_classes = len(set(y))
class_counts = np.zeros((som_size, som_size, num_classes))

for cell_index, counts in class_distribution.items():
    for label, count in counts.items():
        class_counts[cell_index // som_size, cell_index % som_size, class_mapping[label]] = count

# Визалиазация результатов
num_rows = (num_classes // 2) + (num_classes % 2 > 0) 
num_cols = min(num_classes, 2)

plt.figure(figsize=(12, 10))
for i in range(num_classes):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.heatmap(class_counts[:, :, i], annot=True, fmt=".0f", cmap='Blues')
    plt.title(f'Класс {i}')
plt.tight_layout()
plt.show()


# In[ ]:


# 7. Сравнение объектов двух ячеек карты по средним значениям показателе
# Выбираем две ячейки для сравнения
cell1 = (2, 3)  # Пример ячейки 1
cell2 = (4, 8)  # Пример ячейки 2

# Получение индексов объектов, попадающих в выбранные ячейки
indices_cell1 = [i for i, x in enumerate(X_scaled) if som.winner(x) == cell1]
indices_cell2 = [i for i, x in enumerate(X_scaled) if som.winner(x) == cell2]

# Сравнение средних значений показателей для двух ячеек
mean_cell1 = X.iloc[indices_cell1].mean()
mean_cell2 = X.iloc[indices_cell2].mean()

print("Средние значения для ячейки 1 ({}):\n{}".format(cell1, mean_cell1))
print("\nСредние значения для ячейки 2 ({}):\n{}".format(cell2, mean_cell2))


# # Задача 3

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Применение PCA для уменшения размерности
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Применение t-SNE для уменшения размерности 
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

unique_classes = y.unique()
colors = plt.cm.get_cmap('Paired', len(unique_classes))  # Используем цветовую палитру

# Визулизация распределение PCA
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i, class_label in enumerate(unique_classes):
    plt.scatter(X_pca[y == class_label, 0], X_pca[y == class_label, 1], 
                label=class_label, color=colors(i))
plt.title('PCA: Распределение объектов')
plt.xlabel('Компонента 1')
plt.ylabel('Компонента 2')
plt.legend()
plt.grid()

# Визулизация распределение t-SNE
plt.subplot(1, 2, 2)
for i, class_label in enumerate(unique_classes):
    plt.scatter(X_tsne[y == class_label, 0], X_tsne[y == class_label, 1], 
                label=class_label, color=colors(i))
plt.title('t-SNE: Распределение объектов')
plt.xlabel('Компонента 1')
plt.ylabel('Компонента 2')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ## Задача 4

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Создадим функцию для классификации и оценки с помощью модели RandomForest
def classify_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))

# посчитаем точность модели для 2 компонент в PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f'\nPCA с {2} компонентами:')
classify_and_evaluate(X_train_pca, X_test_pca, y_train, y_test)


# In[ ]:


# посчитаем точность модели для 3 компонент в PCA
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f'\nPCA с {3} компонентами:')
classify_and_evaluate(X_train_pca, X_test_pca, y_train, y_test)


# In[ ]:


# посчитаем точность модели для 2 компонент в t-SNE
tsne = TSNE(n_components=2)
X_train_tsne = tsne.fit_transform(X_train)
X_test_tsne = tsne.fit_transform(X_test) 
print('\nt-SNE:')
classify_and_evaluate(X_train_tsne, X_test_tsne, y_train, y_test)


# In[26]:


# посчитаем точность модели для 3 компонент в t-SNE
tsne = TSNE(n_components=3)
X_train_tsne = tsne.fit_transform(X_train)
X_test_tsne = tsne.fit_transform(X_test) 
print('\nt-SNE:')
classify_and_evaluate(X_train_tsne, X_test_tsne, y_train, y_test)


# Как видно из получиных результатов при увеличении компонент для методов PCA и t-NSE с 2 до 3, точность модели увеличивается на 20% и 10% соответственно. Также можно замеить, что точность модели на упрощенных данных с помощью PCA выше точносит модели t-NSE на любом значении компонент.

# ## Задача 5

# In[46]:


from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Кластеризация на исходных данных с помощью KMeans
kmeans_original = KMeans(n_clusters=6)# ищем 6 кластера (кол-во найдено по методу Логтя из лаб. работы с этими данным)
y_kmeans_original = kmeans_original.fit_predict(X)

# Оценка согласованности кластерных решений Rand для исходных данных
rand_index_original = adjusted_rand_score(y, y_kmeans_original)
print(f'Rand Index (исходные данные): {rand_index_original:.2f}')

#  Упрощение PCA и кластеризация упрощенных данных
for n_components in [2, 3]: 
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Кластеризация на данных после упрощения PCA
    kmeans_pca = KMeans(n_clusters=6)
    y_kmeans_pca = kmeans_pca.fit_predict(X_pca)

    # Оценка согласованности результатов
    rand_index_pca = adjusted_rand_score(y, y_kmeans_pca)
    print(f'Rand Index (PCA с {n_components} компонентами): {rand_index_pca:.2f}')


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Кластеризация на исходных данных с помощью KMeans
kmeans_original = KMeans(n_clusters=6)# ищем 6 кластера (кол-во найдено по методу Логтя из лаб. работы с этими данным)
y_kmeans_original = kmeans_original.fit_predict(X)

# Оценка согласованности кластерных решений Rand для исходных данных
rand_index_original = adjusted_rand_score(y, y_kmeans_original)
print(f'Rand Index (исходные данные): {rand_index_original:.2f}')

#  Упрощение PCA и кластеризация упрощенных данных
for n_components in [2, 3]: 
    pca = PCA(n_components=n_components, svd_solver='randomized' ) # изменим метод вычисления
    X_pca = pca.fit_transform(X)

    # Кластеризация на данных после упрощения PCA
    kmeans_pca = KMeans(n_clusters=6)
    y_kmeans_pca = kmeans_pca.fit_predict(X_pca)

    # Оценка согласованности результатов
    rand_index_pca = adjusted_rand_score(y, y_kmeans_pca)
    print(f'Rand Index (PCA с {n_components} компонентами): {rand_index_pca:.2f}')


# In[49]:


from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Кластеризация на исходных данных с помощью KMeans
kmeans_original = KMeans(n_clusters=6)# ищем 6 кластера (кол-во найдено по методу Логтя из лаб. работы с этими данным)
y_kmeans_original = kmeans_original.fit_predict(X)

# Оценка согласованности кластерных решений Rand для исходных данных
rand_index_original = adjusted_rand_score(y, y_kmeans_original)
print(f'Rand Index (исходные данные): {rand_index_original:.2f}')

#  Упрощение PCA и кластеризация упрощенных данных
for n_components in [2, 3]: 
    pca = PCA(n_components=3, svd_solver='full' ) # изменим метод вычисления
    X_pca = pca.fit_transform(X)

    # Кластеризация на данных после упрощения PCA
    kmeans_pca = KMeans(n_clusters=6)
    y_kmeans_pca = kmeans_pca.fit_predict(X_pca)

    # Оценка согласованности результатов
    rand_index_pca = adjusted_rand_score(y, y_kmeans_pca)
    print(f'Rand Index (PCA с {n_components} компонентами): {rand_index_pca:.2f}')


# In[55]:


from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Кластеризация на исходных данных с помощью KMeans
kmeans_original = KMeans(n_clusters=6)# ищем 6 кластера (кол-во найдено по методу Логтя из лаб. работы с этими данным)
y_kmeans_original = kmeans_original.fit_predict(X)

# Оценка согласованности кластерных решений Rand для исходных данных
rand_index_original = adjusted_rand_score(y, y_kmeans_original)
print(f'Rand Index (исходные данные): {rand_index_original:.2f}')

#  Упрощение PCA и кластеризация упрощенных данных
for n_components in [2, 3]: 
    pca = PCA(n_components=0.95 ) # изменим кол-во компонент на процент объясненной дисперсии в данных
    X_pca = pca.fit_transform(X)

    # Кластеризация на данных после упрощения PCA
    kmeans_pca = KMeans(n_clusters=6)
    y_kmeans_pca = kmeans_pca.fit_predict(X_pca)

    # Оценка согласованности результатов
    rand_index_pca = adjusted_rand_score(y, y_kmeans_pca)
    print(f'Rand Index (PCA с {n_components} компонентами): {rand_index_pca:.2f}')

