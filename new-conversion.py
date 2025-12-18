#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install yellowbrick')


# In[2]:


import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


# # Лабораторная работа 5. Кластеризация данных.

# ## Исскуственный набор данных make_moons

# In[3]:


from sklearn.datasets import make_moons, make_circles, make_blobs

# Создание набора moons
X_moons, y_moons = make_moons(n_samples=100, noise=0.1, random_state=42)

plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis')
plt.title("Make_moons")

plt.show()


# ## Алгоритмом кластеризации kmenas

# In[4]:


kmeans = KMeans(n_clusters=2, init='random', n_init=10, max_iter=300, random_state=3)
y_kmeans = kmeans.fit_predict(X_moons)

cluster_indices, counts = np.unique(y_kmeans, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xlabel("Кластеры")
plt.ylabel("Количество объектов")
plt.show()


# In[5]:


plt.scatter(X_moons[y_kmeans == 0, 0], X_moons[y_kmeans == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black',
            label='Кластер 1')
plt.scatter(X_moons[y_kmeans == 1, 0], X_moons[y_kmeans == 1, 1], s=50, c='orange', marker='o', edgecolor='black',
            label='Кластер 2')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 250, marker='*', c = 'red', edgecolor='black',
            label = 'Центроиды')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()


# In[6]:


# Введем другие параметры инициализации центра кластера
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, random_state=3)
y_kmeans = kmeans.fit_predict(X_moons)

cluster_indices, counts = np.unique(y_kmeans, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xlabel("Кластеры")
plt.ylabel("Количество объектов")
plt.show()


plt.scatter(X_moons[y_kmeans == 0, 0], X_moons[y_kmeans == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black',
            label='Кластер 1')
plt.scatter(X_moons[y_kmeans == 1, 0], X_moons[y_kmeans == 1, 1], s=50, c='orange', marker='o', edgecolor='black',
            label='Кластер 2')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 250, marker='*', c = 'red', edgecolor='black',
            label = 'Центроиды')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()


# In[7]:


# Введем другие параметры алгоритма кластеризации, изменим количество раз, которое алгоритм будет запускаться с разными начальными центрами кластеров
kmeans = KMeans(n_clusters=2, init='random', n_init=30, max_iter=300, random_state=3)
y_kmeans = kmeans.fit_predict(X_moons)

cluster_indices, counts = np.unique(y_kmeans, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xlabel("Кластеры")
plt.ylabel("Количество объектов")
plt.show()


plt.scatter(X_moons[y_kmeans == 0, 0], X_moons[y_kmeans == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black',
            label='Кластер 1')
plt.scatter(X_moons[y_kmeans == 1, 0], X_moons[y_kmeans == 1, 1], s=50, c='orange', marker='o', edgecolor='black',
            label='Кластер 2')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 250, marker='*', c = 'red', edgecolor='black',
            label = 'Центроиды')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()


# In[8]:


# Введем другие параметры алгоритма кластеризации, изменим количество раз, которое алгоритм будет запускаться с разными начальными центрами кластеров
kmeans = KMeans(n_clusters=2, init='random', n_init=50, max_iter=300, random_state=3)
y_kmeans = kmeans.fit_predict(X_moons)

cluster_indices, counts = np.unique(y_kmeans, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xlabel("Кластеры")
plt.ylabel("Количество объектов")
plt.show()


plt.scatter(X_moons[y_kmeans == 0, 0], X_moons[y_kmeans == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black',
            label='Кластер 1')
plt.scatter(X_moons[y_kmeans == 1, 0], X_moons[y_kmeans == 1, 1], s=50, c='orange', marker='o', edgecolor='black',
            label='Кластер 2')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 250, marker='*', c = 'red', edgecolor='black',
            label = 'Центроиды')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()


# ### Алгоритм кластеризации agglomerative clustering

# In[9]:


# Применение агломеративной кластеризации
ac = AgglomerativeClustering(n_clusters=2, linkage='single')
y_agg = ac.fit_predict(X_moons)

cluster_indices, counts = np.unique(y_agg, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]
plt.title("Распределение объектов по кластерам AgglomerativeClustering")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xlabel("Кластеры")
plt.ylabel("Количество объектов")
plt.show()

plt.scatter(X_moons[y_agg == 0, 0], X_moons[y_agg == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_moons[y_agg == 1, 0], X_moons[y_agg == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')
plt.title("Агломеративная кластеризация single")
plt.legend()
plt.tight_layout()
plt.show()


# In[10]:


# Применение агломеративной кластеризации
ac = AgglomerativeClustering(n_clusters=2, linkage='complete')
y_agg = ac.fit_predict(X_moons)

cluster_indices, counts = np.unique(y_agg, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]
plt.title("Распределение объектов по кластерам AgglomerativeClustering")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xlabel("Кластеры")
plt.ylabel("Количество объектов")
plt.show()

plt.scatter(X_moons[y_agg == 0, 0], X_moons[y_agg == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_moons[y_agg == 1, 0], X_moons[y_agg == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')
plt.title("Агломеративная кластеризация complete")
plt.legend()
plt.tight_layout()
plt.show()


# In[11]:


# Применение агломеративной кластеризации
ac = AgglomerativeClustering(n_clusters=2, linkage='ward')
y_agg = ac.fit_predict(X_moons)

cluster_indices, counts = np.unique(y_agg, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам AgglomerativeClustering")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xlabel("Кластеры")
plt.ylabel("Количество объектов")
plt.show()

plt.scatter(X_moons[y_agg == 0, 0], X_moons[y_agg == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_moons[y_agg == 1, 0], X_moons[y_agg == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')


plt.title("Агломеративная кластеризация ward")
plt.legend()
plt.tight_layout()
plt.show()


# In[12]:


from sklearn.cluster import DBSCAN

# Применение алгоритма DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=2)
y_dbscan = dbscan.fit_predict(X_moons)

cluster_indices, counts = np.unique(y_dbscan, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам DBSCAN")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xlabel("Кластеры")
plt.ylabel("Количество объектов")
plt.show()


plt.scatter(X_moons[y_dbscan == 0, 0], X_moons[y_dbscan == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_moons[y_dbscan == 1, 0], X_moons[y_dbscan == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')

# Обработка шума
if len(set(y_dbscan)) > 2:  # Если есть больше двух кластеров
    for i in range(2, len(set(y_dbscan))):
        plt.scatter(X_moons[y_dbscan == i, 0], X_moons[y_dbscan == i, 1],
                    edgecolor='black', marker='x', s=40, label=f'Кластер {i + 1}')

plt.title("Кластеризация DBSCAN")
plt.legend()
plt.tight_layout()
plt.show()


# In[13]:


# Применение алгоритма DBSCAN с другими параметрами
dbscan = DBSCAN(eps=0.25, min_samples=5)
y_dbscan = dbscan.fit_predict(X_moons)

cluster_indices, counts = np.unique(y_dbscan, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам DBSCAN")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xlabel("Кластеры")
plt.ylabel("Количество объектов")
plt.show()

plt.scatter(X_moons[y_dbscan == 0, 0], X_moons[y_dbscan == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_moons[y_dbscan == 1, 0], X_moons[y_dbscan == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')


# Обработка шума 
if len(set(y_dbscan)) > 2: 
    for i in range(2, len(set(y_dbscan))):
        plt.scatter(X_moons[y_dbscan == i, 0], X_moons[y_dbscan == i, 1],
                    edgecolor='black', marker='x', s=40, label=f'Кластер {i + 1}')

plt.title("Кластеризация с помощью DBSCAN")
plt.legend()
plt.tight_layout()
plt.show()


# In[14]:


# Применение алгоритма DBSCAN с другими параметрами
dbscan = DBSCAN(eps=0.25, min_samples=2)
y_dbscan = dbscan.fit_predict(X_moons)

cluster_indices, counts = np.unique(y_dbscan, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам DBSCAN")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xlabel("Кластеры")
plt.ylabel("Количество объектов")
plt.show()


plt.scatter(X_moons[y_dbscan == 0, 0], X_moons[y_dbscan == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_moons[y_dbscan == 1, 0], X_moons[y_dbscan == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')

# Обработка шума
if len(set(y_dbscan)) > 2: 
    for i in range(2, len(set(y_dbscan))):
        plt.scatter(X_moons[y_dbscan == i, 0], X_moons[y_dbscan == i, 1],
                    edgecolor='black', marker='x', s=40, label=f'Кластер {i + 1}')

plt.title("Кластеризация с помощью DBSCAN")
plt.legend()
plt.tight_layout()
plt.show()


# ## Исскуственный набор данных make_circles

# In[15]:


# Создание набора данных
X_circles, y_circles = make_circles(n_samples=100, noise=0.1, random_state=4)

plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis')
plt.title("Make_circles")
plt.show()


# ## Алгоритмом кластеризации kmenas

# In[16]:


# Применение K-means
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, random_state=3)
y_kmeans = kmeans.fit_predict(X_circles)

# Подсчет объектов в каждом кластере
cluster_indices, counts = np.unique(y_kmeans, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xlabel("Кластеры")
plt.ylabel("Количество объектов")
plt.show()

plt.scatter(X_circles[y_kmeans == 0, 0], X_circles[y_kmeans == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_circles[y_kmeans == 1, 0], X_circles[y_kmeans == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=250, marker='*', c='red', edgecolor='black', label='Центроиды')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.title("Кластеризация K-means")
plt.show()


# In[17]:


# Применение K-means c с другими параметрами
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=20, max_iter=300, random_state=3)
y_kmeans = kmeans.fit_predict(X_circles)

# Подсчет объектов в каждом кластере
cluster_indices, counts = np.unique(y_kmeans, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.ylabel("Количество объектов")
plt.show()

plt.scatter(X_circles[y_kmeans == 0, 0], X_circles[y_kmeans == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_circles[y_kmeans == 1, 0], X_circles[y_kmeans == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')
plt.scatter(X_circles[y_kmeans == 2, 0], X_circles[y_kmeans == 2, 1], s=50, c='skyblue', marker='o', edgecolor='black', label='Кластер 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=250, marker='*', c='red', edgecolor='black', label='Центроиды')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.title("Кластеризация K-means")
plt.show()


# ## Алгоритмом кластеризации AgglomerativeClustering

# In[18]:


# Применение агломеративной кластеризации
ac = AgglomerativeClustering(n_clusters=2, linkage='single')
y_agg = ac.fit_predict(X_circles)

cluster_indices, counts = np.unique(y_agg, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам AgglomerativeClustering")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xlabel("Кластеры")
plt.ylabel("Количество объектов")
plt.show()

plt.scatter(X_circles[y_agg == 0, 0], X_circles[y_agg == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_circles[y_agg == 1, 0], X_circles[y_agg == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')
plt.title("Агломеративная кластеризация single")
plt.legend()
plt.tight_layout()
plt.show()


# In[19]:


# Применение агломеративной кластеризации
ac = AgglomerativeClustering(n_clusters=2, linkage='complete')
y_agg = ac.fit_predict(X_circles)

cluster_indices, counts = np.unique(y_agg, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам AgglomerativeClustering")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xlabel("Кластеры")
plt.ylabel("Количество объектов")
plt.show()

plt.scatter(X_circles[y_agg == 0, 0], X_circles[y_agg == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_circles[y_agg == 1, 0], X_circles[y_agg == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')
plt.title("Агломеративная кластеризация complete")
plt.legend()
plt.tight_layout()
plt.show()


# In[20]:


# Применение агломеративной кластеризации
ac = AgglomerativeClustering(n_clusters=3, linkage='complete')
y_agg = ac.fit_predict(X_circles)

cluster_indices, counts = np.unique(y_agg, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам AgglomerativeClustering")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.ylabel("Количество объектов")
plt.show()

plt.scatter(X_circles[y_agg == 0, 0], X_circles[y_agg == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_circles[y_agg == 1, 0], X_circles[y_agg == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')
plt.scatter(X_circles[y_agg == 2, 0], X_circles[y_agg == 2, 1], s=50, c='skyblue', marker='o', edgecolor='black', label='Кластер 3')
plt.title("Агломеративная кластеризация complete")
plt.legend()
plt.tight_layout()
plt.show()


# In[21]:


# Применение агломеративной кластеризации
ac = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_agg = ac.fit_predict(X_circles)

cluster_indices, counts = np.unique(y_agg, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам AgglomerativeClustering")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.ylabel("Количество объектов")
plt.show()

plt.scatter(X_circles[y_agg == 0, 0], X_circles[y_agg == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_circles[y_agg == 1, 0], X_circles[y_agg == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')
plt.scatter(X_circles[y_agg == 2, 0], X_circles[y_agg == 2, 1], s=50, c='skyblue', marker='o', edgecolor='black', label='Кластер 3')
plt.title("Агломеративная кластеризация ward")
plt.legend()
plt.tight_layout()
plt.show()


# ## Алгоритмом кластеризации DBSCAN

# In[22]:


# Применение алгоритма DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=2)
y_dbscan = dbscan.fit_predict(X_circles)

cluster_indices, counts = np.unique(y_dbscan, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам DBSCAN")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xticks(rotation=45) 
plt.ylabel("Количество объектов")
plt.show()


plt.scatter(X_circles[y_dbscan == 0, 0], X_circles[y_dbscan == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_circles[y_dbscan == 1, 0], X_circles[y_dbscan == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')

if len(set(y_dbscan)) > 2:
    for i in range(2, len(set(y_dbscan))):
        plt.scatter(X_circles[y_dbscan == i, 0], X_circles[y_dbscan == i, 1],
                    edgecolor='black', marker='x', s=40, label=f'Кластер {i + 1}')

plt.title("Кластеризация DBSCAN")
plt.legend()
plt.tight_layout()
plt.show()


# In[23]:


# Применение алгоритма DBSCAN с другими параметрами
dbscan = DBSCAN(eps=0.23, min_samples=5)
y_dbscan = dbscan.fit_predict(X_circles)

cluster_indices, counts = np.unique(y_dbscan, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам DBSCAN")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xticks(rotation=45) 
plt.ylabel("Количество объектов")
plt.show()


plt.scatter(X_circles[y_dbscan == 0, 0], X_circles[y_dbscan == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_circles[y_dbscan == 1, 0], X_circles[y_dbscan == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')

if len(set(y_dbscan)) > 2:
    for i in range(2, len(set(y_dbscan))):
        plt.scatter(X_circles[y_dbscan == i, 0], X_circles[y_dbscan == i, 1],
                    edgecolor='black', marker='x', s=40, label=f'Кластер {i + 1}')

plt.title("Кластеризация DBSCAN")
plt.legend()
plt.tight_layout()
plt.show()


# ## Исскуственный набор данных make_blobs

# In[24]:


# Создание набора данных
X_blobs, y_blobs = make_blobs(n_samples=100, centers=3, random_state=4)

plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap='viridis')
plt.title("Make_blobs")
plt.show()


# ## Алгоритмом кластеризации kmenas

# In[25]:


# Применение K-means
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=20, max_iter=300, random_state=3)
y_kmeans = kmeans.fit_predict(X_blobs)

# Подсчет объектов в каждом кластере
cluster_indices, counts = np.unique(y_kmeans, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.ylabel("Количество объектов")
plt.show()

plt.scatter(X_blobs[y_kmeans == 0, 0], X_blobs[y_kmeans == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_blobs[y_kmeans == 1, 0], X_blobs[y_kmeans == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')
plt.scatter(X_blobs[y_kmeans == 2, 0], X_blobs[y_kmeans == 2, 1], s=50, c='skyblue', marker='o', edgecolor='black', label='Кластер 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=250, marker='*', c='red', edgecolor='black', label='Центроиды')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.title("Кластеризация K-means")
plt.show()


# In[26]:


# Применение K-means c с другими параметрами
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=200, random_state=3)
y_kmeans = kmeans.fit_predict(X_blobs)

# Подсчет объектов в каждом кластере
cluster_indices, counts = np.unique(y_kmeans, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.ylabel("Количество объектов")
plt.show()

plt.scatter(X_blobs[y_kmeans == 0, 0], X_blobs[y_kmeans == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_blobs[y_kmeans == 1, 0], X_blobs[y_kmeans == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')
plt.scatter(X_blobs[y_kmeans == 2, 0], X_blobs[y_kmeans == 2, 1], s=50, c='skyblue', marker='o', edgecolor='black', label='Кластер 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=250, marker='*', c='red', edgecolor='black', label='Центроиды')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.title("Кластеризация K-means")
plt.show()


# ## Алгоритмом кластеризации AgglomerativeClustering

# In[27]:


# Применение агломеративной кластеризации
ac = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_agg = ac.fit_predict(X_blobs)

cluster_indices, counts = np.unique(y_agg, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам AgglomerativeClustering")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.ylabel("Количество объектов")
plt.show()

plt.scatter(X_blobs[y_agg == 0, 0], X_blobs[y_agg == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_blobs[y_agg == 1, 0], X_blobs[y_agg == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')
plt.scatter(X_blobs[y_agg == 2, 0], X_blobs[y_agg == 2, 1], s=50, c='skyblue', marker='o', edgecolor='black', label='Кластер 3')
plt.title("Агломеративная кластеризация ward")
plt.legend()
plt.tight_layout()
plt.show()


# In[28]:


# Применение агломеративной кластеризации c другими параметрами
ac = AgglomerativeClustering(n_clusters=3, linkage='complete')
y_agg = ac.fit_predict(X_blobs)

cluster_indices, counts = np.unique(y_agg, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам AgglomerativeClustering")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.ylabel("Количество объектов")
plt.show()

plt.scatter(X_blobs[y_agg == 0, 0], X_blobs[y_agg == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_blobs[y_agg == 1, 0], X_blobs[y_agg == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')
plt.scatter(X_blobs[y_agg == 2, 0], X_blobs[y_agg == 2, 1], s=50, c='skyblue', marker='o', edgecolor='black', label='Кластер 3')
plt.title("Агломеративная кластеризация complete")
plt.legend()
plt.tight_layout()
plt.show()


# ## Алгоритмом кластеризации DBSCAN

# In[29]:


# Применение алгоритма DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=3)
y_dbscan = dbscan.fit_predict(X_blobs)

cluster_indices, counts = np.unique(y_dbscan, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам DBSCAN")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xticks(rotation=45) 
plt.ylabel("Количество объектов")
plt.show()


plt.scatter(X_blobs[y_dbscan == 0, 0], X_blobs[y_dbscan == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Кластер 1')
plt.scatter(X_blobs[y_dbscan == 1, 0], X_blobs[y_dbscan == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')

if len(set(y_dbscan)) > 2:
    for i in range(2, len(set(y_dbscan))):
        plt.scatter(X_blobs[y_dbscan == i, 0], X_blobs[y_dbscan == i, 1],
                    edgecolor='black', marker='x', s=40, label=f'Кластер {i + 1}')

plt.title("Кластеризация DBSCAN")
plt.legend()
plt.tight_layout()
plt.show()


# In[30]:


# Применение алгоритма DBSCAN c другими параметрами
dbscan = DBSCAN(eps=0.3, min_samples=4)
y_dbscan = dbscan.fit_predict(X_blobs)

cluster_indices, counts = np.unique(y_dbscan, return_counts=True)
cluster_names = ["Кластер {0}".format(idx + 1) for idx in cluster_indices]

plt.title("Распределение объектов по кластерам DBSCAN")
plt.bar(x=cluster_names, height=counts, color='skyblue')
plt.xticks(rotation=45) 
plt.ylabel("Количество объектов")
plt.show()


plt.scatter(X_blobs[y_dbscan == 0, 0], X_blobs[y_dbscan == 0, 1], s=50, c='lightgreen', marker='o', edgecolor='black', label='Кластер 1')
plt.scatter(X_blobs[y_dbscan == 1, 0], X_blobs[y_dbscan == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Кластер 2')

if len(set(y_dbscan)) > 2:
    for i in range(2, len(set(y_dbscan))):
        plt.scatter(X_blobs[y_dbscan == i, 0], X_blobs[y_dbscan == i, 1],
                    edgecolor='black', marker='o', s=40, label=f'Кластер {i + 1}')

plt.title("Кластеризация DBSCAN")
plt.legend()
plt.tight_layout()
plt.show()


# Как видно на исскуственных данных, ближе всего к делению на исходные кластеры подходят данные алгоритмов KMeans и AgglomerativeClustering. Точки для кластеров выбираются приблизительно так же, как в сгенирированных данных, особенно это видно для данных moons и bubles.
# DBscan же, на искуственных данных, делит на слишком большое кол-во кластеров. Чтобы достичь правильное деление, требуется очень крапотливый подбор праметров кластеризации. Как видно в следующем пункте с многомерными данными, DBScan получил самый высокий коэффициент силуэта из 3 алгоритмов. И при этом единственный выделяет шум. Можно сделать вывод, что этот алгоритм хорошо подходит для реальных данных, с высоким уровнем полотности кластеров. Так же он подойдет, когда мы ищем кластеры исходя из конкретнных параметров (максимальное расстояние между точками и минимальное кол-во точек в окретсности)

# ## Набор многомерных данных

# In[31]:


data = pd.read_excel("Dry_Bean_Dataset.xlsx").dropna()
data


# In[32]:


# Удаление признака класса
X = data.drop(columns=['Class'])

# Стандартизация данных
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[55]:


#Метод логтя, для определения оптимального кол-ва кластеров

from sklearn.cluster import KMeans

distortions = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)


plt.plot(range(1, 11), distortions, marker = 'o')
plt.xlabel('Количество кластеров')
plt.ylabel('Искажение')
plt.tight_layout()
plt.show()


# In[46]:


# Применение K-Means
kmeans = KMeans(n_clusters=4, random_state=4)  # Выберите количество кластеров
kmeans.fit(X_scaled)

# Получение меток кластеров
data['Cluster'] = kmeans.labels_

# Вывод результатов
print(data)


# In[47]:


# Оценка качества кластеров с помощью коэффициента силуэта
silhouette_avg = silhouette_score(X_scaled, data['Cluster'])
print(f'Средний коэффициент силуэта: {silhouette_avg:.2f}')

# Оценка распределения объектов по классам
class_distribution = data.groupby(['Cluster', 'Class']).size().unstack(fill_value=0)

# Вывод распределения в табличной форме
print("\nРаспределение объектов по классам:")
print(class_distribution)

# Визуализация распределения объектов по классам
class_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Распределение объектов по классам в кластерах')
plt.xlabel('Кластеры')
plt.ylabel('Количество объектов')
plt.legend(title='Класс')
plt.grid(axis='y')
plt.show()


# In[51]:


from sklearn.metrics import silhouette_samples

# Обучение модели KMeans
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0)
labels = kmeans.fit_predict(X)

# Оценка качества кластеров с помощью коэффициента силуэта
silhouette_avg = silhouette_score(X_scaled, data['Cluster'])
print(f'Средний коэффициент силуэта: {silhouette_avg:.2f}')

# Оценка распределения объектов по классам
class_distribution = data.groupby(['Cluster', 'Class']).size().unstack(fill_value=0)

# Вывод распределения в табличной форме
print("\nРаспределение объектов по классам:")
print(class_distribution)

# Визуализация распределения объектов по классам
class_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Распределение объектов по классам в кластерах')
plt.xlabel('Кластеры')
plt.ylabel('Количество объектов')
plt.legend(title='Класс')
plt.grid(axis='y')
plt.show()

# Оценка коэффициента силуэта
silhouette_values = silhouette_samples(X, labels, metric='euclidean')

# Визуализация значений коэффициента силуэта
y_ax_lower, y_ax_upper = 0, 0
yticks = []
cluster_labels = np.unique(labels)
n_clusters = cluster_labels.shape[0]
colors = ['orange', 'skyblue', 'green', 'red']

for c in cluster_labels:
    c_silhouette_values = silhouette_values[labels == c]
    c_silhouette_values.sort()
    y_ax_upper += len(c_silhouette_values)
    color = colors[c]
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_values,
             height=1.0,
             edgecolor='none', color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_values)

silhouette_avg = np.mean(silhouette_values)
plt.axvline(silhouette_avg, color='black', linestyle='--')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Кластер')
plt.xlabel('Коэффициент силуэта')
plt.title('Коэффициенты силуэта для каждого объекта')
plt.tight_layout()
plt.show()


# In[64]:


# Масштабирование данных
X_scaled = StandardScaler().fit_transform(X)

# Обучение модели DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Оценка качества с помощью коэффициента силуэта
# Не учитываем шум
if len(set(labels)) > 1 and -1 in labels:
    silhouette_avg = silhouette_score(X_scaled[labels != -1], labels[labels != -1])
    print(f'Средний коэффициент силуэта: {silhouette_avg:.2f}')
else:
    print('Недостаточно кластеров для оценки коэффициента силуэта.')

# Оценка распределения объектов по классам
class_distribution = data.groupby(['Cluster', 'Class']).size().unstack(fill_value=0)

# Вывод распределения в табличной форме
print("\nРаспределение объектов по классам:")
print(class_distribution)

# Визуализация распределения объектов по классам
class_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Распределение объектов по классам в кластерах DBSCAN')
plt.xlabel('Кластеры')
plt.ylabel('Количество объектов')
plt.legend(title='Класс')
plt.grid(axis='y')
plt.show()

# Оценка коэффициента силуэта для объектов в кластерах
if len(set(labels)) > 1 and -1 in labels:
    silhouette_values = silhouette_samples(X_scaled[labels != -1], labels[labels != -1])

    # Визуализация значений коэффициента силуэта
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    cluster_labels = np.unique(labels[labels != -1])
    n_clusters = cluster_labels.shape[0]
    colors = ['orange', 'skyblue', 'green', 'red', 'purple']

    for c in cluster_labels:
        c_silhouette_values = silhouette_values[labels[labels != -1] == c]
        c_silhouette_values.sort()
        y_ax_upper += len(c_silhouette_values)
        color = colors[c]
        plt.barh(range(y_ax_lower, y_ax_upper),
                 c_silhouette_values,
                 height=1.0,
                 edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_values)

    silhouette_avg = np.mean(silhouette_values)
    plt.axvline(silhouette_avg, color='black', linestyle='--')
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Кластер')
    plt.xlabel('Коэффициент силуэта')
    plt.title('Коэффициенты силуэта для каждого объекта DBSCAN')
    plt.tight_layout()
    plt.show()
else:
    print('Недостаточно кластеров для визуализации коэффициента силуэта.')


# In[92]:


agg_clustering = AgglomerativeClustering(n_clusters=4)
labels = agg_clustering.fit_predict(X_scaled)


# Оценка качества кластеров с помощью коэффициента силуэта
silhouette_avg = silhouette_score(X_scaled, labels)
print(f'Средний коэффициент силуэта: {silhouette_avg:.2f}')

# Оценка распределения объектов по классам
class_distribution = data.groupby(['Cluster', 'Class']).size().unstack(fill_value=0)

print("\nРаспределение объектов по классам:")
print(class_distribution)


class_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Распределение объектов по классам в кластерах Agglomerative Clustering')
plt.xlabel('Кластеры')
plt.ylabel('Количество объектов')
plt.legend(title='Класс')
plt.grid(axis='y')
plt.show()

# Оценка коэффициента силуэта для объектов в кластерах
silhouette_values = silhouette_samples(X_scaled, labels)

# Визуализация значений коэффициента силуэта
y_ax_lower, y_ax_upper = 0, 0
yticks = []
cluster_labels = np.unique(labels)
n_clusters = cluster_labels.shape[0]
colors = ['orange', 'skyblue', 'green', 'red']

for c in cluster_labels:
    c_silhouette_values = silhouette_values[labels == c]
    c_silhouette_values.sort()
    y_ax_upper += len(c_silhouette_values)
    color = colors[c]
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_values,
             height=1.0,
             edgecolor='none', color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_values)

silhouette_avg = np.mean(silhouette_values)
plt.axvline(silhouette_avg, color='black', linestyle='--')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Кластер')
plt.xlabel('Коэффициент силуэта')
plt.title('Коэффициенты силуэта для каждого объекта Agglomerative Clustering')
plt.tight_layout()
plt.show()

