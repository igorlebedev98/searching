#!/usr/bin/env python
# coding: utf-8

# # Практическая работа №2

# ## О наборе данных
# 
# Это набор изображений 50 типов деталей автомобиля. Он включает тренировочный набор, тестовый набор и набор проверки. Существует 50 классов деталей автомобиля. Тренировочный набор не сбалансирован. Класс Ignition Coil имеет наибольшее количество обучающих изображений = 200. Класс Leaf Spring имеет наименьшее количество изображений = 110. Наборы проверки и тестирования имеют по 5 изображений для каждого из 50 классов. Изображения имеют размеры 224 X 224 X3 в формате jpg. Все изображения являются оригинальными, в наборе данных нет дополненных изображений.

# ## Построение классификаторов на исходных данных. Экспериментальный анализ и подбор гиперпараметров.

# In[1]:


import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import models, layers



#Путь до файлов с исходнными данными
train_dir = r'D:\py\car parts\train\IGNITION COIL'
test_dir = r'D:\py\car parts\test\IGNITION COIL'
valid_dir = r'D:\py\car parts\valid\IGNITION COIL'

# Создаем генератор для загрузки изображений
train_datagen = ImageDataGenerator(rescale=1./255) # Нормализуем данные
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.dirname(train_dir),
    target_size=(224, 224),
    batch_size=16, 
    class_mode='categorical', # Выбираем категориальная классификация
    shuffle=False  # Не перемешиваем данные
)

test_generator = test_datagen.flow_from_directory(
    os.path.dirname(test_dir),
    target_size=(224, 224),
    batch_size=16, 
    class_mode='categorical', # Выбираем категориальная классификация
    shuffle=False  # Не перемешиваем данные
)


# In[ ]:


# Модель Random Forest
X_train_rf = []
y_train_rf = []

# Получение данных из генератора
for _ in range(len(train_generator)):
    images, labels = next(train_generator) 
    X_train_rf.append(images)
    y_train_rf.append(labels)

# Объединение массивов 
X_train_rf = np.vstack(X_train_rf) 
y_train_rf = np.vstack(y_train_rf) 

# Преобразование в двухмерный массив для Random Forest
X_train_rf = X_train_rf.reshape(X_train_rf.shape[0], -1)

# Обучение модели
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_rf, np.argmax(y_train_rf, axis=1)) 

# Оценка модели
X_test_rf = []
y_test_rf = []

# Получение данных из тестового генератора
test_generator.reset()  # Сброс генератора
for _ in range(len(test_generator)):
    images, labels = next(test_generator)  # next() для получения данных
    X_test_rf.append(images)
    y_test_rf.append(labels)

 # Объединение массивов
X_test_rf = np.vstack(X_test_rf)
y_test_rf = np.vstack(y_test_rf)

# Преобразование в двухмерный массив для Random Forest
X_test_rf = X_test_rf.reshape(X_test_rf.shape[0], -1)

# Предсказаные данные
y_pred_rf = rf_model.predict(X_test_rf)

# Вывод справки о классификации
print("Random Forest Classification Report:")
print(classification_report(np.argmax(y_test_rf, axis=1), y_pred_rf))
accuracy = np.sum(y_pred_rf == np.argmax(y_test_rf, axis=1)) / len(y_test_rf) * 100
print(f"Процент правильных предсказаний: {accuracy:.2f}%")


# In[ ]:


import matplotlib.pyplot as plt
incorrect_indices = np.where(y_pred_rf != np.argmax(y_test_rf, axis=1))[0]

#Вывод ошибочно предсказанных изображений
num_incorrect = len(incorrect_indices)
print(f"Количество ошибочно предсказанных изображений: {num_incorrect}")

original_height = 224
original_width = 224
num_channels = 3

plt.figure(figsize=(15, 5))

num_to_display = min(num_incorrect, 10)  #максимум 10 изображений

for i in range(num_to_display):
    index = incorrect_indices[i]
    plt.subplot(2, num_to_display // 2, i + 1) 
    plt.imshow(X_test_rf[index].reshape(original_height, original_width, num_channels)) 
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[17]:


# Создание модели многослойного перцептрон


# Создаем генератор для валидационных данных
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(
    valid_dir,  
    target_size=(224, 224),
    batch_size=16, 
    class_mode='categorical',  
    shuffle=False 
)

test_generator.reset()
train_generator.reset()
from tensorflow.keras import layers, models
mlp_model = models.Sequential()
mlp_model.add(layers.Flatten(input_shape=(224, 224, 3))) # Преобразование в одномерный массив
mlp_model.add(layers.Dense(128, activation='relu')) # Полносвязный слой с 128 нейронами и relu активацией
mlp_model.add(layers.Dropout(0.3))
# Выходной слой с нейронмами (сколько классов столько нейронов) и активацией softmax
mlp_model.add(layers.Dense(len(train_generator.class_indices), activation='softmax')) 


# Компиляция модели с оптимизатором adam, функция потерь
# которая используется для многоклассовой классификации и выбранной метрикой оценки модели - точность (доля правильных предсказаний модели)
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели на тренировочных данных, эпох обучения - 10, 
mlp_model.fit(train_generator, epochs=10, validation_data=valid_generator)

# Оценка модели на тестовых данных
mlp_loss, mlp_accuracy = mlp_model.evaluate(test_generator)
print(f'MLP Test Accuracy: {mlp_accuracy}')


# In[20]:


X_test_rf = X_test_rf.reshape(-1, 224, 224, 3)

y_pred_mlp = mlp_model.predict(X_test_rf)
incorrect_indices = np.where(np.argmax(y_pred_mlp, axis=1) != np.argmax(y_test_rf, axis=1))[0]
num_incorrect = len(incorrect_indices)
print(f"Количество ошибочно предсказанных изображений: {num_incorrect}")

original_height = 224
original_width = 224
num_channels = 3

plt.figure(figsize=(15, 5))

num_to_display = min(num_incorrect, 10)  # Максимум 10 изображений

for i in range(num_to_display):
    index = incorrect_indices[i]
    plt.subplot(2, num_to_display // 2, i + 1) 
    plt.imshow(X_test_rf[index].reshape(original_height, original_width, num_channels)) 
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[21]:


# Создание сверточной нейронной сети


from tensorflow.keras.layers import Dropout
cnn_model = models.Sequential()
cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3))) # свёрточный слой с 32 нейронами и размером 3x3 с relu активацией 
cnn_model.add(layers.MaxPooling2D((2, 2))) # настраивает на выборку макисмальных значений блоков 2 на 2
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu')) # второй слой с 64 нейронами и размером 3x3 с relu активацией 
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(64, activation='relu'))
cnn_model.add(Dropout(0.3))
cnn_model.add(layers.Dense(len(train_generator.class_indices), activation='softmax')) # выходной слой с нейронами(кол-во совпадает с кол-вом классов)и активацией softmax

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение сверточной нейронной сети 
cnn_model.fit(train_generator, epochs=10, validation_data=valid_generator)

# Оценка модели
cnn_loss, cnn_accuracy = cnn_model.evaluate(valid_generator)
print(f'Validation Loss: {cnn_loss}')
print(f'Validation Accuracy: {cnn_accuracy}')

# Предсказание на тестовом наборе
test_generator.reset()
y_pred = cnn_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Получение классов
y_true = test_generator.classes

# Вывод результатов
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))


# In[22]:


y_pred_cnn = cnn_model.predict(X_test_rf)

# Находим индексы ошибочных предсказаний
incorrect_indices = np.where(np.argmax(y_pred_cnn, axis=1) != np.argmax(y_test_rf, axis=1))[0]
num_incorrect = len(incorrect_indices)
print(f"Количество ошибочно предсказанных изображений: {num_incorrect}")

# Параметры для отображения изображений
original_height = 224
original_width = 224
num_channels = 3

plt.figure(figsize=(15, 5))

num_to_display = min(num_incorrect, 10)  # Максимум 10 изображений

for i in range(num_to_display):
    index = incorrect_indices[i]
    plt.subplot(2, num_to_display // 2, i + 1) 
    plt.imshow(X_test_rf[index].reshape(original_height, original_width, num_channels)) 
    plt.axis('off')

plt.tight_layout()
plt.show()


# ## Расширение данных. Генерация новых данных на базе исходного набора с модификациями.

# In[23]:


import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Меняем параметры генерации данных для их расширения
train_datagen = ImageDataGenerator(
    rotation_range=90,        # позволяем вращать изображение на угол до 90 градусов(макс)
    width_shift_range=0.2,    # сдвигать по ширине до 20% (макс)
    height_shift_range=0.2,   # сдвигать по высоте до 20%(макс)
    zoom_range=0.5,           # увеличивать на 50%(макс)
    horizontal_flip=True,     # отражение по горизонатли
    fill_mode='nearest'       # метод заполнения пустых пикселей nearest
)


train_generator = train_datagen.flow_from_directory(
    os.path.dirname(train_dir),  
    target_size=(224, 224),     
    batch_size=16,               
    class_mode='categorical'      
)

valid_datagen = ImageDataGenerator()  
valid_generator = valid_datagen.flow_from_directory(
    os.path.dirname(valid_dir),  
    target_size=(224, 224),      
    batch_size=16,                
    class_mode='categorical'      
)


test_datagen = ImageDataGenerator()  
test_generator = test_datagen.flow_from_directory(
    os.path.dirname(test_dir),  
    target_size=(224, 224),    
    batch_size=16,               
    class_mode='categorical'     
)


# In[24]:


import matplotlib.pyplot as plt

# Берем один батч изображений и меток из генератора
images, labels = next(train_generator)
num_images = 9

#  Выводим изображения
plt.figure(figsize=(10, 10))

for i in range(num_images):
    # Создаем подграфик
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])  # Отображаем изображение
    plt.axis('off')
    plt.title(f'Class: {labels[i].argmax()}')  # Отображаем метку класса

plt.tight_layout()
plt.show()


# In[ ]:


# Модель Random Forest
X_train_rf = []
y_train_rf = []

# Получение данных из генератора
for _ in range(len(train_generator)):
    images, labels = next(train_generator) 
    X_train_rf.append(images)
    y_train_rf.append(labels)

# Объединение массивов 
X_train_rf = np.vstack(X_train_rf) 
y_train_rf = np.vstack(y_train_rf) 


X_train_rf = X_train_rf.reshape(X_train_rf.shape[0], -1)


rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_rf, np.argmax(y_train_rf, axis=1)) 


X_test_rf = []
y_test_rf = []

test_generator.reset()  
for _ in range(len(test_generator)):
    images, labels = next(test_generator)  # Используем next() для получения данных
    X_test_rf.append(images)
    y_test_rf.append(labels)


X_test_rf = np.vstack(X_test_rf)
y_test_rf = np.vstack(y_test_rf)

X_test_rf = X_test_rf.reshape(X_test_rf.shape[0], -1)



# In[30]:


y_pred_labels = y_pred_rf  # Получаем индексы предсказанных классов
y_test_labels = np.argmax(y_test_rf, axis=1)  # Получаем индексы истинных классов

# Расчет точности
accuracy = np.sum(y_pred_labels == y_test_labels) / len(y_test_labels) * 100
print(f"Процент правильных предсказаний: {accuracy:.2f}%")

# Вывод отчета о классификации
print("Random Forest Classification Report:")
print(classification_report(y_test_labels, y_pred_labels))


# In[31]:


import matplotlib.pyplot as plt
incorrect_indices = np.where(y_pred_rf != np.argmax(y_test_rf, axis=1))[0]

#Вывод ошибочно предсказанных изображений
num_incorrect = len(incorrect_indices)
print(f"Количество ошибочно предсказанных изображений: {num_incorrect}")

original_height = 224
original_width = 224
num_channels = 3

plt.figure(figsize=(15, 5))

num_to_display = min(num_incorrect, 10)  #максимум 10 изображений

for i in range(num_to_display):
    index = incorrect_indices[i]
    plt.subplot(2, num_to_display // 2, i + 1) 
    plt.imshow(X_test_rf[index].reshape(original_height, original_width, num_channels)) 
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[34]:


# Модель многослойного перцептрон
mlp_model = models.Sequential()
mlp_model.add(layers.Flatten(input_shape=(224, 224, 3))) # Преобразование в одномерный массив
mlp_model.add(layers.Dense(128, activation='relu')) # Полносвязный слой с 128 нейронами и relu активацией
mlp_model.add(layers.Dropout(0.3))
# Выходной слой с нейронмами (сколько классов столько нейронов) и активацией softmax
mlp_model.add(layers.Dense(len(train_generator.class_indices), activation='softmax')) 


# Компиляция модели с оптимизатором adam, функция потерь
# которая используется для многоклассовой классификации и выбранной метрикой оценки модели - точность (доля правильных предсказаний модели)
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели на тренировочных данных, эпох обучения - 10, 
mlp_model.fit(train_generator, epochs=10, validation_data=valid_generator)

# Оценка модели на тестовых данных
mlp_loss, mlp_accuracy = mlp_model.evaluate(test_generator)
print(f'MLP Test Accuracy: {mlp_accuracy}')

y_pred = mlp_model.predict(test_generator) # Предсказания модели в тестовом наборе
y_pred_classes = np.argmax(y_pred, axis=1) # Индекс класса с максимальной вероятностью попадания изображения
y_true = test_generator.classes  # получение классов 

X_test_rf = X_test_rf.reshape(-1, 224, 224, 3)

y_pred_mlp = mlp_model.predict(X_test_rf)
incorrect_indices = np.where(np.argmax(y_pred_mlp, axis=1) != np.argmax(y_test_rf, axis=1))[0]
num_incorrect = len(incorrect_indices)
print(f"Количество ошибочно предсказанных изображений: {num_incorrect}")

original_height = 224
original_width = 224
num_channels = 3

plt.figure(figsize=(15, 5))

num_to_display = min(num_incorrect, 10)  # Максимум 10 изображений

for i in range(num_to_display):
    index = incorrect_indices[i]
    plt.subplot(2, num_to_display // 2, i + 1) 
    plt.imshow(X_test_rf[index].reshape(original_height, original_width, num_channels)) 
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
# Создание модели
cnn_model = models.Sequential()
cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3))) # свёрточный слой с 32 нейронами и размером 3x3 с relu активацией 
cnn_model.add(layers.MaxPooling2D((2, 2))) # настраивает на выборку макисмальных значений блоков 2 на 2
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu')) # второй слой с 64 нейронами и размером 3x3 с relu активацией 
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(64, activation='relu'))
cnn_model.add(Dropout(0.3))
cnn_model.add(layers.Dense(len(train_generator.class_indices), activation='softmax'))

# Компиляция модели
cnn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
cnn_model.fit(train_generator, validation_data=valid_generator, epochs=10)

# Оценка модели
valid_generator.reset() 
predictions = cnn_model.predict(valid_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = valid_generator.classes


print(classification_report(true_classes, predicted_classes))


conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)

y_pred_cnn = cnn_model.predict(X_test_rf)


incorrect_indices = np.where(np.argmax(y_pred_cnn, axis=1) != np.argmax(y_test_rf, axis=1))[0]
num_incorrect = len(incorrect_indices)
print(f"Количество ошибочно предсказанных изображений: {num_incorrect}")


original_height = 224
original_width = 224
num_channels = 3

plt.figure(figsize=(15, 5))

num_to_display = min(num_incorrect, 10)  # Максимум 10 изображений

for i in range(num_to_display):
    index = incorrect_indices[i]
    plt.subplot(2, num_to_display // 2, i + 1) 
    plt.imshow(X_test_rf[index].reshape(original_height, original_width, num_channels)) 
    plt.axis('off')

plt.tight_layout()
plt.show()

