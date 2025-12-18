#!/usr/bin/env python
# coding: utf-8

# # model for VKR CNN+Embeding, text is vectorized

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, SpatialDropout1D
import numpy as np
import pickle


# In[10]:


# 1. Загрузка данных
data = pd.read_csv('D:/py/bot_detection_data.csv')


# In[11]:


# 2. Предварительная обработка данных
data.dropna(subset=['Tweet'], inplace=True)
X = data['Tweet']
y = data['Bot Label']

# 3. Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Векторизация текста
vectorizer = TfidfVectorizer(max_features=5000)  # Максимум 5000 уникальных слов
X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
X_test_vectorized = vectorizer.transform(X_test).toarray()

# 5. Изменение размерности для входа в CNN
X_train_vectorized = np.expand_dims(X_train_vectorized, axis=2)
X_test_vectorized = np.expand_dims(X_test_vectorized, axis=2)

# 6. Создание модели CNN
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X_train_vectorized.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Выходной слой для бинарной классификации

# 7. Компиляция модели
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 8. Обучение модели
history = model.fit(X_train_vectorized, y_train, epochs=5, batch_size=64, validation_data=(X_test_vectorized, y_test), verbose=2)

# 9. Оценка модели
score, accuracy = model.evaluate(X_test_vectorized, y_test, verbose=2)
print(f'Точность модели: {accuracy * 100:.2f}%')

# 10. Применение модели для предсказания
def predict_bots(tweets):
    tweet_vectorized = vectorizer.transform(tweets).toarray()
    tweet_vectorized = np.expand_dims(tweet_vectorized, axis=2)
    predictions = model.predict(tweet_vectorized)
    return (predictions > 0.5).astype(int)

# Пример предсказания
new_tweets = ["Это великолепный продукт!", "Купите наш продукт сейчас!", "Я не бот, а настоящий человек"]
predictions = predict_bots(new_tweets)
print(predictions)


# In[16]:


import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def train_and_save_model(data_path):
    # Загрузка и подготовка данных
    data = pd.read_csv(data_path)
    data.dropna(subset=['Tweet'], inplace=True)
    X = data['Tweet'].astype(str)
    y = data['Bot Label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Векторизация
    max_features = 5000
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vectorized = vectorizer.fit_transform(X_train).toarray()  # shape (n_samples, max_features)
    X_test_vectorized = vectorizer.transform(X_test).toarray()

    # Паддинг/усечение последовательностей по длине (если нужно) — здесь maxlen задаём равным max_features
    maxlen = max_features
    X_train_padded = pad_sequences(X_train_vectorized, padding='post', maxlen=maxlen, dtype='float32')
    X_test_padded = pad_sequences(X_test_vectorized, padding='post', maxlen=maxlen, dtype='float32')

    # Подготовка формы для Conv1D/LSTM: (timesteps, features). 
    # Интерпретируем каждую TF-IDF векторную компоненту как timestep и используем 1 feature на timestep
    X_train_input = np.expand_dims(X_train_padded, axis=2)  # shape (n_samples, maxlen, 1)
    X_test_input = np.expand_dims(X_test_padded, axis=2)

    # Построение модели
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(maxlen, 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        LSTM(64, return_sequences=False),
        Dropout(0.25),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # предполагается бинарная классификация
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Обучение
    history = model.fit(
        X_train_input, y_train,
        validation_data=(X_test_input, y_test),
        epochs=5,
        batch_size=32
    )

    # Сохранение модели в нативном формате Keras (.keras) и векторизатора
    model.save('bot_detection_model.keras')
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Оценка на тесте
    loss, acc = model.evaluate(X_test_input, y_test, verbose=0)

    return {
        'model_path': 'bot_detection_model.keras',
        'vectorizer_path': 'tfidf_vectorizer.pkl',
        'test_loss': float(loss),
        'test_accuracy': float(acc),
        'history': history.history
    }


# In[17]:


train_and_save_model('D:/py/bot_detection_data.csv')


# In[2]:


import pickle
from tensorflow.keras.models import load_model

def load_model_and_vectorizer():
    # Загрузка модели (нативный формат Keras .keras)
    model = load_model('bot_detection_model.keras')

    # Загрузка векторизатора
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


# In[3]:


# Загрузка модели и векторизации для последующего использования
model, vectorizer = load_model_and_vectorizer()

# Пример предсказания
def predict_bots(tweets):
    tweet_vectorized = vectorizer.transform(tweets).toarray()
    tweet_vectorized = np.expand_dims(tweet_vectorized, axis=2)
    predictions = model.predict(tweet_vectorized)
    return (predictions > 0.5).astype(int)



# In[ ]:


import pandas as pd
import random
import string
from typing import Optional, List
from datetime import datetime

class RealisticCommentGenerator:
    """Генератор реалистичных комментариев с реальными английскими словами"""

    # Словарь реальных английских слов
    ENGLISH_WORDS = [
        # Common words
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",

        # Verbs
        "is", "are", "was", "were", "have", "has", "had", "do", "does", "did",
        "can", "could", "will", "would", "should", "may", "might", "must",
        "think", "believe", "feel", "know", "understand", "agree", "disagree",
        "like", "love", "hate", "enjoy", "appreciate", "support", "share",
        "read", "write", "comment", "discuss", "learn", "teach", "explain",

        # Nouns
        "article", "post", "content", "topic", "point", "idea", "thought",
        "perspective", "view", "opinion", "argument", "discussion", "comment",
        "information", "knowledge", "experience", "work", "job", "research",
        "author", "writer", "reader", "people", "person", "someone", "everyone",
        "time", "day", "year", "life", "world", "thing", "something", "nothing",

        # Adjectives
        "good", "great", "excellent", "amazing", "wonderful", "awesome",
        "fantastic", "perfect", "brilliant", "nice", "cool", "interesting",
        "helpful", "useful", "informative", "important", "valuable", "essential",
        "clear", "concise", "insightful", "thoughtful", "meaningful", "relevant",
        "true", "false", "right", "wrong", "different", "similar", "same",

        # Adverbs
        "very", "really", "quite", "truly", "especially", "particularly",
        "actually", "basically", "essentially", "generally", "usually",
        "sometimes", "often", "always", "never", "almost", "nearly",

        # Transition words
        "however", "therefore", "moreover", "furthermore", "additionally",
        "consequently", "nevertheless", "nonetheless", "meanwhile",
        "firstly", "secondly", "finally", "in conclusion", "to summarize",

        # Question words
        "what", "why", "how", "when", "where", "who", "which"
    ]

    # Дополнительные тематические слова
    TOPIC_WORDS = {
        "technology": ["digital", "online", "software", "hardware", "computer", 
                      "internet", "data", "algorithm", "system", "code"],
        "science": ["research", "study", "experiment", "theory", "evidence",
                   "scientific", "method", "discovery", "innovation"],
        "education": ["learning", "teaching", "student", "teacher", "school",
                     "university", "knowledge", "skill", "education"],
        "politics": ["government", "policy", "democracy", "freedom", "rights",
                    "election", "vote", "leader", "political"],
        "health": ["health", "medical", "doctor", "patient", "treatment",
                  "medicine", "wellness", "fitness", "nutrition"]
    }

    @staticmethod
    def get_random_word() -> str:
        """Возвращает случайное реальное английское слово"""
        return random.choice(RealisticCommentGenerator.ENGLISH_WORDS)

    @staticmethod
    def get_realistic_sentence(num_words: int, topic: Optional[str] = None) -> str:
        """Генерирует реалистичное предложение"""
        words = []

        for i in range(num_words):
            # Добавляем разнообразие - иногда используем тематические слова
            if topic and random.random() < 0.3:
                topic_words = RealisticCommentGenerator.TOPIC_WORDS.get(topic, [])
                if topic_words:
                    word = random.choice(topic_words)
                else:
                    word = RealisticCommentGenerator.get_random_word()
            else:
                word = RealisticCommentGenerator.get_random_word()

            # Первое слово с заглавной буквы
            if i == 0:
                word = word.capitalize()

            words.append(word)

        sentence = ' '.join(words)

        # Добавляем знаки препинания
        punctuation = random.choices(['.', '!', '?'], weights=[0.7, 0.2, 0.1])[0]
        sentence += punctuation

        return sentence

    @staticmethod
    def get_random_username() -> str:
        """Генерирует случайное имя пользователя"""
        prefixes = ['user', 'anon', 'reader', 'viewer', 'guest', 'commenter',
                   'thinker', 'writer', 'observer', 'participant']
        suffixes = ['123', '2023', '2024', '99', '42', '7', 'alpha', 'beta']
        numbers = ['', str(random.randint(1, 999))]

        parts = [
            random.choice(prefixes),
            random.choice(suffixes),
            random.choice(numbers)
        ]

        return ''.join(filter(None, parts))

    @staticmethod
    def get_random_timestamp() -> str:
        """Генерирует случайную дату и время"""
        year = random.randint(2023, 2024)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        return f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"


def generate_realistic_comments(
    is_short: bool = True,
    num_comments: int = 100,
    seed: Optional[int] = None,
    filename: Optional[str] = None,
    topic: Optional[str] = None
) -> pd.DataFrame:
    """
    Генерирует реалистичный датасет с английскими комментариями.

    Args:
        is_short (bool): True - короткие комментарии (1-10 слов), 
                        False - длинные комментарии (11-300 слов)
        num_comments (int): Количество комментариев для генерации
        seed (int, optional): Seed для воспроизводимости
        filename (str, optional): Имя файла для сохранения
        topic (str, optional): Тематика комментариев

    Returns:
        pd.DataFrame: Сгенерированный датасет с комментариями
    """

    if seed is not None:
        random.seed(seed)

    comments_data = []
    generator = RealisticCommentGenerator()


    SHORT_TEMPLATES = [
        # Полные предложения
        "This is a great article!",
        "Thanks for sharing this.",
        "I completely agree with you.",
        "Very interesting perspective.",
        "Well written and informative.",
        "This was really helpful, thank you.",
        "I learned something new today.",
        "Excellent point you made here.",
        "Couldn't have said it better myself.",
        "Food for thought indeed.",


        "Great post!",
        "Nice work!",
        "Well said!",
        "Good point.",
        "I agree.",
        "Interesting!",
        "Helpful info.",
        "Thanks!",
        "Well done.",
        "Spot on!",


        "Has anyone else experienced this?",
        "What do others think?",
        "Can you explain further?",
        "Why is this important?",
        "How does this work?",


        "Love this!",
        "Amazing content!",
        "Brilliant analysis!",
        "Fantastic read!",
        "Perfect explanation!",


        "I have a different perspective.",
        "While I agree, I also think...",
        "This reminds me of something similar.",
        "Looking forward to more on this topic.",
        "Keep up the good work!"
    ]


    LONG_STARTERS = [
        "In my opinion,",
        "From my experience,",
        "I've been thinking about this and",
        "What I find interesting is that",
        "Based on what I've read,",
        "To add to the discussion,",
        "Building on this idea,",
        "I would like to share that",
        "It's important to consider that",
        "What strikes me is how"
    ]


    LONG_MIDDLES = [
        "this raises important questions about",
        "the evidence suggests that",
        "many people tend to overlook",
        "this connects to the broader issue of",
        "we should also think about",
        "another aspect to consider is",
        "this has implications for",
        "what this means in practice is",
        "the real challenge lies in",
        "this demonstrates why"
    ]

    # Заключительные фразы
    LONG_ENDINGS = [
        "and that's why this matters.",
        "which is something we should all consider.",
        "so let's continue this discussion.",
        "and I'm curious what others think.",
        "and I believe this is worth exploring further.",
        "which leads me to my final point.",
        "and this is just my perspective.",
        "so thank you for raising this topic.",
        "and I hope this adds to the conversation.",
        "which makes this such an important discussion."
    ]

    for i in range(num_comments):
        if is_short:
            # Короткие комментарии (1-10 слов)
            # 80% - готовые шаблоны, 20% - сгенерированные
            if random.random() < 0.8:
                comment = random.choice(SHORT_TEMPLATES)

                # Иногда немного меняем шаблон
                if random.random() < 0.3:
                    comment = comment.replace("!", ".").replace(".", "!")
                elif random.random() < 0.2:
                    # Добавляем эмоцию в конец
                    emotions = [" :)", "!", "!!", " :D", " :)"]
                    comment = comment.rstrip(".!?") + random.choice(emotions)
            else:
                # Генерируем короткий комментарий
                num_words = random.randint(3, 10)
                comment = generator.get_realistic_sentence(num_words, topic)

            comment_type = "short"

        else:
            # Длинные комментарии (11-300 слов)
            num_words = random.randint(11, 300)

            # Стратегия генерации в зависимости от длины
            if num_words < 50:
                # 2-4 предложения
                sentences = []

                # Первое предложение - с начальной фразой
                starter = random.choice(LONG_STARTERS)
                first_sentence = starter + " " + generator.get_realistic_sentence(
                    random.randint(5, 15), topic
                ).lower()
                sentences.append(first_sentence)

                # Добавляем еще 1-3 предложения
                for _ in range(random.randint(1, 3)):
                    sentences.append(generator.get_realistic_sentence(
                        random.randint(8, 20), topic
                    ))

                comment = ' '.join(sentences)

            elif num_words < 150:
                # Небольшой абзац
                sentences = []
                words_used = 0

                while words_used < num_words:
                    if len(sentences) == 0:
                        # Первое предложение с начальной фразой
                        starter = random.choice(LONG_STARTERS)
                        sentence_words = random.randint(10, 20)
                        sentence = starter + " " + generator.get_realistic_sentence(
                            sentence_words, topic
                        ).lower()
                    elif len(sentences) < 3:
                        # Средние предложения с соединительными фразами
                        middle = random.choice(LONG_MIDDLES)
                        sentence_words = random.randint(8, 18)
                        sentence = middle.capitalize() + " " + generator.get_realistic_sentence(
                            sentence_words, topic
                        ).lower()
                    else:
                        # Обычные предложения
                        sentence_words = random.randint(6, 15)
                        sentence = generator.get_realistic_sentence(sentence_words, topic)

                    sentences.append(sentence)
                    words_used += len(sentence.split())

                comment = ' '.join(sentences)

            else:
                # Несколько абзацев
                paragraphs = []
                words_used = 0

                while words_used < num_words:
                    paragraph_sentences = []
                    paragraph_words = 0

                    # Определяем длину абзаца
                    paragraph_target = min(random.randint(40, 80), num_words - words_used)

                    while paragraph_words < paragraph_target:
                        if len(paragraph_sentences) == 0:
                            # Первое предложение абзаца
                            if len(paragraphs) == 0:
                                starter = random.choice(LONG_STARTERS)
                                sentence = starter + " " + generator.get_realistic_sentence(
                                    random.randint(12, 20), topic
                                ).lower()
                            else:
                                starters = ["Furthermore,", "Additionally,", "Moreover,", 
                                          "Another important point is that", 
                                          "It's also worth noting that"]
                                starter = random.choice(starters)
                                sentence = starter + " " + generator.get_realistic_sentence(
                                    random.randint(10, 18), topic
                                ).lower()
                        else:
                            # Обычное предложение
                            sentence = generator.get_realistic_sentence(
                                random.randint(8, 16), topic
                            )

                        paragraph_sentences.append(sentence)
                        paragraph_words += len(sentence.split())

                    paragraphs.append(' '.join(paragraph_sentences))
                    words_used += paragraph_words

                comment = '\n\n'.join(paragraphs)
                comment_type = "long"

        # Проверяем и корректируем количество слов
        actual_words = len(comment.split())

        if is_short:
            # Для коротких: обрезаем если больше 10 слов
            if actual_words > 10:
                words = comment.split()
                comment = ' '.join(words[:10])
                actual_words = 10
            # И добавляем если меньше 1 слова
            elif actual_words < 1:
                comment = "Good point."
                actual_words = 2
        else:
            # Для длинных: добавляем если меньше 11 слов
            if actual_words < 11:
                additional = generator.get_realistic_sentence(random.randint(5, 10), topic)
                comment = comment + " " + additional
                actual_words = len(comment.split())

        comments_data.append({
            'id': i + 1,
            'text': comment,
            'word_count': actual_words,
            'comment_type': 'short' if is_short else 'long',
            'author': generator.get_random_username(),
            'created_at': generator.get_random_timestamp(),
            'topic': topic if topic else 'general'
        })

    # Создаем DataFrame
    df = pd.DataFrame(comments_data)

    # Определяем имя файла
    if filename is None:
        filename = "short.csv" if is_short else "long.csv"

    # Сохраняем в CSV
    df.to_csv(filename, index=False, encoding='utf-8')

    # Выводим статистику
    print(f"\n{'='*60}")
    print(f" УСПЕШНО СОЗДАНО: {num_comments} комментариев")
    print(f"ТИП: {'Короткие (1-10 слов)' if is_short else 'Длинные (11-300 слов)'}")

    print(f" СТАТИСТИКА:")
    print(f"   • Минимум слов: {df['word_count'].min()}")
    print(f"   • Максимум слов: {df['word_count'].max()}")
    print(f"   • Среднее слов: {df['word_count'].mean():.1f}")
    print(f"   • Медиана слов: {df['word_count'].median():.1f}")
    print(f"{'='*60}")



if __name__ == "__main__":

    # Пример 1: Короткие комментарии
    print("\n1. Генерация 20000 коротких комментариев:")
    short_df = generate_realistic_comments(
        is_short=True,
        num_comments=5000,
        seed=42,
        filename="short.csv"
    )

    # Пример 2: Длинные комментарии
    print("\n2. Генерация 20000 длинных комментариев на тему технологии:")
    long_df = generate_realistic_comments(
        is_short=False,
        num_comments=5000,
        seed=123,
        filename="long.csv",
        topic="technology"
    )




# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import os
import re
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

class LoadTester:
    """Класс для нагрузочного тестирования модели с обработкой ошибок векторизации"""

    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self.results = {}
        self.log_file = "load_test_log.txt"
        self.error_log = "load_test_errors.txt"

        # Очищаем лог файлы
        with open(self.log_file, 'w') as f:
            f.write("Лог нагрузочного тестирования\n")
            f.write("=" * 50 + "\n\n")

        with open(self.error_log, 'w') as f:
            f.write("Лог ошибок тестирования\n")
            f.write("=" * 50 + "\n\n")

    def log_message(self, message: str, is_error: bool = False):
        """Запись сообщения в лог файл"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"

        if is_error:
            with open(self.error_log, 'a') as f:
                f.write(log_line)
        else:
            with open(self.log_file, 'a') as f:
                f.write(log_line)

        print(log_line.strip())

    def preprocess_text(self, text: str) -> str:
        """Предобработка текста для совместимости с векторизатором"""
        # Приводим к нижнему регистру
        text = str(text).lower()

        # Удаляем специальные символы, оставляем только буквы, цифры и пробелы
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Заменяем множественные пробелы на один
        text = re.sub(r'\s+', ' ', text).strip()

        # Если текст пустой после обработки, возвращаем дефолтное значение
        if not text:
            return "default comment"

        return text

    def load_comments_from_file(self, filename: str, max_comments: int = None) -> List[str]:
        """Загрузка и предобработка комментариев из CSV файла"""
        try:
            df = pd.read_csv(filename)
            comments = df['text'].tolist()

            # Предобрабатываем каждый комментарий
            processed_comments = []
            for i, comment in enumerate(comments):
                try:
                    processed = self.preprocess_text(comment)
                    processed_comments.append(processed)
                except Exception as e:
                    self.log_message(f"Ошибка обработки комментария {i} в {filename}: {e}", is_error=True)
                    # Добавляем дефолтное значение
                    processed_comments.append("default comment")

            if max_comments:
                processed_comments = processed_comments[:max_comments]

            self.log_message(f"Загружено и обработано {len(processed_comments)} комментариев из {filename}")
            return processed_comments

        except Exception as e:
            self.log_message(f"Ошибка загрузки {filename}: {e}", is_error=True)
            return []

    def safe_predict(self, texts: List[str]) -> Tuple[List[bool], List[float]]:
        """Безопасное предсказание с обработкой ошибок"""
        try:
            # Векторизация текстов
            text_vectorized = self.vectorizer.transform(texts)

            # Предсказание
            predictions_proba = self.model.predict(text_vectorized, verbose=0)

            # Проверяем результат
            if len(predictions_proba) == 0:
                raise ValueError("Модель вернула пустой результат")

            predictions = (predictions_proba >= 0.5).flatten()
            probabilities = predictions_proba.flatten()

            return predictions, probabilities

        except Exception as e:
            self.log_message(f"Ошибка при предсказании: {e}", is_error=True)
            # Возвращаем дефолтные значения
            default_predictions = [False] * len(texts)
            default_probabilities = [0.0] * len(texts)
            return default_predictions, default_probabilities

    def run_test_for_file(self, filename: str, batch_sizes: List[int]):
        """Запуск тестирования для файла"""
        self.log_message(f"\nНачало тестирования для файла: {filename}")

        # Загружаем все комментарии
        all_comments = self.load_comments_from_file(filename)
        if not all_comments:
            self.log_message(f"Нет комментариев для тестирования из {filename}", is_error=True)
            return None

        results = []

        for batch_size in batch_sizes:
            if batch_size > len(all_comments):
                self.log_message(f"Пропускаем размер {batch_size}: недостаточно комментариев в файле", is_error=True)
                continue

            self.log_message(f"Тестирование на {batch_size} комментариях...")

            # Берем batch_size комментариев
            batch_comments = all_comments[:batch_size]

            # Измеряем метрики системы до
            cpu_before = psutil.cpu_percent(interval=0.1)
            ram_before = psutil.virtual_memory().used / (1024**3)

            # Замер времени
            start_time = time.time()
            predictions, probabilities = self.safe_predict(batch_comments)
            elapsed_time = time.time() - start_time

            # Измеряем метрики системы после
            cpu_after = psutil.cpu_percent(interval=0.1)
            ram_after = psutil.virtual_memory().used / (1024**3)

            # Собираем результаты
            bot_count = np.sum(predictions)
            bot_percentage = (bot_count / batch_size) * 100

            result = {
                'batch_size': batch_size,
                'total_time_sec': elapsed_time,
                'time_per_comment_ms': (elapsed_time / batch_size) * 1000,
                'cpu_before': cpu_before,
                'cpu_after': cpu_after,
                'cpu_max': max(cpu_before, cpu_after),
                'ram_before_gb': ram_before,
                'ram_after_gb': ram_after,
                'ram_max_gb': max(ram_before, ram_after),
                'bot_count': bot_count,
                'bot_percentage': bot_percentage,
                'prob_mean': np.mean(probabilities),
                'prob_std': np.std(probabilities),
                'prob_min': np.min(probabilities),
                'prob_max': np.max(probabilities)
            }

            results.append(result)

            self.log_message(f"  Время: {elapsed_time:.2f} сек ({result['time_per_comment_ms']:.1f} мс/коммент)")
            self.log_message(f"  CPU: {cpu_before:.1f}% -> {cpu_after:.1f}%")
            self.log_message(f"  RAM: {ram_before:.2f} -> {ram_after:.2f} GB")
            self.log_message(f"  Ботов: {bot_count} ({bot_percentage:.1f}%)")

        if results:
            return pd.DataFrame(results)
        else:
            self.log_message(f"Нет результатов для {filename}", is_error=True)
            return None

    def run_load_test(self, batch_sizes: List[int] = None):
        """Запуск нагрузочного тестирования с указанием размеров батчей"""
        self.log_message("=" * 50)
        self.log_message("НАЧАЛО НАГРУЗОЧНОГО ТЕСТИРОВАНИЯ")
        self.log_message("=" * 50)

        # Если размеры батчей не указаны, используем дефолтные
        if batch_sizes is None:
            batch_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

        self.log_message(f"Размеры батчей для тестирования: {batch_sizes}")

        # Тестируем для каждого файла
        files_to_test = ['short.csv', 'long.csv']
        for filename in files_to_test:
            if os.path.exists(filename):
                self.log_message(f"\nОбработка файла: {filename}")
                results_df = self.run_test_for_file(filename, batch_sizes)
                if results_df is not None and not results_df.empty:
                    self.results[filename] = results_df
                    # Сохраняем результаты в CSV
                    results_filename = f'results_{os.path.splitext(filename)[0]}.csv'
                    results_df.to_csv(results_filename, index=False)
                    self.log_message(f"Результаты сохранены в {results_filename}")
                else:
                    self.log_message(f"Не удалось получить результаты для {filename}", is_error=True)
            else:
                self.log_message(f"Файл {filename} не найден, пропускаем", is_error=True)

    def plot_results(self):
        """Построение графиков результатов"""
        if not self.results:
            self.log_message("Нет данных для построения графиков")
            return

        # Создаем директорию для графиков
        os.makedirs("load_test_graphs", exist_ok=True)

        # Для каждого файла строим графики
        for filename, results_df in self.results.items():
            if results_df.empty:
                continue

            # Определяем тип комментариев из имени файла
            if 'short' in filename:
                comment_type = 'Короткие'
                color = 'blue'
            else:
                comment_type = 'Длинные'
                color = 'red'

            # 1. График времени выполнения
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Результаты тестирования: {comment_type} комментарии', fontsize=14)

            # График 1: Общее время выполнения
            ax = axes[0, 0]
            ax.plot(results_df['batch_size'], results_df['total_time_sec'], 
                   f'{color[0]}-o', linewidth=2)
            ax.set_xlabel('Количество комментариев')
            ax.set_ylabel('Общее время (сек)')
            ax.set_title('Общее время выполнения')
            ax.grid(True, alpha=0.3)

            # График 2: Время на один комментарий
            ax = axes[0, 1]
            ax.plot(results_df['batch_size'], results_df['time_per_comment_ms'], 
                   f'{color[0]}-s', linewidth=2)
            ax.set_xlabel('Количество комментариев')
            ax.set_ylabel('Время на комментарий (мс)')
            ax.set_title('Время на один комментарий')
            ax.grid(True, alpha=0.3)

            # График 3: Использование CPU
            ax = axes[1, 0]
            ax.plot(results_df['batch_size'], results_df['cpu_max'], 
                   f'{color[0]}-^', linewidth=2)
            ax.set_xlabel('Количество комментариев')
            ax.set_ylabel('CPU (%)')
            ax.set_title('Максимальное использование CPU')
            ax.grid(True, alpha=0.3)

            # График 4: Использование RAM
            ax = axes[1, 1]
            ax.plot(results_df['batch_size'], results_df['ram_max_gb'], 
                   f'{color[0]}-d', linewidth=2)
            ax.set_xlabel('Количество комментариев')
            ax.set_ylabel('RAM (GB)')
            ax.set_title('Максимальное использование RAM')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            graph_filename = f'load_test_graphs/{comment_type.lower()}_graphs.png'
            plt.savefig(graph_filename, dpi=150, bbox_inches='tight')
            plt.close()

            self.log_message(f"Графики сохранены в {graph_filename}")

        # Сравнительный график если есть оба типа
        if len(self.results) >= 2:
            try:
                short_df = self.results.get('short.csv')
                long_df = self.results.get('long.csv')

                if short_df is not None and long_df is not None and not short_df.empty and not long_df.empty:
                    # Находим общие размеры батчей
                    common_sizes = sorted(set(short_df['batch_size']).intersection(set(long_df['batch_size'])))

                    if common_sizes:
                        # Фильтруем данные
                        short_filtered = short_df[short_df['batch_size'].isin(common_sizes)].sort_values('batch_size')
                        long_filtered = long_df[long_df['batch_size'].isin(common_sizes)].sort_values('batch_size')

                        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                        fig.suptitle('Сравнение коротких и длинных комментариев', fontsize=14)

                        # Сравнение времени на комментарий
                        ax = axes[0, 0]
                        ax.plot(common_sizes, short_filtered['time_per_comment_ms'], 'b-o', 
                               linewidth=2, label='Короткие')
                        ax.plot(common_sizes, long_filtered['time_per_comment_ms'], 'r-s', 
                               linewidth=2, label='Длинные')
                        ax.set_xlabel('Количество комментариев')
                        ax.set_ylabel('Время на комментарий (мс)')
                        ax.set_title('Сравнение времени обработки')
                        ax.legend()
                        ax.grid(True, alpha=0.3)

                        # Сравнение CPU
                        ax = axes[0, 1]
                        ax.plot(common_sizes, short_filtered['cpu_max'], 'b-o', 
                               linewidth=2, label='Короткие')
                        ax.plot(common_sizes, long_filtered['cpu_max'], 'r-s', 
                               linewidth=2, label='Длинные')
                        ax.set_xlabel('Количество комментариев')
                        ax.set_ylabel('CPU (%)')
                        ax.set_title('Сравнение использования CPU')
                        ax.legend()
                        ax.grid(True, alpha=0.3)

                        # Сравнение RAM
                        ax = axes[1, 0]
                        ax.plot(common_sizes, short_filtered['ram_max_gb'], 'b-o', 
                               linewidth=2, label='Короткие')
                        ax.plot(common_sizes, long_filtered['ram_max_gb'], 'r-s', 
                               linewidth=2, label='Длинные')
                        ax.set_xlabel('Количество комментариев')
                        ax.set_ylabel('RAM (GB)')
                        ax.set_title('Сравнение использования RAM')
                        ax.legend()
                        ax.grid(True, alpha=0.3)

                        # Сравнение процента ботов
                        ax = axes[1, 1]
                        ax.plot(common_sizes, short_filtered['bot_percentage'], 'b-o', 
                               linewidth=2, label='Короткие')
                        ax.plot(common_sizes, long_filtered['bot_percentage'], 'r-s', 
                               linewidth=2, label='Длинные')
                        ax.set_xlabel('Количество комментариев')
                        ax.set_ylabel('% обнаруженных ботов')
                        ax.set_title('Сравнение обнаружения ботов')
                        ax.legend()
                        ax.grid(True, alpha=0.3)

                        plt.tight_layout()
                        plt.savefig('load_test_graphs/comparison_graphs.png', dpi=150, bbox_inches='tight')
                        plt.close()

                        self.log_message("Сравнительные графики сохранены в load_test_graphs/comparison_graphs.png")
            except Exception as e:
                self.log_message(f"Ошибка при создании сравнительных графиков: {e}", is_error=True)

    def generate_summary_report(self):
        """Генерация итогового отчета"""
        if not self.results:
            self.log_message("Нет результатов для генерации отчета")
            return

        report_file = "load_test_summary.txt"
        self.log_message(f"\nГенерация итогового отчета в {report_file}")

        with open(report_file, 'w') as f:
            f.write("ИТОГОВЫЙ ОТЧЕТ ПО НАГРУЗОЧНОМУ ТЕСТИРОВАНИЮ\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Дата тестирования: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Анализ словаря векторизатора
            try:
                vocab_size = len(self.vectorizer.vocabulary_)
                f.write(f"РАЗМЕР СЛОВАРЯ ВЕКТОРИЗАТОРА: {vocab_size} слов\n\n")
            except:
                f.write("Информация о словаре векторизатора недоступна\n\n")

            for filename, results_df in self.results.items():
                if results_df.empty:
                    continue

                if 'short' in filename:
                    comment_type = 'КОРОТКИЕ'
                else:
                    comment_type = 'ДЛИННЫЕ'

                f.write(f"{comment_type} КОММЕНТАРИИ:\n")
                f.write("-" * 40 + "\n")

                # Основная статистика
                max_size = results_df['batch_size'].max()
                if not results_df[results_df['batch_size'] == max_size].empty:
                    max_row = results_df[results_df['batch_size'] == max_size].iloc[0]

                    f.write(f"Максимальный протестированный размер: {max_size} комментариев\n")
                    f.write(f"Время обработки {max_size} комментариев: {max_row['total_time_sec']:.2f} сек\n")
                    f.write(f"Среднее время на комментарий: {max_row['time_per_comment_ms']:.1f} мс\n")
                    f.write(f"Максимальное использование CPU: {max_row['cpu_max']:.1f}%\n")
                    f.write(f"Максимальное использование RAM: {max_row['ram_max_gb']:.2f} GB\n")
                    f.write(f"Обнаружено ботов в последнем тесте: {max_row['bot_count']} ({max_row['bot_percentage']:.1f}%)\n")
                    f.write(f"Средняя вероятность бота: {max_row['prob_mean']:.3f}\n\n")

            # Системная информация
            f.write("СИСТЕМНАЯ ИНФОРМАЦИЯ:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Процессор: {psutil.cpu_count()} ядер\n")
            f.write(f"Объем RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB\n")
            f.write(f"Система: {os.name}\n\n")

            # Статистика по ошибкам
            try:
                error_count = 0
                if os.path.exists(self.error_log):
                    with open(self.error_log, 'r') as err_file:
                        error_count = len(err_file.readlines()) - 3  # минус заголовок
                f.write(f"Количество зарегистрированных ошибок: {error_count}\n")
            except:
                pass

            f.write("=" * 60 + "\n")
            f.write("Тестирование завершено\n")


# Функция для запуска тестирования
def run_comprehensive_load_test(model, vectorizer):
    """Запуск комплексного нагрузочного тестирования"""

    print("Инициализация нагрузочного тестирования...")

    # Создаем тестер с исправлениями
    tester = LoadTester(model, vectorizer)

    # Определяем размеры батчей для тестирования
    # Шаг 50 до 5000:
    batch_sizes = list(range(50, 3001, 50))
    # Всего: 100 значений (от 50 до 5000 с шагом 50)

    # Запускаем тестирование с указанием размеров батчей
    tester.run_load_test(batch_sizes=batch_sizes)

    # Строим графики
    tester.plot_results()

    # Генерируем отчет
    tester.generate_summary_report()

    print("\nНагрузочное тестирование завершено!")
    print("Проверьте файлы:")
    print("1. load_test_log.txt - детальный лог выполнения")
    print("2. load_test_errors.txt - ошибки (если есть)")
    print("3. load_test_summary.txt - итоговый отчет")
    print("4. results_short.csv - результаты для коротких комментариев")
    print("5. results_long.csv - результаты для длинных комментариев")
    print("6. Папка load_test_graphs/ - графики результатов")

    return tester


# Пример использования
if __name__ == "__main__":
    try:
        # Загружаем модель и векторизатор
        from tensorflow.keras.models import load_model
        import pickle

        print("Загрузка модели...")
        model = load_model('bot_detection_model.keras')

        print("Загрузка векторизатора...")
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        # Проверяем наличие файлов с комментариями
        if not os.path.exists('short.csv'):
            print("Ошибка: файл short.csv не найден")
        elif not os.path.exists('long.csv'):
            print("Ошибка: файл long.csv не найден")
        else:
            # Запускаем тестирование
            tester = run_comprehensive_load_test(model, vectorizer)

    except FileNotFoundError as e:
        print(f"Ошибка загрузки файлов: {e}")
        print("Убедитесь, что в текущей директории есть:")
        print("- bot_detection_model.keras")
        print("- tfidf_vectorizer.pkl")
        print("- short.csv")
        print("- long.csv")
    except Exception as e:
        print(f"Критическая ошибка: {e}")


# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Загрузка стоп-слов и пунктуации
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('russian'))
punctuation = set(string.punctuation)


# In[ ]:


def preprocess_text(text):
    words = word_tokenize(text.lower())  # Привести к нижнему регистру и токенизировать
    filtered_words = [word for word in words if word not in stop_words and word not in punctuation]
    return " ".join(filtered_words)


# In[ ]:


# Функция для проверки наличия модели и векторизации
def check_and_load_model(data_path):
    model_file = 'bot_detection_model.h5'
    vectorizer_file = 'tfidf_vectorizer.pkl'

    try:
        if os.path.exists(model_file) and os.path.exists(vectorizer_file):
            model = load_model(model_file)
            with open(vectorizer_file, 'rb') as f:
                vectorizer = pickle.load(f)
            return model, vectorizer
        else:
            train_and_save_model(data_path)
            return check_and_load_model(data_path)
    except Exception as e:
        print(f"Произошла ошибка: {e}")

# Использование функции
model, vectorizer = check_and_load_model('bot_detection_dataset.csv')

