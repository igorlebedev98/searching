#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as nd
import statistics as st
import matplotlib.pyplot as plt
import seaborn as sns
import mlxtend.frequent_patterns as ml


# # Лабораторная работа 2

# ## Выбор группы из нескольких стран

# In[3]:


data = pd.read_csv("lastfm.csv")
data['country'].value_counts()


# In[4]:


#Выбираем данные для стран Россия и Беларуссия
sweden = data[data.country == 'Sweden']
belarus = data[data.country == 'Belarus']
finland = data[data.country == 'Finland']
print(sweden.shape)
sweden.sample(5)


# ## Подготовка набора для поиска ассоциативных правил

# In[5]:


#Группируем по пользоватлям
group = data.groupby('user')['artist'].apply(';'.join)
print(group)


# In[6]:


#Бинаризация артистов по пользователям
bin = group.str.get_dummies(";").astype(bool)
print(bin)


# ## Поиск характерных комбинаций 

# In[ ]:


# Характерные комбинации для бинаризованных данных по исполнителям пользователей. 
#Выбираем исполнителей, за которым следят минимум 5% пользователей из выбранных стран

freq_items = ml.apriori(bin, min_support = 0.05, use_colnames = True)
print("Найдено %d характерных комбинаций" % len(freq_items))
freq_items


# In[8]:


#Характерные комбинации по полу - мужчины
man = data[data.sex == 'm']
# группируем и бинуризируем
group_m = man.groupby('user')['artist'].apply(';'.join)
bin_m = group_m.str.get_dummies(";").astype(bool)
# ищем характерные комбинации
comb_m = ml.apriori(bin_m, min_support = 0.05, use_colnames = True)
print("Характерных комбинаций - %d" % len(comb_m))
comb_m.sort_values(by = 'support', ascending=False)


# In[9]:


#Характерные комбинации по полу - женщины
woman = data[data.sex == 'f']
# группируем и бинуризируем
group_w = woman.groupby('user')['artist'].apply(';'.join)
bin_w = group_w.str.get_dummies(";").astype(bool)
# ищем характерные комбинации
comb_w = ml.apriori(bin_w, min_support = 0.05, use_colnames = True)
print("Характерных комбинаций - %d" % len(comb_w))
comb_w.sort_values(by = 'support', ascending=False)


# In[10]:


#Характерные комбинации по cтранам - Швеция
# группируем и бинуризируем
group_s = sweden.groupby('user')['artist'].apply(';'.join)
bin_s = group_s.str.get_dummies(";").astype(bool)
# ищем характерные комбинации
freq_s = ml.apriori(bin_s, min_support = 0.05, use_colnames = True)
print("Характерных комбинаций - %d" % len(freq_s))
freq_s.sort_values(by = 'support', ascending=False)


# In[11]:


#Характерные комбинации по cтранам - Белaрусь
# группируем и бинуризируем
group_b = belarus.groupby('user')['artist'].apply(';'.join)
bin_b = group_b.str.get_dummies(";").astype(bool)
# ищем характерные комбинации
comb_b = ml.apriori(bin_b, min_support = 0.05, use_colnames = True)
print("Характерных комбинаций - %d" % len(comb_b))
comb_b.sort_values(by = 'support', ascending=False)


# ## Сравнение алгоритмов apriori, fpgrowth, fpmax

# In[12]:


from mlxtend.frequent_patterns import fpgrowth

# Применение алгоритма FPGrowth, для бинаризованных данных по Швеции
# Алоритм FPG высчитывает поддержку имеющихся комбинаций.
# Поэтому получен тот же результат, но более оптимальным способом. 
fpgrowth_freq_swd = ml.fpgrowth(bin_s, min_support = 0.05, use_colnames = True)
print("Характерных комбинаций - %d" % len(fpgrowth_freq_swd))
fpgrowth_freq_swd


# In[13]:


# Применение алгоритма FPGrowth, для бинаризованных данных по Беларуси 
fpgrowth_freq_bel = ml.fpgrowth(bin_b, min_support = 0.05, use_colnames = True)
print("Характерных комбинаций - %d" % len(fpgrowth_freq_bel))
fpgrowth_freq_bel


# In[14]:


# Алгоритм FPMax концентрируется на частых максимальных комбинациях.
# Из-за этого, алгоритма FPMax нашел меньшее число комбинаций.

fpmax_swd = ml.fpmax(bin_s, min_support = 0.05, use_colnames = True)
print("Найдено %d характерных комбинаций" % len(fpmax_swd))
fpmax_swd


# ## Построение ассоциативных правил

# In[ ]:


#Confidence - это мера вероятности, с которой правило выполняется, основываясь на условии.

# Для anathema, rammstein и	depeche mode значение confidence = 1. То есть слушателе первых двух, обязательно послушают третью
# В этом есть смысл, так как deepche mode нечто среднее между anathema и rammstein. Более спокойный и обычный чем rammstein, 
# но более роковый, чем anathema.

rules = ml.association_rules(fpgrowth_freq_bel, metric = "confidence", min_threshold = 0.60, num_itemsets = 184)
rules


# In[ ]:


# Lift - это мера того, насколько сильно правило влияет на появление следствия, по сравнению с случаем, когда правило отсутствует.
# Для air и radiohead	lift = 7, то есть между группами есть сильная связь. Что не удивительно, так как из музыка сильна похожа друг на друга.

rules = ml.association_rules(fpgrowth_freq_bel, metric = "lift", min_threshold = 3.5, num_itemsets = 184)
rules


# In[ ]:


# Support указывает на частату того, как часто правило встречается в наборе данных.
# Например the prodigy и depeche mode имеют значение support примерно 0,09. То есть почти 10% слушателей prodigy, так же слушают depeche mode
rules = ml.association_rules(fpgrowth_freq_bel, metric = "support", min_threshold = 0.05, num_itemsets = 184)
rules


# In[ ]:


# Leverage - это мера того, насколько правило отличается от независимости между условием и следствием.
# Можно заметить, что у элеиетов небольшая, но все положительная ассоциация. Возможно слушателям radiohead, стоит предложить air.	

rules = ml.association_rules(fpgrowth_freq_bel, metric = "leverage", min_threshold = 0.045, num_itemsets = 184)
rules


# In[ ]:


# Conviction - мера степени зависимости между условием и следствием правила.

# Как можно заметить, the cinematic orchestra и depeche mode имеет значение больше 3,  
# поэтому lastfm стоит советовать случшателем the cinematic orchestra послушать depeche mode

rules = ml.association_rules(fpgrowth_freq_bel, metric = "conviction", min_threshold = 2, num_itemsets = 184)
rules


# In[ ]:


# Мера Чана максимизирует support и confidence одновременно, что позволяет идентифицировать наиболее сильные и значимые правила.
# Можно заметить что anathema и rammstein имеет довольно сильную свзять, хоть и обе группы представляют рок, но разные поджанры, поэтому их
# музыка сильно разлечается.

rules = ml.association_rules(fpgrowth_freq_bel, metric = "zhangs_metric", min_threshold = 0.8, num_itemsets = 184)
rules


# ## Поиск характерных комбинаций и ассоциативных правил длины не менее 5

# In[17]:


# Правила с длинной больше 5

apriori = ml.apriori(bin_b, min_support = 0.03, use_colnames = True)

rules = ml.association_rules(apriori, num_itemsets = 106)
rules[rules.apply(lambda x: len(x.antecedents)+len(x.consequents)>=5, axis=1)]


# In[18]:


# Короткие правила (длина меньше 3)
rules = ml.association_rules(apriori, num_itemsets = 106)
rules[rules.apply(lambda x: len(x.antecedents)+len(x.consequents)<=3, axis=1)]


# В коротких правилах lift, leverage и zhangs metric в среднем меньше, чем в длинных.  Confidence в длинных и коротких правилах почти всегда равен еденице.
# В support во многом совпадает, но есть привалирующие значения в длинных правилах.

# ## Тривиальные и нетривиальные правила

# Функциональна зависимоть тривиальна тогда и только тогда, когда правая часть символиче­ской записи данной зависимости является подмножеством (не обязательно собствен­ным подмножеством) левой части.
# Заметим, что в наших правилах множества не пересекаются, можно сказать, поэтому тревиальными правилами будут, если исполнители относятся к одному музыкальному жанру, как Jay-z -> Kanye west. 
# 
# Соответствено, нетривиальными правилами, в таком случае были бы исполнители из разных жанров, как Kanye west -> Arctic monkeys.

# ## Анализ зависимости оценок качества правил от поддержки и длины правил.

# In[25]:


apriori = ml.apriori(bin_b, min_support = 0.02, use_colnames = True)
print("Найдено %d характерных комбинаций" % len(apriori))
apriori

rules = ml.association_rules(apriori, num_itemsets = 106)
rules


# In[26]:


sns.scatterplot(data=rules, x="support", y="confidence", hue="lift", size="lift", sizes=(20, 200), alpha=0.6)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.show()


# In[28]:


rules['length'] = rules.apply(lambda x: len(x.antecedents)+len(x.consequents), axis=1)
plt.figure(figsize=(10, 6))
sns.boxplot(x='length', y='confidence', data=rules)
plt.xlabel("Length")
plt.ylabel("confidence")
plt.show()

