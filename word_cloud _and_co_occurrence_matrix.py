#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import re
import pandas as pd
import time
import numpy as np

from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from textblob import TextBlob, Word
from nltk.corpus import stopwords
from collections import OrderedDict
from operator import getitem
from nltk.corpus import wordnet as wn


# In[2]:


stop_words = stopwords.words('english')
stop_words.extend(['covid', 'virus', 'https', 'com', 'http', 'que', ' ', 'mierda', 'super', 'new', 'bbmp'])
shortword = re.compile(r'\W*\b\w{1,2}\b')

def lem(a):   
    sent = TextBlob(a)
    result = " ".join([w.lemmatize() for w in sent.words])
    return result   #lemmentization

def process_text(a):
    a = str(re.sub("\S*\d\S*", "", a).strip()) 
    a = a.lower()    
    a = re.sub(r"[^a-zA-Z0-9]"," ",a) #특수문자 제거
    a = re.sub("([^\x00-\x7F])+","",a) #영어이외제거    
    a = re.sub("(.)\\1{3,}", "\\1", a) #긴 반복문 제거
    a = re.sub('https?://\S+|www\.\S+', '', a) #하이퍼링크 제거
    a = shortword.sub(' ', a)    #짧은 단어 제거        
    a = lem(a)    
    tokenizer = TreebankWordTokenizer()
    # tokenize texts
    tokens = tokenizer.tokenize(a)  
    
    result = []
    for b in tokens:
        b = shortword.sub(' ', b)    #짧은 단어 제거  
        if b not in stop_words:  #stopwords 제거
            result.append(b)   
    
    return result

def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


# In[3]:


df = pd.read_csv('vaccination_all_tweets.csv')


# In[4]:


start_time = time.time()
df['process'] = ''
# df_2021.process = df_2021.Content.apply(process_text)
df.process = df.text.apply(process_text)

df['pos'] = df['process'].apply(lambda x: nltk.pos_tag(x)) # token에 pos tagging 

NN_list = [] # 명사 리스트 생성
for i in range(0, len(df['pos'])) :
    NN_words = []
    for word, pos in df['pos'].iloc[i]: 
        if 'NN' in pos: #pos에 nn 이 들어가 있으면 
            NN_words.append(word)
    NN_list.append(NN_words)    

df['nn_list'] = NN_list
end_time = time.time()
print(f'소요 시간 : {end_time - start_time}')


# In[5]:


mask_1 = (df['date'] >= '2020-12-12 00:00:00') & (df['date'] <= '2021-02-11 23:59:59')
df1 = df.loc[mask_1]
df1 = df1.reset_index()

mask_2 = (df['date'] >= '2021-02-12 00:00:00') & (df['date'] <= '2021-05-11 23:59:59')
df2 = df.loc[mask_2]
df2 = df2.reset_index()

mask_3 = (df['date'] >= '2021-05-12 00:00:00') & (df['date'] <= '2021-08-11 23:59:59')
df3 = df.loc[mask_3]
df3 = df3.reset_index()

mask_4 = (df['date'] >= '2021-08-12 00:00:00') & (df['date'] <= '2021-12-31 23:59:59')
df4 = df.loc[mask_4]
df4 = df4.reset_index()


# In[6]:


import copy
df1_cp = copy.deepcopy(df1)
df2_cp = copy.deepcopy(df2)
df3_cp = copy.deepcopy(df3)
df4_cp = copy.deepcopy(df4)


# In[26]:


df1


# In[7]:


word_list_1 = []
for i in range(len(df1)):
    data = df1['nn_list'][i]
    if len(df1['nn_list'][i])>0:
        word_list_1.extend(data)

word_list_2 = []
for i in range(len(df2)):
    data = df2['nn_list'][i]
    if len(df2['nn_list'][i])>0:
        word_list_2.extend(data)
        
word_list_3= []
for i in range(len(df3)):
    data = df3['nn_list'][i]
    if len(df3['nn_list'][i])>0:
        word_list_3.extend(data)
        
word_list_4 = []
for i in range(len(df4)):
    data = df4['nn_list'][i]
    if len(df4['nn_list'][i])>0:
        word_list_4.extend(data)


# In[8]:


print(len(df1), len(df2), len(df3), len(df4))


# # 워드클라우드 #

# In[9]:


from collections import Counter
frequency_1 = dict(Counter(word_list_1).most_common(100))
# frequency_1 = dict(Counter(word_list_1).most_common(300))
frequency_2 = dict(Counter(word_list_2).most_common(100))
# frequency_2 = dict(Counter(word_list_1).most_common(300))
frequency_3 = dict(Counter(word_list_3).most_common(100))
# frequency_3 = dict(Counter(word_list_1).most_common(300))
frequency_4 = dict(Counter(word_list_4).most_common(100))
# frequency_4 = dict(Counter(word_list_1).most_common(300))


# In[35]:


import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from konlpy.tag import Twitter
from PIL import Image


# In[38]:


# mask = np.array(Image.open('corona.png')) # 코로나 mask 적용
wordcloud = WordCloud(background_color='white',
                      width=1000, height=1000,
                      prefer_horizontal=True,
                      font_path = 'malgun',
                      random_state=124,
                      # mask=mask
                      ).generate_from_frequencies(frequency_1)

plt.figure(figsize=(20, 20))
plt.imshow(wordcloud)
plt.axis('off')


# In[133]:


wordcloud = WordCloud(background_color='white',
                      width=1000, height=1000,
                      prefer_horizontal=True,
                      font_path = 'malgun',
                      random_state=123
                      # mask=mask
                      ).generate_from_frequencies(frequency_2)

plt.figure(figsize=(20, 20))
plt.imshow(wordcloud)
plt.axis('off')


# In[138]:


wordcloud = WordCloud(background_color='white',
                      width=1000, height=1000,
                      prefer_horizontal=True,
                      font_path = 'malgun',
                      random_state=1234
                      # mask = mask
                      ).generate_from_frequencies(frequency_3)

plt.figure(figsize=(15, 15))
plt.imshow(wordcloud)
plt.axis('off')


# In[137]:


wordcloud = WordCloud(background_color='white',
                      width=1000, height=1000,
                      prefer_horizontal=True,
                      font_path = 'malgun',
                      random_state=3333
                      # mask = mask
                      ).generate_from_frequencies(frequency_4)

plt.figure(figsize=(20, 20))
plt.imshow(wordcloud)
plt.axis('off')


# # 단어 동시 출현 행렬 생성 #

# In[46]:


high_frequency_word = []
high_frequency_word2 = []
high_frequency_word3 = []
high_frequency_word4 = []


from collections import Counter

frequency_1 = dict(Counter(word_list_1).most_common(100))
for i in frequency_1:
    high_frequency_word.append(i)
    
frequency_2 = dict(Counter(word_list_2).most_common(100))
for i in frequency_2:
    high_frequency_word2.append(i)
    
frequency_3 = dict(Counter(word_list_3).most_common(100))
for i in frequency_3:
    high_frequency_word3.append(i)

frequency_4 = dict(Counter(word_list_4).most_common(100))
for i in frequency_4:
    high_frequency_word4.append(i)

print(len(high_frequency_word))
print(len(high_frequency_word2))
print(len(high_frequency_word3))
print(len(high_frequency_word4))


# In[47]:


for idx1, st1 in enumerate(df1_cp['nn_list']):
    for idx2, st2 in enumerate(df1_cp['nn_list'][idx1]):
        if df1_cp['nn_list'][idx1][idx2] not in high_frequency_word:
            del df1_cp['nn_list'][idx1][idx2]
        else:
            continue
            
for idx1, st1 in enumerate(df2_cp['nn_list']):
    for idx2, st2 in enumerate(df2_cp['nn_list'][idx1]):
        if df2_cp['nn_list'][idx1][idx2] not in high_frequency_word2:
            del df2_cp['nn_list'][idx1][idx2]
        else:
            continue
            
for idx1, st1 in enumerate(df3_cp['nn_list']):
    for idx2, st2 in enumerate(df3_cp['nn_list'][idx1]):
        if df3_cp['nn_list'][idx1][idx2] not in high_frequency_word3:
            del df3_cp['nn_list'][idx1][idx2]
        else:
            continue    
            
for idx1, st1 in enumerate(df4_cp['nn_list']):
    for idx2, st2 in enumerate(df4_cp['nn_list'][idx1]):
        if df4_cp['nn_list'][idx1][idx2] not in high_frequency_word4:
            del df4_cp['nn_list'][idx1][idx2]
        else:
            continue    


# In[13]:


def tf(t, d):
    temp = d.count(t)
#     if temp >= 1000:
    return temp

def generate_co_occurrence_matrix(data):
    
    node = []
    for i in data:
        node.extend(i)
        
    node = list(set(node))
    
    td_matrix = []
    for i in range(len(data)):
        td_matrix.append([])
        d = data[i]
        for j in range(len(node)):
            t = node[j]        
            td_matrix[-1].append(tf(t, d))
    
    tf_ = pd.DataFrame(td_matrix, columns = node)
    tf_=tf_.astype('float')
    column_name = tf_.columns.tolist()
    
    adjacency_matrix = np.dot(tf_.T,tf_)
    cooccurrence_matrix = pd.DataFrame(adjacency_matrix, index = column_name, columns = column_name)
    
    return cooccurrence_matrix


# In[32]:


print(len(df1), len(df2), len(df3), len(df4))


# In[33]:


period_1 = list(df1_cp['nn_list'])
period_2 = list(df2_cp['nn_list'])
period_3 = list(df3_cp['nn_list'])
period_4 = list(df4_cp['nn_list'])


# In[38]:


start_time = time.time()
period_1_matrix = generate_co_occurrence_matrix(period_1)
end_time = time.time()
print(f'소요 시간 : {end_time - start_time}')


# In[40]:


start_time = time.time()
period_2_matrix = generate_co_occurrence_matrix(period_2)
end_time = time.time()
print(f'소요 시간 : {end_time - start_time}')


# In[48]:


start_time = time.time()
period_3_matrix = generate_co_occurrence_matrix(period_3)
end_time = time.time()
print(f'소요 시간 : {end_time - start_time}')


# In[50]:


start_time = time.time()
period_4_matrix = generate_co_occurrence_matrix(period_4)
end_time = time.time()
print(f'소요 시간 : {end_time - start_time}')


# In[39]:


period_1_matrix


# In[41]:


period_2_matrix


# In[49]:


period_3_matrix


# In[51]:


period_4_matrix


# In[ ]:


period_1_matrix.to_csv("C:\\db\\2020-12.12~2021-02.11.csv")
period_2_matrix.to_csv("C:\\db\\2021-02.12~2021-05.11.csv")
period_3_matrix.to_csv("C:\\db\\2021-05.12~2020-08.11.csv")
period_4_matrix.to_csv("C:\\db\\2021-08.12~2020-11.24.csv")

