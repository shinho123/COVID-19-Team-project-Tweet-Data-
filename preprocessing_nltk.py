# -*- coding: utf-8 -*-
"""
Created on Thu May 19 13:22:14 2022

@author: young
"""
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from textblob import TextBlob, Word
from nltk.corpus import stopwords

word_list = []  #custom stop word
stop_words = stopwords.words('english') + stopwords.words('spanish') + word_list + stopwords.words('french')+ stopwords.words('portuguese') + stopwords.words('russian')
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

## 전처리 전 user_description 결측치 제거 필요

#명사 추출 
# input 데이터 변수명을 twitter로 설정, twitter['user_description']은 process_text 완료 되어있는 상태
twitter['pos'] = twitter['user_description'].apply(lambda x: nltk.pos_tag(x)) # token에 pos tagging 

NN_list = [] # 명사 리스트 생성
for i in range(0, len(twitter['pos'])) :
    NN_words = []
    for word, pos in twitter['pos'].iloc[i]: 
        if 'NN' in pos: #pos에 nn 이 들어가 있으면 
            NN_words.append(word)
    NN_list.append(NN_words)    

twitter['user_description'] = NN_list #명사만 추출

