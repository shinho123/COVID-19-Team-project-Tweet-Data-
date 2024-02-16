# Team-project
**데이터 분석 프로젝트**

**프로젝트 수행기간** : 2022.03 ~ 2022.05
**역할** : 데이터 전처리, 네트워크&텍스트 데이터 분석, 워드 클라우드 분석

# 분석 목적
 * 사람들이 COVID-19에 대해서 어떠한 생각을 가지고 있는지 파악하기 위해 트위터 데이터를 수집하고 Word colud 및 Network visualization 방법을 사용하여 분석을 수행함
   
# 데이터 수집
  * Kaggle의 'COVID-19 All Vaccines Tweet'의 데이터 셋 활용
  * 수집 기간 : 2020.12.12 ~ 2021.11.24
  * 주요 백신 : Sinopharm, Sinovac, Moderna, Oxford/Astra-Zeneca, Covaxin and Sputnik V vaccines
  * 주요 언어 : English

# 데이터 출력
```python
df = pd.read_csv('./data/vaccination_all_tweets.csv')
```
<img width="562" alt="image" src="https://github.com/shinho123/Team-project/assets/105840783/15a73b09-4743-4e95-aee7-07cedd933d1f">

* 본 데이터에서는 'user_description'속성(열)을 주로 활용하여 이후 분석을 수행함

# 데이터 전처리
 ## 전처리에 필요한 모듈 및 함수 정의
```python
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
```
* 텍스트 데이터 분석에 불필요한 데이터를 제거하는 작업을 수행함

* 이모지, 특수문자, 영어가 아닌 다른 나라의 언어, 하이퍼링크, 극단적으로 길거나 짧은 단어, 반복되는 문장...etc

# Visualization - word_cloud & co-occurrence matrix
 ## word cloud
<img width="491" alt="image" src="https://github.com/shinho123/Team-project/assets/105840783/aa9cc286-508c-4540-a3ae-bcf52ca45df4">

* **주요 키워드**
  * 2020.12.12 ~ 2021.02.11 : 'vaccine', 'pfizerbiontech', 'coronavirus', 'side', 'effect'
  * 2021.02.12 ~ 2021.05.11 : 'vaccine', 'moderna', 'covaxin', 'sputnikv', 'russia'

<img width="451" alt="image" src="https://github.com/shinho123/Team-project/assets/105840783/c5c42b38-ebb9-4da1-9eb8-6f333f522240">

* **주요 키워드**
  * 2021.05.12 ~ 2021.08.11 : 'vaccine', 'covaxin', 'moderna', 'hostpital', 'fee'
  * 2021.08.12 ~ 2021.11.24 : 'vaccine', 'covaxin', 'dose', 'fee', 'covishield'

# Network Visualization

```python
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
```

* gephi를 통해 네트워크 그래프를 시각화하기 위해서는 텍스트 데이터를 co-coccurrence matrix 형태로 변환하는 함수를 사용함

```python
period_1_matrix.to_csv("C:\\db\\2020-12.12~2021-02.11.csv")
period_2_matrix.to_csv("C:\\db\\2021-02.12~2021-05.11.csv")
period_3_matrix.to_csv("C:\\db\\2021-05.12~2021-08.11.csv")
period_4_matrix.to_csv("C:\\db\\2021-08.12~2021-11.24.csv")
```

* 코로나가 가장 활발한 시기인 2020년부터 2021까지 데이터를 4개로 분리하여 저장함
  * 2020.12.12 ~ 2021.02.11
  * 2021.02.12 ~ 2021.05.11
  * 2021.05.12 ~ 2021.08.11
  * 2021.08.12 ~ 2021.11.24

## gephi

* Gephi 옵션(모두 동일하게 적용)
  * Node Color - Modularity
  * Node Size - Weighted degree(Min size : 300, Max size : 700)
  * Node Label size - Weighted degree(Min size : 4, Max size : 7)
  * Force Atlas : default
  * Noverlap :
    * speed : 3.0
    * ratio : 1.0
    * margin : 5.0
  * Label Adjust : default

## 2020.12.12 ~ 2021.05.11

<img width="527" alt="image" src="https://github.com/shinho123/Team-project/assets/105840783/c872d5ba-22e2-4222-b9b4-64d5453f38d5">

* 2020.12.12 ~ 2020.02.11 / 2021.02.12 ~ 2021.05.11 : 본 시기는 코로나19가 가장 활발했던 시기로 많은 사람들이 모더나 백신을 의무적으로 접종해야 했던 시기로 키워드에도 "백신", "모더나"와 같은 의료 키워드가 대부분을 차지하고 있음

* 당시 **모더나 초기 접종**과 관련하여 **부작용이 많은 문제로 대두**되었기 때문에 "side", "effect"와 같은 키워드도 등장하는 것을 볼 수 있음

## 2021.05.12 ~ 2021.11.24

<img width="518" alt="image" src="https://github.com/shinho123/Team-project/assets/105840783/ea1b8ab6-c991-463d-aed0-efdc7cb72483">

* 2021.05.12 ~ 2021.08.11 / 2021.08.12 ~ 2021.11.24 : **시간이 지날수록 백신의 종류도 많아지고 부작용도 자연스럽게 줄어들면서** "side", "effect"와 같은 키워드가 독립적인 그룹을 형성하지는 않지만 **백신마다 비용의 차이가 커지면서** 코로나 백신을 의미하는 "covaxi"키워드가 "fee"와 같은 키워드와 같은 그룹에 속하여 나타나는 것을 볼 수 있음
