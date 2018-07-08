

```python
#Necessary Imports
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
```


```python
#Reading The Data From File
data = pd.read_csv('SMSSpam', sep='\t', names=['Status','Message'])
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Status</th>
      <th>Message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Length Of The DataFrame
len(data)
```




    5572




```python
#No Of Spams
sum(data.Status == 'spam')
```




    747




```python
#Converting The Status Variables To 0 And 1 For More Fesability
data.loc[data.Status == 'spam', 'Status'] = 0
data.loc[data.Status == 'ham', 'Status'] = 1
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Status</th>
      <th>Message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Separating The DataFrame To Represent Features And Labels.
data_x = data['Message']
data_y = data['Status']

#Splitting The Data For Training And Testing
x_train, x_text, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=100)
```


```python
#Example - #Implementing The Count Vectorizer
cv = CountVectorizer()
lst = ['Hi, How are you, what are you doing', 'Hey, Whats Up, How is it Going', 'Count Vectorizer is cool', 
       'Text learning is great']
lst_cv = cv.fit_transform(lst)
array = lst_cv.toarray()
print(array)
```

    [[2 0 0 1 0 0 0 1 1 0 0 0 0 0 0 1 0 2]
     [0 0 0 0 1 0 1 0 1 1 1 0 0 1 0 0 1 0]
     [0 1 1 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0]
     [0 0 0 0 0 1 0 0 0 1 0 1 1 0 0 0 0 0]]
    


```python
cv.inverse_transform(array[0])
```




    [array(['are', 'doing', 'hi', 'how', 'what', 'you'], 
           dtype='<U10')]




```python
#List OF Features Identified By CountVectorizer
print(cv.get_feature_names())
```

    ['are', 'cool', 'count', 'doing', 'going', 'great', 'hey', 'hi', 'how', 'is', 'it', 'learning', 'text', 'up', 'vectorizer', 'what', 'whats', 'you']
    


```python
#Implementing The Above Count Vectorizer To SMSData
cv1 = CountVectorizer()
x_trainCV = cv1.fit_transform(x_train)
array = x_trainCV.toarray()
print(array)
```

    [[0 0 0 ..., 0 0 0]
     [0 0 0 ..., 0 0 0]
     [0 0 0 ..., 0 0 0]
     ..., 
     [0 0 0 ..., 0 0 0]
     [0 0 0 ..., 0 0 0]
     [0 0 0 ..., 0 0 0]]
    


```python
#Total Number Of Features Identified
print(len(cv1.get_feature_names()))
```

    7764
    


```python
x_train.iloc[0]
```




    'K da:)how many page you want?'




```python
#Applying Inverse transform
print(cv1.inverse_transform(array[0]))
```

    [array(['da', 'how', 'many', 'page', 'want', 'you'], 
          dtype='<U34')]
    


```python
#Example - Implementing The TdIdfVectorizer
cv2 = TfidfVectorizer(stop_words = 'english')
lst_cv2 = cv2.fit_transform(lst)
array = lst_cv2.toarray()
print(array)
```

    [[ 0.          0.          0.70710678  0.          0.          0.
       0.70710678  0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.57735027  0.          0.57735027
       0.          0.          0.          0.          0.57735027]
     [ 0.57735027  0.57735027  0.          0.          0.          0.          0.
       0.          0.          0.57735027  0.        ]
     [ 0.          0.          0.          0.          0.57735027  0.          0.
       0.57735027  0.57735027  0.          0.        ]]
    


```python
print(cv2.get_feature_names())
```

    ['cool', 'count', 'doing', 'going', 'great', 'hey', 'hi', 'learning', 'text', 'vectorizer', 'whats']
    


```python
print(cv2.inverse_transform(array[0]))
```

    [array(['doing', 'hi'], 
          dtype='<U10')]
    


```python
print(lst[0])
```

    Hi, How are you, what are you doing
    


```python
#Implementing The Above TfIdf Vectorizer To SMSData
cv3 = TfidfVectorizer(stop_words='english')
x_train_TfIdf = cv3.fit_transform(x_train)
array = x_train_TfIdf.toarray()
print(array)
```

    [[ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     ..., 
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]]
    


```python
#Total Number Of Features Identified.
print(len(cv3.get_feature_names()))
```

    7499
    


```python
x_train.iloc[0]
```




    'K da:)how many page you want?'




```python
#Applying The inverse Transform
print(cv3.inverse_transform(array[0]))
```

    [array(['da', 'page', 'want'], 
          dtype='<U34')]
    


```python
#Now Lets Apply The Above Created Vectorizer To Naive Bayes Classifier
y_train = y_train.astype('int')
x_test_Tfidf = cv3.transform(x_text)
cls = MultinomialNB()
cls.fit(x_train_TfIdf,y_train)
```




    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)




```python
#Now Lets Calculate The Accuracy Of The Classifier.
pred = cls.predict(x_test_Tfidf)
pred = np.array(pred)
y_test = np.array(y_test)

count = 0
length = len(y_test)
for i in range(length):
    if y_test[i] == pred[i]:
        count += 1
print('Accuracy: ',(count/length)) 
```

    Accuracy:  0.9739910313901345
    
