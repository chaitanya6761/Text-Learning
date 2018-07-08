
# Sentiment Analysis


```python
#Required Imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection  import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import random
```


```python
#Reading The Data Into A DataFrame
data_train = pd.read_csv('original_train_data.csv',delimiter="\t", quoting=3, names=['Sentiment','Text'])
data_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sentiment</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>The Da Vinci Code book is just awesome.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>this was the first clive cussler i've ever rea...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>i liked the Da Vinci Code a lot.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>i liked the Da Vinci Code a lot.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>I liked the Da Vinci Code but it ultimatly did...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Reading The Testing Data Into DataFrame 
data_test = pd.read_csv('original_test_data.csv', header=None, delimiter="\t", quoting=3, names=['Text'])
data_test.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>" I don't care what anyone says, I like Hillar...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>have an awesome time at purdue!..</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Yep, I'm still in London, which is pretty awes...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Have to say, I hate Paris Hilton's behavior bu...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>i will love the lakers.</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Dataset Summary
print('Length Of The Training Dataset: ',len(data_train))
print('Length Of The Testing Dataset: ',len(data_test))
print('Number Of Positive Reviews In Training Dataset: ', sum(data_train.Sentiment == 1))
print('Number Of Negative Reviews In Testing Dataset: ', sum(data_train.Sentiment == 0))
```

    Length Of The Training Dataset:  7086
    Length Of The Testing Dataset:  33052
    Number Of Positive Reviews In Training Dataset:  3995
    Number Of Negative Reviews In Testing Dataset:  3091
    


```python
#Lets Seperate The Data Into Features And Labels
features_train = data_train.Text
labels_train = data_train.Sentiment
```


```python
#Lets Create A MultiNomialNB Classifier 
model = make_pipeline(CountVectorizer(stop_words='english', max_features=100), MultinomialNB())

#Lets Fit The Data To The Model 
model.fit(features_train,labels_train)
```




    Pipeline(steps=[('countvectorizer', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=100, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words='english',
            strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
            tokenizer=None, vocabulary=None)), ('multinomialnb', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))])




```python
#Now Lets Use This Model To Predict Labels For Test Data
labels = model.predict(data_test.Text)
print(labels)
```

    [1 1 1 ..., 1 1 0]
    


```python
#Lets Print Out Some Random Elements Of Test Data To Verify Results.
for i in range(0,10):
    k = random.randint(0,101) 
    print(data_test.iloc[k][0],'------------->', labels[k],'\n')
```

    I want a ThinkPad or something. -------------> 1 
    
    harvard is for dumb people. -------------> 1 
    
    i will love the lakers. -------------> 1 
    
    and honda's are awesome:). -------------> 1 
    
    I like Honda... -------------> 1 
    
    seattle sucks anyways. -------------> 0 
    
    I like honda civics!!!!!!. -------------> 1 
    
    I think Angelina Jolie is so much more beautiful than Jennifer Anniston, who, by the way, is majorly OVERRATED. -------------> 1 
    
    i liked MIT though, esp their little info book( -------------> 1 
    
    Before I left Missouri, I thought London was going to be so good and cool and fun and a really great experience and I was really excited. -------------> 1 
    
    


```python
#Now Lets Use The Above Model To Predict The Sentiment On Some Random Data.
def predict(s, model=model):
    pred = model.predict([s])
    return pred[0]
```


```python
predict('I would Love To Vist London City')
```




    1




```python
predict('I Hate Loud Music')
```




    0


