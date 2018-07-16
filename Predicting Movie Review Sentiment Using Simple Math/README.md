
## Moview Review Sentiment Analysis


```python
#Required Imports
import os
import re
import string
from random import shuffle
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
```


```python
root = 'Data/'
folders = os.listdir(root)
movie_reviews = []
translator = str.maketrans('','', string.punctuation)
stemmer = SnowballStemmer("english")
stop_words = stopwords.words("english")

def parseOutText(all_text):
    '''
       Function To Remove Punctuations, Alpha-Numeric Words 
       And To Stem Those Words. 
    '''
    all_text = all_text.translate(translator).replace('\n', ' ')
    words_lst = all_text.split(' ')
    complete_sentence = ''
    for word in words_lst:
        word = stemmer.stem(word.strip())
        if word != '' and word.isalpha() and word not in stop_words:
            complete_sentence += (word + ' ')

    return complete_sentence.strip()


# This Code Is To Read Data From Different Folders And Combine Them Into A Single Array
for folder in folders:
    path = (root + folder +'/')
    files = os.listdir(path)
    for file in files:
        f = open(path + file)
        if folder == 'neg':
            movie_reviews.append([parseOutText(f.read()),-1])
        elif folder == 'pos':
            movie_reviews.append([parseOutText(f.read()),1])     
            
print('Total Number Data Points In Reviews: ',len(movie_reviews))            
```

    Total Number Data Points In Reviews:  2000
    


```python
#Lets Seperate The Data Into Training And Testing Sets.
shuffle(movie_reviews)

train_data = movie_reviews[:1800]
test_data = movie_reviews[1800:]

print('Total Number Of Data Points In Training Set: ',len(train_data))
print('Total Number Of Data Points In Testing Set: ',len(test_data))
```

    Total Number Of Data Points In Training Set:  1800
    Total Number Of Data Points In Testing Set:  200
    


```python
def get_text(reviews, score):
    '''
        This function is to collect data for a particular tone
    '''
    return "".join([r[0].lower() for r in reviews if r[1] == (score)])

def count_text(text):
    '''
        This function is to collect features from a particular tone.
    '''
    words = re.split("\s+", text)
    return Counter(words)
    
positive_text = get_text(train_data, 1)
negative_text = get_text(train_data, -1)

positive_counts = count_text(positive_text)
negative_counts = count_text(negative_text)
    
print('Positive text sample: ', positive_text[:320])
print('------------------------------------------------------------------------------------------------------------')
print('Negative text sample: ', negative_text[:320]) 
```

    Positive text sample:  devil take ask rhetor lull voic spoil titl charact onegin pronounc ohneggin wait death reliev lifetim rapaci behaviour martha fienn debut featur quit liter film poetri base epic russian poem alexand pushkin profound studi regret confus shame guilt first meet eugen onegin ralph act sister anoth brother magnus compos sco
    ------------------------------------------------------------------------------------------------------------
    Negative text sample:  yet anoth brainless teen flick one surpris drug sex star kati holm sarah polli couldnt look bore charact cardboard cutout everi clich teenag one thing need know realli hate movi everyth annoy hell act script plot end director fluke hit swinger could veri well direct bunch nonam actor watchab film big star go pretti muc
    


```python
def get_y_count(score):
    '''
        This function is to return the total number of reviews of a particular tone 
    '''
    return len([r for r in train_data if r[1] == score])

positive_review_count = get_y_count(1)
negative_review_count = get_y_count(-1)


#Prior probabilities
prob_positive = positive_review_count/len(train_data)
prob_negative = negative_review_count/len(train_data)

print('Total Number Of Positive Reviews In Training Set: ',positive_review_count)
print('Total Number Of Negative Reviews In Training Set: ',negative_review_count)
print('------------------------------------------------------------------')
print('Prior Probability Of Being Positive Review: ',prob_positive)
print('Prior Probability Of Being Negative Review: ',prob_negative)
```

    Total Number Of Positive Reviews In Training Set:  905
    Total Number Of Negative Reviews In Training Set:  895
    ------------------------------------------------------------------
    Prior Probability Of Being Positive Review:  0.5027777777777778
    Prior Probability Of Being Negative Review:  0.49722222222222223
    


```python
def make_class_prediction(text, counts, class_prob, class_count):
    
    prediction = 1
    text_counts = Counter(re.split("\s+", text))
    
    for word in text_counts:
        prediction *= text_counts.get(word) * ((counts.get(word,0)+1) / (sum(counts.values()) + class_count))
    return class_prob * prediction
print(train_data[0][0])
print("\nNegative prediction: {0}".format(make_class_prediction(train_data[0][0], negative_counts, prob_negative, negative_review_count)))
print("Positive prediction: {0}".format(make_class_prediction(train_data[0][0], positive_counts, prob_positive, positive_review_count)))    
```

    yet anoth brainless teen flick one surpris drug sex star kati holm sarah polli couldnt look bore charact cardboard cutout everi clich teenag one thing need know realli hate movi everyth annoy hell act script plot end director fluke hit swinger could veri well direct bunch nonam actor watchab film big star go pretti much drown project ani origin felt like watch dawson creek episod although film still would stay red despit cast surpris end sooo predict sinc male charact sudden outing closet consid surpris hollywood anymor go dawson creek varsiti blue shes go home watch someth els
    
    Negative prediction: 9.30633075462063e-283
    Positive prediction: 3.0618995093496967e-291
    


```python
def make_decision(text, make_class_prediction):
    negative_prediction = make_class_prediction(text, negative_counts, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, positive_counts, prob_positive, positive_review_count)    
    
    if negative_prediction < positive_prediction:
        return 1
    return -1

def calculate_Accuracy(predictions):
    actual_labels = [r[1] for r in test_data]
    count = 0

    for i in range(len(predictions)):
        if actual_labels[i] == (predictions[i]):
            count += 1
        
    print('Accuracy: ',count/len(actual_labels))
    
    
predictions = [make_decision(r[0], make_class_prediction) for r in test_data]
calculate_Accuracy(predictions)
```

    Accuracy:  0.54
    


```python
#The above performed steps can be reduced into few steps using count vectorizer and multinomial naive bayes.
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words = 'english')
train_features  = vectorizer.fit_transform([r[0] for r in train_data])
test_features = vectorizer.transform([r[0] for r in test_data])

classifier = MultinomialNB()
classifier.fit(train_features, [int(r[1]) for r in train_data])

predictions = classifier.predict(test_features)
calculate_Accuracy(predictions)

```

    Accuracy:  0.86
    
