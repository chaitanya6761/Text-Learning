
### 1. Bayes Theorem Intro


```python
days = [["ran", "was tired"], ["ran", "was not tired"], 
        ["didn't run", "was tired"], ["ran", "was tired"], 
        ["didn't run", "was not tired"], ["ran", "was not tired"], ["ran", "was tired"]]

#lets say that we want to calculate the odds that someone was tired, given that they ran using naive bayes.
#This is P(A)
prob_tired = len([d for d in days if d[1] == 'was tired'])/len(days)

#This is P(B)
prob_ran = len([d for d in days if d[0] == 'ran'])/len(days)

#This is P(B|A)
prob_ran_given_tired = len([d for d in days if d[0] == "ran" and d[1] == "was tired"]) / len([d for d in days if d[1] == "was tired"])  

#Now we can calculate P(A|B)

prob_tired_given_ran = (prob_ran_given_tired * prob_tired) / prob_ran

print('The probability of being tired, given that you ran: ',prob_tired_given_ran)
```

    The probability of being tired, given that you ran:  0.6
    

### 2.  Naive Bayes


```python
# Here's our data, but with "woke up early" or "didn't wake up early" added.
days = [["ran", "was tired", "woke up early"], ["ran", "was not tired", "didn't wake up early"], 
        ["didn't run", "was tired", "woke up early"], ["ran", "was tired", "didn't wake up early"], 
        ["didn't run", "was tired", "woke up early"], ["ran", "was not tired", "didn't wake up early"], 
        ["ran", "was tired", "woke up early"]]

# We're trying to predict whether or not the person was tired on this day.
new_day = ["ran", "didn't wake up early"]

def calc_y_prob(y_label, days):
    return len([d for d in days if d[1] == y_label])/len(days)

def calc_ran_prob_given_y(ran_label, y_label, days):
    return len([d for d in days if d[0] == ran_label and d[1] == y_label])/len(days)

def calc_woke_early_prob_given_y(woke_label, y_label, days):
    return len([d for d in days if d[2] == woke_label and d[1] == y_label])/len(days)

denominator = len([d for d in days if d[0] == new_day[0] and d[2] == new_day[1]])/len(days)


#Lets plugin all the values and find out the label for given data point.
prob_tired = calc_y_prob('was tired', days) * calc_ran_prob_given_y(new_day[0], 'was tired', days) * calc_woke_early_prob_given_y(new_day[1], 'was tired', days) / denominator 

prob_not_tired = calc_y_prob('was not tired', days) * calc_ran_prob_given_y(new_day[0], 'was not tired', days) * calc_woke_early_prob_given_y(new_day[1], 'was not tired', days) / denominator 


#Now lets make a classifiaction deceision based on probabilities

print('Tired Probability: ',prob_tired)
print('Not Tired Probability: ',prob_not_tired)

classifiaction = 'was tired'
if prob_tired < prob_not_tired:
    classifiaction = 'was not tired'
    
print('Classificaton: ',classifiaction)    
```

    Tired Probability:  0.10204081632653061
    Not Tired Probability:  0.054421768707482984
    Classificaton:  was tired
    

### 3. Text Learning


```python
from collections import Counter
import csv
import re

#Read In The Training Data
with open('train.csv', 'r',encoding="utf8") as file:
    reviews = list(csv.reader(file))
    
def get_text(reviews, score):
    return ' '.join([r[0].lower() for r in reviews if r[1] == str(score)])
    
def count_text(text):
    words = re.split('\s+', text)
    return Counter(words)

    
#positive reviews:
positive_text = get_text(reviews, '1')
#negative reviews:
negative_text = get_text(reviews, '-1')

#Generate Word Counts For Positive Text
positive_counts = count_text(positive_text)

#Generate Word Counts For Negative Text
negative_counts = count_text(negative_text)

print('-----------------------------------------------')
print("Positive Text Sample: ",positive_text[:100])
print("Negative Text Sample: ",negative_text[:100])

print('-----------------------------------------------')
print("Features In Positive Text: ",len(positive_counts))
print("Features In Negative Text: ",len(negative_counts))

print('-----------------------------------------------')
```

    -----------------------------------------------
    Positive Text Sample:  bromwell high is a cartoon comedy. it ran at the same time as some other programs about school life,
    Negative Text Sample:  story of a man who has unnatural feelings for a pig. starts out with a opening scene that is a terri
    -----------------------------------------------
    Features In Positive Text:  29094
    Features In Negative Text:  29254
    -----------------------------------------------
    


```python
def get_y_count(score):
    return len([r for r in reviews if r[1] == str(score)])

positive_review_count = get_y_count(-1)
negative_review_count = get_y_count(1)

print('-----------------------------------------------')
print('Total Number Of Reviews: ',len(reviews))
print('Total Number Of Positive Reviews: ',positive_review_count)
print('Total Number Of Negative Reviews: ',positive_review_count)    
print('-----------------------------------------------')

prob_positive = positive_review_count/len(reviews)
prob_negative = negative_review_count/len(reviews)

print('Class Probabilty Of Being Positive Text: ',prob_positive)
print('Class Probabilty Of Being Negative Text: ',prob_negative)
print('-----------------------------------------------')
```

    -----------------------------------------------
    Total Number Of Reviews:  2000
    Total Number Of Positive Reviews:  1000
    Total Number Of Negative Reviews:  1000
    -----------------------------------------------
    Class Probabilty Of Being Positive Text:  0.5
    Class Probabilty Of Being Negative Text:  0.5
    -----------------------------------------------
    


```python
def make_class_prediction(text, counts, class_prob, class_count):
    prediction = 1
    text_counts = Counter(re.split("\s+", text))
    #print(text_counts)
    for word in text_counts:
        prediction *=  float(text_counts.get(word)) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))
        #print(prediction)
    return prediction * class_prob

print("Review: ",reviews[0])
print("Negative prediction: ",make_class_prediction(reviews[10][0], negative_counts, prob_negative, negative_review_count))
print("Positive prediction: ",make_class_prediction(reviews[100][0], positive_counts, prob_positive, positive_review_count))
```

    Review:  ["Story of a man who has unnatural feelings for a pig. Starts out with a opening scene that is a terrific example of absurd comedy. A formal orchestra audience is turned into an insane, violent mob by the crazy chantings of it's singers. Unfortunately it stays absurd the WHOLE time with no general narrative eventually making it just too off putting. Even those from the era should be turned off. The cryptic dialogue would make Shakespeare seem easy to a third grader. On a technical level it's better than you might think with some good cinematography by future great Vilmos Zsigmond. Future stars Sally Kirkland and Frederic Forrest can be seen briefly.", '-1']
    Negative prediction:  0.0
    Positive prediction:  0.0
    

### Predicting The TestSet


```python
def make_decision(text, make_class_prediction):
    neg_pred = make_class_prediction(text, negative_counts, prob_negative, negative_review_count)
    pos_pred = make_class_prediction(text, positive_counts, prob_positive, negative_review_count) 
    
    if neg_pred > pos_pred :
        return -1
    
    return 1

with open('test.csv', 'r', encoding="utf8") as file:
    test = list(csv.reader(file))
    
predictions = [make_decision(r[0], make_class_prediction) for r in test]     
```


```python
#Accuracy
from sklearn.metrics import accuracy_score
actual_labels = [r[1] for r in test]
count = 0

for i in range(len(predictions)):
    if actual_labels[i] == str(predictions[i]):
        count += 1
        
print('Accuracy: ',count/len(actual_labels))
```

    Accuracy:  0.559
    
