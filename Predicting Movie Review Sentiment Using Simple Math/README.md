
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
positive_count = count_text(positive_text)

#Generate Word Counts For Negative Text
negative_count = count_text(negative_text)

print('-----------------------------------------------')
print("Positive Text Sample: ",positive_text[:100])
print("Negative Text Sample: ",negative_text[:100])

print('-----------------------------------------------')
print("Word Count Of Positive Text: ",len(positive_count))
print("Word Count Of Positive Text: ",len(negative_count))

print('-----------------------------------------------')
```

    -----------------------------------------------
    Positive Text Sample:  bromwell high is a cartoon comedy. it ran at the same time as some other programs about school life,
    Negative Text Sample:  story of a man who has unnatural feelings for a pig. starts out with a opening scene that is a terri
    -----------------------------------------------
    Word Count Of Positive Text:  29094
    Word Count Of Positive Text:  29254
    -----------------------------------------------
    
