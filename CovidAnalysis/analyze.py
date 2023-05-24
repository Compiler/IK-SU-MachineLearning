


import random
def coin_toss():
  if random.random() <= 0.5:
    return 1
  return 0

def coin_trial():
  heads = 0
  for i in range(100):
      heads +=coin_toss()
  return heads
def simulate(n):
    trials = []
    for i in range(n):
        trials.append(coin_trial())
    return(sum(trials)/n)

simulate(1)

simulate(10)

simulate(100)

simulate(1000)

simulate(10000)



for i in range(1, 10):
  n = i * 10
  print(f'{n}: {simulate(n)}')


# return percentage chance of winning the game
TAILS = 0
HEADS = 1
def simulate_game(loss, payout, trials):
  successful_games = 0
  for i in range(trials):
    money = 0
    while(coin_toss() == TAILS): money -= loss
    money += payout
    success = money > 0
    successful_games += success
  return successful_games / trials


print("15$ payout:", simulate_game(10, 15, 1000000))
print("100$ payout:", simulate_game(10, 100, 1000000))

print("9$ payout:", simulate_game(10, 9, 1000000))
print("10$ payout:", simulate_game(10, 10, 1000000))
print("11$ payout:", simulate_game(10, 11, 1000000))


## Dice Roll
import random
def roll_dice(): return random.randint(1, 6)
def simulate_casino(pay_in, pay_out, trials = 1000):
  starting_money = 0
  for i in range(trials):
    dice1 = roll_dice() 
    dice2 = roll_dice()
    starting_money += pay_out if dice1 == 6 and dice2 == 6 else -pay_in
  return starting_money


payouts = []
for i in range(1000):
  payouts.append(simulate_casino(0.05, 100))

print(sum(payouts) / len(payouts)) #answer is ~2700






import pandas as pd

df= pd.read_csv("https://raw.githubusercontent.com/Compiler/IK-SU-MachineLearning/main/CovidAnalysis/data/covid.csv")

df.head()



#1
only_sick = df['actual_diagnostic'] == 1
tests_needed = df.loc[only_sick]
print(tests_needed.shape[0], 'tests needed')

#2 
pos_fast = df.loc[(df['covid_fast_kit'] == 1) & (df['actual_diagnostic'] == 1)]
pos_pcr = df.loc[(df['covid_rt_pcr'] == 1) & (df['actual_diagnostic'] == 1)]
print("P(sick | pos_fast) =", pos_fast.shape[0] / df.shape[0])
print("P(sick | pos_pcr) =", pos_pcr.shape[0] / df.shape[0])







#should be a 1.0 becuz of infinity

#this is prob of first 'letters' letters being hamlet
def calc_prob_monkey(letters):
  1/26**letters

print(calc_prob_monkey(130000))


#!wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
#!unzip smsspamcollection.zip


import pandas as pd

sms_spam = pd.read_csv('SMSSpamCollection', sep='\t',
header=None, names=['Label', 'SMS'])

print(sms_spam.shape)
sms_spam.head()

sms_spam[sms_spam['Label'] == 'spam']

sms_spam[sms_spam['Label'] == 'ham']




## your code here
p_spam = sms_spam[sms_spam['Label'] == 'spam'].shape[0] / sms_spam.shape[0]
print("P(Spam) =", p_spam)


## your code here
p_ham = sms_spam[sms_spam['Label'] == 'ham'].shape[0] / sms_spam.shape[0]
print("P(Ham) =", p_ham)


# Randomize the dataset
data_randomized = sms_spam.sample(frac=1, random_state=1)

# Calculate index for split
training_test_index = round(len(data_randomized) * 0.8)

# Split into training and test sets
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

print(training_set.shape)
print(test_set.shape)















# After cleaning
training_set['SMS'] = training_set['SMS'].str.replace(
   '\W', ' ') # Removes punctuation
training_set['SMS'] = training_set['SMS'].str.lower()
training_set.head(3)



vocabulary = {}
for list_of_words in training_set['SMS'].str.split().values:
  for word in list_of_words:
    if word in vocabulary: vocabulary[word] = vocabulary[word] + 1
    else: vocabulary[word] = 1


#expected answer if vocabulary is your list or
len(vocabulary)


word_counts_per_sms = {'secret': [1,1,1],
                       'prize': [1,0,1],
                       'claim': [1,0,1],
                       'now': [1,0,1],
                       'coming': [0,1,0],
                       'to': [0,1,0],
                       'my': [0,1,0],
                       'party': [0,1,0],
                       'winner': [0,0,1]
                      }

word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()



word_counts_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}
for ind,words in enumerate(training_set['SMS'].str.split()):
  for word in words:
    if word not in vocabulary:
      continue
    word_counts_per_sms[word][ind] = 1



word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()


training_set_clean = pd.concat([training_set, word_counts], axis=1)
training_set_clean.head(20)


# Isolating spam and ham messages first
spam_messages = training_set_clean.loc[training_set_clean['Label'] == 'spam']
ham_messages = training_set_clean.loc[training_set_clean['Label'] == 'ham']

# P(Spam) and P(Ham)
p_spam = spam_messages.shape[0] / training_set_clean.shape[0]
p_ham = ham_messages.shape[0] / training_set_clean.shape[0]

# N_Spam

n_spam = spam_messages.shape[0]

# N_Ham

n_ham = ham_messages.shape[0]
# N_Vocabulary
n_vocabulary = len(vocabulary)

# Laplace smoothing
alpha = 1


dfc = training_set_clean
dfc.loc[dfc['Label'] == 'spam']
from numpy.random import laplace
# Initiate parameters
parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_ham = {unique_word:0 for unique_word in vocabulary}
max_prob_word_spam = (0,'nill') 
max_prob_word_ham = (0,'nill') 


# Calculate parameters
for word in vocabulary:
   word_filt = (dfc['Label'] == 'spam')
   filtered = dfc.loc[word_filt, word]
   n_word_given_spam = sum(filtered)
   # your code here  - use the likelihood formula 𝑃(𝑤𝑖|𝑆𝑝𝑎𝑚)
   p_word_given_spam = (n_word_given_spam + alpha) / (spam_messages.shape[0] + (alpha * n_vocabulary)) 
   parameters_spam[word] = p_word_given_spam
   if(p_word_given_spam > max_prob_word_spam[0]):max_prob_word_spam = (p_word_given_spam, word)

   word_filt = (dfc['Label'] == 'ham')
   filtered = dfc.loc[word_filt, word]
   n_word_given_ham = sum(filtered)# your code here  # ham_messages already defined 𝑃(𝑤𝑖|𝑆𝑝𝑎𝑚)
   p_word_given_ham = (n_word_given_ham + alpha) / (ham_messages.shape[0] + (alpha * n_vocabulary)) 
   parameters_ham[word] = p_word_given_ham
   if(p_word_given_ham > max_prob_word_ham[0]):max_prob_word_ham = (p_word_given_ham, word)

training_set_clean.head(25)



word_cols = training_set_clean.loc[training_set_clean['Label'] == 'spam', 'yep']
print(word_cols == 0)


import re

def classify(message):
   '''
   message: a string
   '''

   message = re.sub('\W', ' ', message)
   message = message.lower().split()

   p_spam_given_message = p_spam
   p_ham_given_message = p_ham

   for word in message:
      if word in parameters_spam:
         p_spam_given_message *= parameters_spam[word]

      if word in parameters_ham: 
         p_ham_given_message *= parameters_ham[word]

   print('P(Spam|message):', p_spam_given_message)
   print('P(Ham|message):', p_ham_given_message)

   if p_ham_given_message > p_spam_given_message:
      print('Label: Ham')
   elif p_ham_given_message < p_spam_given_message:
      print('Label: Spam')
   else:
      print('Equal proabilities, have a human classify this!')


classify('WINNER!! This is the secret code to unlock the money: C3421.')

classify("Sounds good, Tom, then see u there")


def classify_test_set(message):
   '''
   message: a string
   '''

   message = re.sub('\W', ' ', message)
   message = message.lower().split()

   p_spam_given_message = p_spam
   p_ham_given_message = p_ham

   for word in message:
      if word in parameters_spam:
         p_spam_given_message *= parameters_spam[word]

      if word in parameters_ham:
         p_ham_given_message *= parameters_ham[word]

   if p_ham_given_message > p_spam_given_message:
      return 'ham'
   elif p_spam_given_message > p_ham_given_message:
      return 'spam'
   else:
      return 'needs human classification'


test_set['predicted'] = test_set['SMS'].apply(classify_test_set)
test_set.head()


correct = 0
total = test_set.shape[0]

for row in test_set.iterrows():
   row = row[1]
   if row['Label'] == row['predicted']:
      correct += 1

print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)