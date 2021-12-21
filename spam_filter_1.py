# Spam filter with Naive Bayes algorithm

import pandas as pd
import re

sms_spam = pd.read_csv('..\sms-spam-collection-dataset\spam.csv', encoding='latin-1')[['v1', 'v2']]
sms_spam.columns = ['Label', 'SMS']
print(sms_spam.shape)
print(sms_spam.head())

print(sms_spam ['Label'].value_counts(normalize = True))

# Randomize the dataset
data_randomized = sms_spam.sample(frac=1, random_state=1)

# Calculate index for split
training_test_index = round(len(data_randomized) * 0.8)

# Split into training and test sets
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

print(training_set.shape)
print(test_set.shape)

print(training_set['Label'].value_counts(normalize=True))

print(test_set['Label'].value_counts(normalize=True))

# Before cleaning
print(training_set.head(3))

# After cleaning
training_set['SMS'] = training_set['SMS'].str.replace(
   '\W', ' ') # Removes punctuation
training_set['SMS'] = training_set['SMS'].str.lower()
print(training_set.head(3))

training_set['SMS'] = training_set['SMS'].str.split()

vocabulary = []
for sms in training_set['SMS']:
   for word in sms:
      vocabulary.append(word)

vocabulary = list(set(vocabulary))

print(len(vocabulary))

word_counts_per_sms = {'convey': [1,0,0],
                       'my': [1,0,1],
                       'regards': [1,0,0],
                       'to': [1,1,0],
                       'him': [1,0,0],
                       'anyway': [0,1,0],
                       'many': [0,1,0],
                       'good': [0,1,0],
                       'evenings': [0,1,0],
                       'u': [0,1,0],
                       's': [0,1,0],
                       'sort': [0,0,1],
                       'code': [0,0,1],
                       'is': [0,0,3],
                       'and': [0,0,1],
                       'acc': [0,0,1],
                       'no': [0,0,1],
                       'the': [0,0,1],
                       'bank': [0,0,1],
                       'n': [0,0,1]
                      }

word_counts = pd.DataFrame(word_counts_per_sms)
print(word_counts.head())

word_counts_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}

for index, sms in enumerate(training_set['SMS']):
   for word in sms:
      word_counts_per_sms[word][index] += 1

word_counts = pd.DataFrame(word_counts_per_sms)
print(word_counts.head())

training_set_clean = pd.concat([training_set, word_counts], axis=1)
print(training_set_clean.head())

# Isolating spam and ham messages first
spam_messages = training_set_clean[training_set_clean['Label'] == 'spam']
ham_messages = training_set_clean[training_set_clean['Label'] == 'ham']

# P(Spam) and P(Ham)
p_spam = len(spam_messages) / len(training_set_clean)
p_ham = len(ham_messages) / len(training_set_clean)

# N_Spam
n_words_per_spam_message = spam_messages['SMS'].apply(len)
n_spam = n_words_per_spam_message.sum()

# N_Ham
n_words_per_ham_message = ham_messages['SMS'].apply(len)
n_ham = n_words_per_ham_message.sum()

# N_Vocabulary
n_vocabulary = len(vocabulary)

# Laplace smoothing
alpha = 1

# Initiate parameters
parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_ham = {unique_word:0 for unique_word in vocabulary}

# Calculate parameters
for word in vocabulary:
   n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
   p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
   parameters_spam[word] = p_word_given_spam

   n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
   p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
   parameters_ham[word] = p_word_given_ham

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

classify('Sounds good, Tom, then see u there')

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
print(test_set.head())

correct = 0
total = test_set.shape[0]

for row in test_set.iterrows():
   row = row[1]
   if row['Label'] == row['predicted']:
      correct += 1

print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)
