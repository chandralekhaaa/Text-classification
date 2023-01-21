!unzip '/content/MLHW1_DataFiles.zip' 
!unzip 'DataFiles/*.zip'

import string
import os
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import re
import math

def read_data_from_text_file(filename):
  with open(filename, "r", errors="ignore") as f:
    data = f.read()
  return data

enron1_training_data_ham = []
enron1_training_data_spam = []
enron1_test_data_ham = []
enron1_test_data_spam = []
enron4_training_data_ham = []
enron4_training_data_spam = []
enron4_test_data_ham = []
enron4_test_data_spam = []
hw1_training_data_ham = []
hw1_training_data_spam = []
hw1_test_data_ham = []
hw1_test_data_spam = []

#"ham" is encoded with 0 and "spam" is encoded with 1

current_working_directory = os.getcwd()
for directory in os.listdir(current_working_directory):
  if "enron1" in directory:
    if "train" in directory:
      for subdirectory1 in os.listdir(directory):      
        for subdirectory2 in os.listdir(directory+"/"+subdirectory1):
          if "ham" in subdirectory2:
            for file in os.listdir(directory+"/"+subdirectory1+"/"+subdirectory2):
              enron1_training_data_ham.append(read_data_from_text_file(f"{current_working_directory}/{directory}/{subdirectory1}/{subdirectory2}/{file}"))
          if "spam" in subdirectory2:
            for file in os.listdir(directory+"/"+subdirectory1+"/"+subdirectory2):
              enron1_training_data_spam.append(read_data_from_text_file(f"{current_working_directory}/{directory}/{subdirectory1}/{subdirectory2}/{file}"))
    elif "test" in directory:
      for subdirectory1 in os.listdir(directory):      
        for subdirectory2 in os.listdir(directory+"/"+subdirectory1):
          if "ham" in subdirectory2:
            for file in os.listdir(directory+"/"+subdirectory1+"/"+subdirectory2):
              enron1_test_data_ham.append(read_data_from_text_file(f"{current_working_directory}/{directory}/{subdirectory1}/{subdirectory2}/{file}"))
          if "spam" in subdirectory2:
            for file in os.listdir(directory+"/"+subdirectory1+"/"+subdirectory2):
              enron1_test_data_spam.append(read_data_from_text_file(f"{current_working_directory}/{directory}/{subdirectory1}/{subdirectory2}/{file}"))
  if "enron4" in directory:
    if "train" in directory:
      for subdirectory1 in os.listdir(directory):      
        for subdirectory2 in os.listdir(directory+"/"+subdirectory1):
          if "ham" in subdirectory2:
            for file in os.listdir(directory+"/"+subdirectory1+"/"+subdirectory2):
              enron4_training_data_ham.append(read_data_from_text_file(f"{current_working_directory}/{directory}/{subdirectory1}/{subdirectory2}/{file}"))
          if "spam" in subdirectory2:
            for file in os.listdir(directory+"/"+subdirectory1+"/"+subdirectory2):
              enron4_training_data_spam.append(read_data_from_text_file(f"{current_working_directory}/{directory}/{subdirectory1}/{subdirectory2}/{file}"))
    elif "test" in directory:
      for subdirectory1 in os.listdir(directory):      
        for subdirectory2 in os.listdir(directory+"/"+subdirectory1):
          if "ham" in subdirectory2:
            for file in os.listdir(directory+"/"+subdirectory1+"/"+subdirectory2):
              enron4_test_data_spam.append(read_data_from_text_file(f"{current_working_directory}/{directory}/{subdirectory1}/{subdirectory2}/{file}"))
          if "spam" in subdirectory2:
            for file in os.listdir(directory+"/"+subdirectory1+"/"+subdirectory2):
              enron4_test_data_ham.append(read_data_from_text_file(f"{current_working_directory}/{directory}/{subdirectory1}/{subdirectory2}/{file}"))
  if "hw1" in directory:
    if "train" in directory:
      for subdirectory in os.listdir(directory): 
          if "ham" in subdirectory:
            for file in os.listdir(directory+"/"+subdirectory):
              hw1_training_data_ham.append(read_data_from_text_file(f"{current_working_directory}/{directory}/{subdirectory}/{file}"))
          if "spam" in subdirectory:
            for file in os.listdir(directory+"/"+subdirectory):
              hw1_training_data_spam.append(read_data_from_text_file(f"{current_working_directory}/{directory}/{subdirectory}/{file}"))
    elif "test" in directory:
      for subdirectory in os.listdir(directory):      
          if "ham" in subdirectory:
            for file in os.listdir(directory+"/"+subdirectory):
              hw1_test_data_ham.append(read_data_from_text_file(f"{current_working_directory}/{directory}/{subdirectory}/{file}"))
          if "spam" in subdirectory:
            for file in os.listdir(directory+"/"+subdirectory):
              hw1_test_data_spam.append(read_data_from_text_file(f"{current_working_directory}/{directory}/{subdirectory}/{file}"))

#Preprocessing

def preprocess_data(data):
  preprocessed_data = []
  for doc in data:
    preprocessed_sample = doc.lower() #converting into lowercase 
    preprocessed_sample = re.sub(r'[^\w\s]', '', preprocessed_sample)
    #tokenizing the data
    preprocessed_sample =  nltk.word_tokenize(preprocessed_sample)
    #removing stop words
    stop_words = stopwords.words('english')
    for token in preprocessed_sample:
      if token in stop_words:
        preprocessed_sample.remove(token)
    preprocessed_data.append(preprocessed_sample)
  return preprocessed_data

def build_vocabulary(data):
  vocabulary = {}
  count = 0
  for doc in data:
    for token in doc:
      if token not in vocabulary:        
        vocabulary[token] = count
        count+=1
  return vocabulary

#training data
enron1_training_data_ham = preprocess_data(enron1_training_data_ham)
enron1_training_vocabulary_ham = build_vocabulary(enron1_training_data_ham)

enron1_training_data_spam = preprocess_data(enron1_training_data_spam)
enron1_training_vocabulary_spam = build_vocabulary(enron1_training_data_spam)

enron1_training_data = enron1_training_data_ham + enron1_training_data_spam
enron1_training_vocabulary = build_vocabulary(enron1_training_data)

enron4_training_data_ham = preprocess_data(enron4_training_data_ham)
enron4_training_vocabulary_ham = build_vocabulary(enron4_training_data_ham)

enron4_training_data_spam = preprocess_data(enron4_training_data_spam)
enron4_training_vocabulary_spam = build_vocabulary(enron4_training_data_spam)

enron4_training_data = enron4_training_data_ham + enron4_training_data_spam
enron4_training_vocabulary = build_vocabulary(enron4_training_data)

hw1_training_data_ham = preprocess_data(hw1_training_data_ham)
hw1_training_vocabulary_ham = build_vocabulary(hw1_training_data_ham)

hw1_training_data_spam = preprocess_data(hw1_training_data_spam)
hw1_training_vocabulary_spam = build_vocabulary(hw1_training_data_spam)

hw1_training_data = hw1_training_data_ham + hw1_training_data_spam
hw1_training_vocabulary = build_vocabulary(hw1_training_data)

#test data
enron1_test_data_ham = preprocess_data(enron1_test_data_ham)
enron1_test_data_spam = preprocess_data(enron1_test_data_spam)

enron4_test_data_ham = preprocess_data(enron4_test_data_ham)
enron4_test_data_spam = preprocess_data(enron4_test_data_spam)

hw1_test_data_ham = preprocess_data(hw1_test_data_ham)
hw1_test_data_spam = preprocess_data(hw1_test_data_spam)

#Bag of words
bow_enron1_training_data_ham = []
bow_enron1_training_data_spam = []
bow_enron1_training_data = []

bow_enron4_training_data_ham = []
bow_enron4_training_data_spam = []
bow_enron4_training_data = []

bow_hw1_training_data_ham = []
bow_hw1_training_data_spam = []
bow_hw1_training_data = []

def bag_of_words(data, vocabulary):
  bag_of_words = []
  for doc in data:
    encoded_doc = [0]*len(vocabulary)
    for token in doc:
      if token in vocabulary:
        encoded_doc[vocabulary[token]]+=1
    bag_of_words.append(encoded_doc)
  return bag_of_words

#training data
bow_enron1_training_data_ham = bag_of_words(enron1_training_data_ham, enron1_training_vocabulary_ham)
bow_enron1_training_data_spam = bag_of_words(enron1_training_data_spam, enron1_training_vocabulary_spam)
bow_enron1_training_data= bag_of_words(enron1_training_data, enron1_training_vocabulary)
bow_enron4_training_data_ham = bag_of_words(enron4_training_data_ham, enron4_training_vocabulary_ham)
bow_enron4_training_data_spam = bag_of_words(enron4_training_data_spam, enron4_training_vocabulary_spam)
bow_enron4_training_data= bag_of_words(enron4_training_data, enron4_training_vocabulary)
bow_hw1_training_data_ham = bag_of_words(hw1_training_data_ham, hw1_training_vocabulary_ham)
bow_hw1_training_data_spam = bag_of_words(hw1_training_data_spam, hw1_training_vocabulary_spam)
bow_hw1_training_data= bag_of_words(hw1_training_data, hw1_training_vocabulary)

#Bernoulli model
bernoulli_enron1_training_data_ham = []
bernoulli_enron1_training_data_spam = []
bernoulli_enron1_training_data = []
bernoulli_enron4_training_data_ham = []
bernoulli_enron4_training_data_spam = []
bernoulli_enron4_training_data = []
bernoulli_hw1_training_data_ham = []
bernoulli_hw1_training_data_spam = []
bernoulli_hw1_training_data = []

def bernoulli_model(data, vocabulary):
  bernoulli_model = []
  for doc in data:
    encoded_doc = [0]*len(vocabulary)
    for token in doc:
      if token in vocabulary:
        encoded_doc[vocabulary[token]]=1
    bernoulli_model.append(encoded_doc)
  return bernoulli_model

#training data
bernoulli_enron1_training_data_ham = bernoulli_model(enron1_training_data_ham, enron1_training_vocabulary_ham)
bernoulli_enron1_training_data_spam = bernoulli_model(enron1_training_data_spam, enron1_training_vocabulary_spam)
bernoulli_enron1_training_data = bernoulli_model(enron1_training_data, enron1_training_vocabulary)
bernoulli_enron4_training_data_ham = bernoulli_model(enron4_training_data_ham, enron4_training_vocabulary_ham)
bernoulli_enron4_training_data_spam = bernoulli_model(enron4_training_data_spam, enron4_training_vocabulary_spam)
bernoulli_enron4_training_data = bernoulli_model(enron4_training_data, enron4_training_vocabulary)
bernoulli_hw1_training_data_ham = bernoulli_model(hw1_training_data_ham, hw1_training_vocabulary_ham)
bernoulli_hw1_training_data_spam = bernoulli_model(hw1_training_data_spam, hw1_training_vocabulary_spam)
bernoulli_hw1_training_data = bernoulli_model(hw1_training_data, hw1_training_vocabulary)

def evaluate_metrics(actual, prediction):
  total = len(actual)
  tp = 0
  tn = 0
  fp =0
  fn =0
  for i in range(0, len(actual)):
    if actual[i] == 1 and prediction[i] == 1:
      tp +=1
    elif actual[i] == 0 and prediction[i] == 1:
      fp+=1
    elif actual[i] == 1 and prediction[i] == 0:
      fn+=1
    else:
      tn+=1
  accuracy = (tp + tn)/ total
  if tp+fp != 0:
    precision = tp/(tp+fp)
  else:
    precision = 0
  if tp+fn != 0:
    recall = tp/(tp+fn)
  else:
    recall = 0  
  if (precision+recall) != 0:
    f1_score = (2*precision*recall)/(precision+recall)
  else:
    f1_score = 0 
  return { "accuracy": accuracy, "recall" : recall,  "f1_score" : f1_score, "precision" : precision}  

evaluation_metrics = {}
evaluation_metrics["enron1"] = {}
evaluation_metrics["enron4"] = {}
evaluation_metrics["hw1"] = {}
#Dictionary of each dataset will consist of accuracy, , precision, recall, f1_score for

classes = [0, 1]
classes_dictionary = {1:"ham", 0:"spam"}
vocabulary = {"enron1" : { "total": enron1_training_vocabulary, "ham": enron1_training_vocabulary_ham, "spam": enron1_training_vocabulary_spam},
              "enron4" : { "total": enron4_training_vocabulary, "ham": enron4_training_vocabulary_ham, "spam": enron4_training_vocabulary_spam},
              "hw1" : { "total": hw1_training_vocabulary, "ham": hw1_training_vocabulary_ham, "spam": hw1_training_vocabulary_spam}}

def get_vocabulary(dataset, cls):
  return vocabulary[dataset][cls]

def get_token_counts(data, vocabulary, token):
  token_count = 0
  for doc in data:
    token_count += doc[vocabulary[token]]
  return token_count

#Multinomial Naive Bayes Algorithm using Bag of words
classes = [0, 1]
classes_dictionary = {1:"ham", 0:"spam"}
bow_training_data = {"enron1" : {"total": bow_enron1_training_data, "ham": bow_enron1_training_data_ham, "spam": bow_enron1_training_data_spam},
              "enron4" : {"total": bow_enron4_training_data,"ham": bow_enron4_training_data_ham, "spam": bow_enron4_training_data_spam},
              "hw1" : {"total": bow_hw1_training_data,"ham": bow_hw1_training_data_ham, "spam": bow_hw1_training_data_spam}}

def bow_get_data(dataset, cls):
  return bow_training_data[dataset][cls]

def train_multinomial_naive_bayes(classes, dataset):
  data = bow_get_data(dataset, "total")
  total_vocabulary = get_vocabulary(dataset, "total")
  total_number_of_docs = len(data)
  prior_probability = {}
  conditional_probability = []
  for i in range(0, len(total_vocabulary)):
    conditional_probability.append([0]*len(classes))
  for cls in classes:
    class_data = bow_get_data(dataset, classes_dictionary[cls])
    number_of_docs_in_class = len(class_data)
    prior_probability[cls] = number_of_docs_in_class/total_number_of_docs
    class_vocabulary= get_vocabulary(dataset, classes_dictionary[cls])
    for token in class_vocabulary.keys():
      conditional_probability[total_vocabulary[token]][cls] = (get_token_counts(class_data, class_vocabulary, token) + 1) / (get_token_counts(data, total_vocabulary, token) + 1)
  return [prior_probability, conditional_probability] 

def apply_multinomial_naive_bayes(data, vocabulary, classes, prior_probability, conditional_probability):
  scores = []
  for i in range(0, len(data)):
    scores.append([0]*len(classes))
  for i in range(len(data)):
    for cls in classes:
      scores[i][cls] = math.log(prior_probability[cls], 10)
      for token in data[i]:
        if token in vocabulary and conditional_probability[vocabulary[token]][cls] != 0:
          scores[i][cls] += math.log(conditional_probability[vocabulary[token]][cls], 10)
  result = []
  for score in scores:
    curr_max = 0
    for i in range(0, len(score)):
      if score[i] > score[curr_max]:
        curr_max = i
    result.append(curr_max)
  return result

#Training on data
mnb_enron1 = train_multinomial_naive_bayes(classes, "enron1")
mnb_enron1_prior_probability = mnb_enron1[0]
mnb_enron1_conditional_probability = mnb_enron1[1]

mnb_enron4 = train_multinomial_naive_bayes(classes, "enron4")
mnb_enron4_prior_probability = mnb_enron4[0]
mnb_enron4_conditional_probability = mnb_enron4[1]

mnb_hw1 = train_multinomial_naive_bayes(classes, "hw1")
mnb_hw1_prior_probability = mnb_hw1[0]
mnb_hw1_conditional_probability = mnb_hw1[1]

#Testing on test set
mnb_enron1_ham_result = apply_multinomial_naive_bayes(enron1_test_data_ham, get_vocabulary("enron1", "total"), classes, mnb_enron1_prior_probability, mnb_enron1_conditional_probability)
mnb_enron1_spam_result = apply_multinomial_naive_bayes(enron1_test_data_spam, get_vocabulary("enron1", "total"), classes, mnb_enron1_prior_probability, mnb_enron1_conditional_probability)

mnb_enron4_ham_result = apply_multinomial_naive_bayes(enron4_test_data_ham, get_vocabulary("enron4", "total"), classes, mnb_enron4_prior_probability, mnb_enron4_conditional_probability)
mnb_enron4_spam_result = apply_multinomial_naive_bayes(enron4_test_data_spam, get_vocabulary("enron4", "total"), classes, mnb_enron4_prior_probability, mnb_enron4_conditional_probability)

mnb_hw1_ham_result = apply_multinomial_naive_bayes(hw1_test_data_ham, get_vocabulary("hw1", "total"), classes, mnb_hw1_prior_probability, mnb_hw1_conditional_probability)
mnb_hw1_spam_result = apply_multinomial_naive_bayes(hw1_test_data_spam, get_vocabulary("hw1", "total"), classes, mnb_hw1_prior_probability, mnb_hw1_conditional_probability)

#Evaluation metrics
mnb_enron1_prediction = mnb_enron1_ham_result + mnb_enron1_spam_result
mnb_enron1_actual = [1]*len(mnb_enron1_ham_result) + [0]*len(mnb_enron1_spam_result)
mnb_enron1_evaluation_metrics = evaluate_metrics(mnb_enron1_actual, mnb_enron1_prediction)
evaluation_metrics["enron1"]["multinomial_naive_bayes"] = mnb_enron1_evaluation_metrics

mnb_enron4_prediction = mnb_enron4_ham_result + mnb_enron4_spam_result
mnb_enron4_actual = [1]*len(mnb_enron4_ham_result) + [0]*len(mnb_enron4_spam_result)
mnb_enron4_evaluation_metrics = evaluate_metrics(mnb_enron4_actual, mnb_enron4_prediction)
evaluation_metrics["enron4"]["multinomial_naive_bayes"] = mnb_enron4_evaluation_metrics

mnb_hw1_prediction = mnb_hw1_ham_result + mnb_hw1_spam_result
mnb_hw1_actual = [1]*len(mnb_hw1_ham_result) + [0]*len(mnb_hw1_spam_result)
mnb_hw1_evaluation_metrics = evaluate_metrics(mnb_hw1_actual, mnb_hw1_prediction)
evaluation_metrics["hw1"]["multinomial_naive_bayes"] = mnb_hw1_evaluation_metrics

#Discrete Naive Bayes Algorithm using Bernoulli model
classes = [0, 1]
classes_dictionary = {1:"ham", 0:"spam"}
bernoulli_training_data = {"enron1" : {"total": bernoulli_enron1_training_data, "ham": bernoulli_enron1_training_data_ham, "spam": bernoulli_enron1_training_data_spam},
              "enron4" : {"total": bernoulli_enron4_training_data,"ham": bernoulli_enron4_training_data_ham, "spam": bernoulli_enron4_training_data_spam},
              "hw1" : {"total": bernoulli_hw1_training_data,"ham": bernoulli_hw1_training_data_ham, "spam": bernoulli_hw1_training_data_spam}}

def bernoulli_get_data(dataset, cls):
  return bernoulli_training_data[dataset][cls]

def train_discrete_naive_bayes(classes, dataset):
  data = bernoulli_get_data(dataset, "total")
  total_vocabulary = get_vocabulary(dataset, "total")
  total_number_of_docs = len(data)
  prior_probability = {}
  conditional_probability = []
  for i in range(0, len(total_vocabulary)):
    conditional_probability.append([0]*len(classes))
  for cls in classes:
    class_data = bernoulli_get_data(dataset, classes_dictionary[cls])
    number_of_docs_in_class = len(class_data)
    prior_probability[cls] = number_of_docs_in_class/total_number_of_docs
    class_vocabulary= get_vocabulary(dataset, classes_dictionary[cls])
    for token in class_vocabulary.keys():
      conditional_probability[total_vocabulary[token]][cls] = (get_token_counts(class_data, class_vocabulary, token) + 1) / (get_token_counts(data, total_vocabulary, token) + 1)
  return [prior_probability, conditional_probability] 

def apply_discrete_naive_bayes(data, vocabulary, classes, prior_probability, conditional_probability):
  scores = []
  for i in range(len(data)):
    for cls in classes:
      score = []
      score.append(math.log(prior_probability[cls], 10))
      for token in data[i]:
        if token in vocabulary and conditional_probability[vocabulary[token]][cls] != 0:
          score[-1] += math.log(conditional_probability[vocabulary[token]][cls], 10)
      scores.append(score)
  result = []
  for score in scores:
    curr_max = 0
    for i in range(0, len(score)):
      if score[i] > score[curr_max]:
        curr_max = i
    result.append(curr_max)  
  return result

#Training on data
dnb_enron1 = train_discrete_naive_bayes(classes, "enron1")
dnb_enron1_prior_probability = dnb_enron1[0]
dnb_enron1_conditional_probability = dnb_enron1[1]

dnb_enron4 = train_discrete_naive_bayes(classes, "enron4")
dnb_enron4_prior_probability = dnb_enron4[0]
dnb_enron4_conditional_probability = dnb_enron4[1]

dnb_hw1 = train_discrete_naive_bayes(classes, "hw1")
dnb_hw1_prior_probability = dnb_hw1[0]
dnb_hw1_conditional_probability = dnb_hw1[1]

#Testing on test set
dnb_enron1_ham_result = apply_discrete_naive_bayes(enron1_test_data_ham, get_vocabulary("enron1", "total"), classes, dnb_enron1_prior_probability, dnb_enron1_conditional_probability)
dnb_enron1_spam_result = apply_discrete_naive_bayes(enron1_test_data_spam, get_vocabulary("enron1", "total"), classes, dnb_enron1_prior_probability, dnb_enron1_conditional_probability)

dnb_enron4_ham_result = apply_discrete_naive_bayes(enron4_test_data_ham, get_vocabulary("enron4", "total"), classes, dnb_enron4_prior_probability, dnb_enron4_conditional_probability)
dnb_enron4_spam_result = apply_discrete_naive_bayes(enron4_test_data_spam, get_vocabulary("enron4", "total"), classes, dnb_enron4_prior_probability, dnb_enron4_conditional_probability)

dnb_hw1_ham_result = apply_discrete_naive_bayes(hw1_test_data_ham, get_vocabulary("hw1", "total"), classes, dnb_hw1_prior_probability, dnb_hw1_conditional_probability)
dnb_hw1_spam_result = apply_discrete_naive_bayes(hw1_test_data_spam, get_vocabulary("hw1", "total"), classes, dnb_hw1_prior_probability, dnb_hw1_conditional_probability)

#Evaluation metrics
dnb_enron1_prediction = dnb_enron1_ham_result + dnb_enron1_spam_result
dnb_enron1_actual = [1]*len(dnb_enron1_ham_result) + [0]*len(dnb_enron1_spam_result)
dnb_enron1_evaluation_metrics = evaluate_metrics(dnb_enron1_actual, dnb_enron1_prediction)
evaluation_metrics["enron1"]["discrete_naive_bayes"] = dnb_enron1_evaluation_metrics

dnb_enron4_prediction = dnb_enron4_ham_result + dnb_enron4_spam_result
dnb_enron4_actual = [1]*len(dnb_enron4_ham_result) + [0]*len(dnb_enron4_spam_result)
dnb_enron4_evaluation_metrics = evaluate_metrics(dnb_enron4_actual, dnb_enron4_prediction)
evaluation_metrics["enron4"]["discrete_naive_bayes"] = dnb_enron4_evaluation_metrics

dnb_hw1_prediction = dnb_hw1_ham_result + dnb_hw1_spam_result
dnb_hw1_actual = [1]*len(dnb_hw1_ham_result) + [0]*len(dnb_hw1_spam_result)
dnb_hw1_evaluation_metrics = evaluate_metrics(dnb_hw1_actual, dnb_hw1_prediction)
evaluation_metrics["hw1"]["discrete_naive_bayes"] = dnb_hw1_evaluation_metrics

#Data for Logistic Regression and SGDClassifier
#Bag of words
bow_X_enron1 = bow_training_data["enron1"]["total"]
print(bow_training_data)
bow_Y_enron1 = [1]*len(bow_training_data["enron1"]["ham"]) + [0]*len(bow_training_data["enron1"]["spam"])
bow_X_enron4 = bow_training_data["enron4"]["total"]
bow_Y_enron4 = [1]*len(bow_training_data["enron4"]["ham"]) + [0]*len(bow_training_data["enron4"]["spam"])
bow_X_hw1 = bow_training_data["hw1"]["total"]
bow_Y_hw1 = [1]*len(bow_training_data["hw1"]["ham"]) + [0]*len(bow_training_data["hw1"]["spam"])

#Bernoulli model
bernoulli_X_enron1 = bernoulli_training_data["enron1"]["total"]
bernoulli_Y_enron1 = [1]*len(bernoulli_training_data["enron1"]["ham"]) + [0]*len(bernoulli_training_data["enron1"]["spam"])
bernoulli_X_enron4 = bernoulli_training_data["enron4"]["total"]
bernoulli_Y_enron4 = [1]*len(bernoulli_training_data["enron4"]["ham"]) + [0]*len(bernoulli_training_data["enron4"]["spam"])
bernoulli_X_hw1 = bernoulli_training_data["hw1"]["total"]
bernoulli_Y_hw1 = [1]*len(bernoulli_training_data["hw1"]["ham"]) + [0]*len(bernoulli_training_data["hw1"]["spam"])

#MCAP Logistic Regression algorithm with L2 regularization
#Try different values of Lambda
#Split training data into traing and validation - 70% and 30% - Use validation data to find lambda
#Learn using full training set - Gradient descent

#Logistic regression implementation - gradient descent
#Finding lambda using validation
#testing

import numpy as np

def sigmoid_function(X):
  return 1/(1 + np.exp(-X))

def train_logistic_regression(X, Y, lambda_value, learning_rate, iterations):
  m = len(X)
  n = len(X[0])
  weights = np.array([0] * n)
  constant = 0
  for iteration in range(iterations):
    logistic_regression_result = np.dot(X, weights.T) + constant
    logistic_regression_result = sigmoid_function(logistic_regression_result)
    
    #gradient descent
    d_weights = (1/m)*np.dot(X.T, logistic_regression_result-Y)
    d_constant = (1/m)*np.sum(logistic_regression_result - Y)

    weights = weights -learning_rate*lambda_value*d_weights.T
    constant = constant - learning_rate*lambda_value*d_constant.T
  return [weights, constant]

def test_logistic_regression(test_data, weights, constant):
  return np.round(np.dot(test_data, weights.T) + constant)

def train_lambda(training_data, training_data_target, validation_data, validation_data_target):
  learning_rate = 0.01
  accuracy = 0.0
  final_lambda_parameter = 2
  length_validation_data = len(validation_data)
  for lambda_value in range(1, 10, 2):
    weights, constant = train_logistic_regression(training_data, training_data_target, lambda_value, learning_rate, 25)
    prediction_validation_data = test_logistic_regression(validation_data, weights, constant)
    temp_accuracy = evaluate_metrics(validation_data_target, prediction_validation_data)["accuracy"]
    if temp_accuracy > accuracy:
      accuracy = temp_accuracy
      final_lambda_parameter = lambda_value
  return final_lambda_parameter

#Bag of words
total_number_of_records_enron1 = len(bow_training_data["enron1"]["total"])
number_of_records_in_training_set = int(0.7*total_number_of_records_enron1)
number_of_records_in_validation_set = total_number_of_records_enron1 - number_of_records_in_training_set
bow_lg_enron1_training_set = np.array(bow_X_enron1[:number_of_records_in_training_set])
bow_lg_enron1_validation_set = np.array(bow_X_enron1[number_of_records_in_training_set:])
bow_lg_enron1_training_set_target = np.array(bow_Y_enron1[:number_of_records_in_training_set])
bow_lg_enron1_validation_set_target = np.array(bow_Y_enron1[number_of_records_in_training_set:])
bow_enron1_lambda_value = train_lambda(bow_lg_enron1_training_set, bow_lg_enron1_training_set_target, bow_lg_enron1_validation_set, bow_lg_enron1_validation_set_target)
bow_lg_enron1_weights = train_logistic_regression(np.array(bow_X_enron1), np.array(bow_Y_enron1), bow_enron1_lambda_value, 0.01, 100)
bow_enron1_test_data = bag_of_words(enron1_test_data_ham + enron1_test_data_spam, enron1_training_vocabulary)
bow_lg_enron1_prediction = test_logistic_regression(bow_enron1_test_data, bow_lg_enron1_weights[0], bow_lg_enron1_weights[1])
bow_lg_enron1_actual = [1]*len(enron1_test_data_ham) + [0]*len(enron1_test_data_spam)
bow_lg_enron1_evaluation_metrics = evaluate_metrics(bow_lg_enron1_actual, bow_lg_enron1_prediction)
evaluation_metrics["enron1"]["Logistic Regression Bag of words"] = bow_lg_enron1_evaluation_metrics

total_number_of_records_enron4 = len(bow_training_data["enron4"]["total"])
number_of_records_in_training_set = int(0.7*total_number_of_records_enron4)
number_of_records_in_validation_set = total_number_of_records_enron4 - number_of_records_in_training_set
bow_lg_enron4_training_set = np.array(bow_X_enron4[:number_of_records_in_training_set])
bow_lg_enron4_validation_set = np.array(bow_X_enron4[number_of_records_in_training_set:])
bow_lg_enron4_training_set_target = np.array(bow_Y_enron4[:number_of_records_in_training_set])
bow_lg_enron4_validation_set_target = np.array(bow_Y_enron4[number_of_records_in_training_set:])
bow_enron4_lambda_value = train_lambda(bow_lg_enron4_training_set, bow_lg_enron4_training_set_target, bow_lg_enron4_validation_set, bow_lg_enron4_validation_set_target)
bow_lg_enron4_weights = train_logistic_regression(np.array(bow_X_enron4), np.array(bow_Y_enron4), bow_enron4_lambda_value, 0.01, 100)
bow_enron4_test_data = bag_of_words(enron4_test_data_ham + enron4_test_data_spam, enron4_training_vocabulary)
bow_lg_enron4_prediction = test_logistic_regression(bow_enron4_test_data, bow_lg_enron4_weights[0], bow_lg_enron4_weights[1])
bow_lg_enron4_actual = [1]*len(enron4_test_data_ham) + [0]*len(enron4_test_data_spam)
bow_lg_enron4_evaluation_metrics = evaluate_metrics(bow_lg_enron4_actual, bow_lg_enron4_prediction)
evaluation_metrics["enron4"]["Logistic Regression Bag of words"] = bow_lg_enron4_evaluation_metrics

total_number_of_records_hw1 = len(bow_training_data["hw1"]["total"])
number_of_records_in_training_set = int(0.7*total_number_of_records_hw1)
number_of_records_in_validation_set = total_number_of_records_hw1 - number_of_records_in_training_set
bow_lg_hw1_training_set = np.array(bow_X_hw1[:number_of_records_in_training_set])
bow_lg_hw1_validation_set = np.array(bow_X_hw1[number_of_records_in_training_set:])
bow_lg_hw1_training_set_target = np.array(bow_Y_hw1[:number_of_records_in_training_set])
bow_lg_hw1_validation_set_target = np.array(bow_Y_hw1[number_of_records_in_training_set:])
bow_hw1_lambda_value = train_lambda(bow_lg_hw1_training_set, bow_lg_hw1_training_set_target, bow_lg_hw1_validation_set, bow_lg_hw1_validation_set_target)
bow_lg_hw1_weights = train_logistic_regression(np.array(bow_X_hw1), np.array(bow_Y_hw1), bow_hw1_lambda_value, 0.01, 100)
bow_hw1_test_data = bag_of_words(hw1_test_data_ham + hw1_test_data_spam, hw1_training_vocabulary)
bow_lg_hw1_prediction = test_logistic_regression(bow_hw1_test_data, bow_lg_hw1_weights[0], bow_lg_hw1_weights[1])
bow_lg_hw1_actual = [1]*len(hw1_test_data_ham) + [0]*len(hw1_test_data_spam)
bow_lg_hw1_evaluation_metrics = evaluate_metrics(bow_lg_hw1_actual, bow_lg_hw1_prediction)
evaluation_metrics["hw1"]["Logistic Regression Bag of words"] = bow_lg_hw1_evaluation_metrics

#Bernoulli model
total_number_of_records_enron1 = len(bernoulli_training_data["enron1"]["total"])
number_of_records_in_training_set = int(0.7*total_number_of_records_enron1)
number_of_records_in_validation_set = total_number_of_records_enron1 - number_of_records_in_training_set
bernoulli_lg_enron1_training_set = np.array(bernoulli_X_enron1[:number_of_records_in_training_set])
bernoulli_lg_enron1_validation_set = np.array(bernoulli_X_enron1[number_of_records_in_training_set:])
bernoulli_lg_enron1_training_set_target = np.array(bernoulli_Y_enron1[:number_of_records_in_training_set])
bernoulli_lg_enron1_validation_set_target = np.array(bernoulli_Y_enron1[number_of_records_in_training_set:])
bernoulli_enron1_lambda_value = train_lambda(bernoulli_lg_enron1_training_set, bernoulli_lg_enron1_training_set_target, bernoulli_lg_enron1_validation_set, bernoulli_lg_enron1_validation_set_target)
bernoulli_lg_enron1_weights = train_logistic_regression(np.array(bernoulli_X_enron1), np.array(bernoulli_Y_enron1), bernoulli_enron1_lambda_value, 0.01, 100)
bernoulli_enron1_test_data = bernoulli_model(enron1_test_data_ham + enron1_test_data_spam, enron1_training_vocabulary)
bernoulli_lg_enron1_prediction = test_logistic_regression(bernoulli_enron1_test_data, bernoulli_lg_enron1_weights[0], bernoulli_lg_enron1_weights[1])
bernoulli_lg_enron1_actual = [1]*len(enron1_test_data_ham) + [0]*len(enron1_test_data_spam)
bernoulli_lg_enron1_evaluation_metrics = evaluate_metrics(bernoulli_lg_enron1_actual, bernoulli_lg_enron1_prediction)
evaluation_metrics["enron1"]["Logistic Regression Bernoulli Model"] = bernoulli_lg_enron1_evaluation_metrics

total_number_of_records_enron4 = len(bernoulli_training_data["enron4"]["total"])
number_of_records_in_training_set = int(0.7*total_number_of_records_enron4)
number_of_records_in_validation_set = total_number_of_records_enron4 - number_of_records_in_training_set
bernoulli_lg_enron4_training_set = np.array(bernoulli_X_enron4[:number_of_records_in_training_set])
bernoulli_lg_enron4_validation_set = np.array(bernoulli_X_enron4[number_of_records_in_training_set:])
bernoulli_lg_enron4_training_set_target = np.array(bernoulli_Y_enron4[:number_of_records_in_training_set])
bernoulli_lg_enron4_validation_set_target = np.array(bernoulli_Y_enron4[number_of_records_in_training_set:])
bernoulli_enron4_lambda_value = train_lambda(bernoulli_lg_enron4_training_set, bernoulli_lg_enron4_training_set_target, bernoulli_lg_enron4_validation_set, bernoulli_lg_enron4_validation_set_target)
bernoulli_lg_enron4_weights = train_logistic_regression(np.array(bernoulli_X_enron4), np.array(bernoulli_Y_enron4), bernoulli_enron4_lambda_value, 0.01, 100)
bernoulli_enron4_test_data = bernoulli_model(enron4_test_data_ham + enron4_test_data_spam, enron4_training_vocabulary)
bernoulli_lg_enron4_prediction = test_logistic_regression(bernoulli_enron4_test_data, bernoulli_lg_enron4_weights[0], bernoulli_lg_enron4_weights[1])
bernoulli_lg_enron4_actual = [1]*len(enron4_test_data_ham) + [0]*len(enron4_test_data_spam)
bernoulli_lg_enron4_evaluation_metrics = evaluate_metrics(bernoulli_lg_enron4_actual, bernoulli_lg_enron4_prediction)
evaluation_metrics["enron4"]["Logistic Regression Bernoulli Model"] = bernoulli_lg_enron4_evaluation_metrics

total_number_of_records_hw1 = len(bernoulli_training_data["hw1"]["total"])
number_of_records_in_training_set = int(0.7*total_number_of_records_hw1)
number_of_records_in_validation_set = total_number_of_records_hw1 - number_of_records_in_training_set
bernoulli_lg_hw1_training_set = np.array(bernoulli_X_hw1[:number_of_records_in_training_set])
bernoulli_lg_hw1_validation_set = np.array(bernoulli_X_hw1[number_of_records_in_training_set:])
bernoulli_lg_hw1_training_set_target = np.array(bernoulli_Y_hw1[:number_of_records_in_training_set])
bernoulli_lg_hw1_validation_set_target = np.array(bernoulli_Y_hw1[number_of_records_in_training_set:])
bernoulli_hw1_lambda_value = train_lambda(bernoulli_lg_hw1_training_set, bernoulli_lg_hw1_training_set_target, bernoulli_lg_hw1_validation_set, bernoulli_lg_hw1_validation_set_target)
bernoulli_lg_hw1_weights = train_logistic_regression(np.array(bernoulli_X_hw1), np.array(bernoulli_Y_hw1), bernoulli_hw1_lambda_value, 0.01, 100)
bernoulli_hw1_test_data = bernoulli_model(hw1_test_data_ham + hw1_test_data_spam, hw1_training_vocabulary)
bernoulli_lg_hw1_prediction = test_logistic_regression(bernoulli_hw1_test_data, bernoulli_lg_hw1_weights[0], bernoulli_lg_hw1_weights[1])
bernoulli_lg_hw1_actual = [1]*len(hw1_test_data_ham) + [0]*len(hw1_test_data_spam)
bernoulli_lg_hw1_evaluation_metrics = evaluate_metrics(bernoulli_lg_hw1_actual, bernoulli_lg_hw1_prediction)
evaluation_metrics["hw1"]["Logistic Regression Bernoulli Model"] = bernoulli_lg_hw1_evaluation_metrics

#SGDClassifier
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

grid_parameters = {'alpha' : (0.01, 0.05),
              'max_iter' : (range(500, 3000, 1000)),
              'learning_rate': ('optimal', 'invscaling', 'adaptive'),
              'eta0' : (0.3, 0.7),
              'tol' : (0.001, 0.005)}


#Bag of words
bow_clf_enron1 = GridSearchCV( SGDClassifier(), grid_parameters)
bow_clf_enron1.fit(bow_X_enron1, bow_Y_enron1)
bow_enron1_test_data = bag_of_words(enron1_test_data_ham + enron1_test_data_spam, enron1_training_vocabulary)
bow_sgd_enron1_prediction = bow_clf_enron1.predict(bow_enron1_test_data)
bow_sgd_enron1_actual = [1]*len(enron1_test_data_ham) + [0]*len(enron1_test_data_spam)
bow_sgd_enron1_evaluation_metrics = evaluate_metrics(bow_sgd_enron1_actual, bow_sgd_enron1_prediction)
evaluation_metrics["enron1"]["SGDClassifier Bag of words"] = bow_sgd_enron1_evaluation_metrics

bow_clf_enron4 = GridSearchCV( SGDClassifier(), grid_parameters)
bow_clf_enron4.fit(bow_X_enron4, bow_Y_enron4)
bow_enron4_test_data = bag_of_words(enron4_test_data_ham + enron4_test_data_spam, enron4_training_vocabulary)
bow_sgd_enron4_prediction = bow_clf_enron4.predict(bow_enron4_test_data)
bow_sgd_enron4_actual = [1]*len(enron4_test_data_ham) + [0]*len(enron4_test_data_spam)
bow_sgd_enron4_evaluation_metrics = evaluate_metrics(bow_sgd_enron4_actual, bow_sgd_enron4_prediction)
evaluation_metrics["enron4"]["SGDClassifier Bag of words"] = bow_sgd_enron4_evaluation_metrics

bow_clf_hw1 = GridSearchCV( SGDClassifier(), grid_parameters)
bow_clf_hw1.fit(bow_X_hw1, bow_Y_hw1)
bow_hw1_test_data = bag_of_words(hw1_test_data_ham + hw1_test_data_spam, hw1_training_vocabulary)
bow_sgd_hw1_prediction = bow_clf_hw1.predict(bow_hw1_test_data)
bow_sgd_hw1_actual = [1]*len(hw1_test_data_ham) + [0]*len(hw1_test_data_spam)
bow_sgd_hw1_evaluation_metrics = evaluate_metrics(bow_sgd_hw1_actual, bow_sgd_hw1_prediction)
evaluation_metrics["hw1"]["SGDClassifier Bag of words"] = bow_sgd_hw1_evaluation_metrics

#Bernoulli model
bernoulli_clf_enron1 = GridSearchCV( SGDClassifier(), grid_parameters)
bernoulli_clf_enron1.fit(bernoulli_X_enron1, bernoulli_Y_enron1)
bernoulli_enron1_test_data = bernoulli_model(enron1_test_data_ham + enron1_test_data_spam, enron1_training_vocabulary)
bernoulli_sgd_enron1_prediction = bernoulli_clf_enron1.predict(bernoulli_enron1_test_data)
bernoulli_sgd_enron1_actual = [1]*len(enron1_test_data_ham) + [0]*len(enron1_test_data_spam)
bernoulli_sgd_enron1_evaluation_metrics = evaluate_metrics(bernoulli_sgd_enron1_actual, bernoulli_sgd_enron1_prediction)
evaluation_metrics["enron1"]["SGDClassifier Bernoulli"] = bernoulli_sgd_enron1_evaluation_metrics

bernoulli_clf_enron4 = GridSearchCV( SGDClassifier(), grid_parameters)
bernoulli_clf_enron4.fit(bernoulli_X_enron4, bernoulli_Y_enron4)
bernoulli_enron4_test_data = bernoulli_model(enron4_test_data_ham + enron4_test_data_spam, enron4_training_vocabulary)
bernoulli_sgd_enron4_prediction = bernoulli_clf_enron4.predict(bernoulli_enron4_test_data)
bernoulli_sgd_enron4_actual = [1]*len(enron4_test_data_ham) + [0]*len(enron4_test_data_spam)
bernoulli_sgd_enron4_evaluation_metrics = evaluate_metrics(bernoulli_sgd_enron4_actual, bernoulli_sgd_enron4_prediction)
evaluation_metrics["enron4"]["SGDClassifier Bernoulli"] = bernoulli_sgd_enron4_evaluation_metrics

bernoulli_clf_hw1 = GridSearchCV( SGDClassifier(), grid_parameters)
bernoulli_clf_hw1.fit(bernoulli_X_hw1, bernoulli_Y_hw1)
bernoulli_hw1_test_data = bernoulli_model(hw1_test_data_ham + hw1_test_data_spam, hw1_training_vocabulary)
bernoulli_sgd_hw1_prediction = bernoulli_clf_hw1.predict(bernoulli_hw1_test_data)
bernoulli_sgd_hw1_actual = [1]*len(hw1_test_data_ham) + [0]*len(hw1_test_data_spam)
bernoulli_sgd_hw1_evaluation_metrics = evaluate_metrics(bernoulli_sgd_hw1_actual, bernoulli_sgd_hw1_prediction)
evaluation_metrics["hw1"]["SGDClassifier Bernoulli"] = bernoulli_sgd_hw1_evaluation_metrics

print(evaluation_metrics)