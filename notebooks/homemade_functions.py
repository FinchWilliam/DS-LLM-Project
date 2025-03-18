from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import string
import re
string.punctuation
import evaluate
import os
import random

f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")
# config variables:




def accuracy_scorer(y_true, y_pred, average='weighted'):
  """
    Return the various metrics

    Args:
        y_true(array): test results to compare against
        y_pred(array): predicted results

    Returns:
        [accuracy, r2, mse, precision, recall, f1], conf_matrix
  """
  metrics = [
      accuracy_score(y_true, y_pred), 
      r2_score(y_true, y_pred), 
      mean_squared_error(y_true, y_pred), 
      precision_score(y_true, y_pred, average=average, zero_division=0),   
      recall_score(y_true, y_pred, average=average),
      f1_score(y_true, y_pred, average=average)
    ]
  conf_matrix = confusion_matrix(y_true, y_pred)
  return metrics, conf_matrix

def word_count(text):
    """
    Convert a list of lists into a single set of words (unique list of the words)
    
    Args:
        text(array of lists): column from the dataframe that is a list of lists

    Returns: 
        a set of the words from that column
    """
    all_words = []
    for list in text:
        for word in list:
            all_words.append(word)
    return set(all_words)

def string_it(list):
    """
    Convert a list into a string

    Args:
        list(list): to be turned into a string
    
    Returns: 
        a string version of the list
    """
    return " ".join(list)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    
    Args: 
        sparse_mx(sparse matrix): sparse matrix (tfidf or BoW) to be converted

    Returns: 
        tensor ready matrix
    
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def clean_tokenize_drop_stops(text):
    """
    Remove all punctuation, makes all words lower-case and turns it 
    into a list of words instead of a single string, finally removing all stop words

    Args:
        text(string): to be cleaned

    Returns:
        lowercase list of words without punctuation or stop-words
    """
    translator = str.maketrans('','',string.punctuation)
    text = text.translate(translator)
    text = text.lower().split()
    result = []
    for word in text:
        if word not in stop_words:
            result.append(word)
    return result

def stem(text):
    """
    Stem the words using PorterStemmer

    Args:
        text(list): tokenized string
    
    Returns:
        list of stemmed words
    """
    stemmer = PorterStemmer()
    result = []
    for word in text:
        result.append(stemmer.stem(word))
    return result

def lemma(text):
    """
    Stem the words using WordNetLemmatizer

    Args:
        text(list): tokenized string
    
    Returns:
        list of Lemma words
    """
    lemmatizer = WordNetLemmatizer()
    result = []
    for word in text:
        result.append(lemmatizer.lemmatize(word))
    return result

def trunc_analysis(model, text, max_length = 510):
    """
    Takes the model and truncates the text to fit in the model, then runs the model

    Args:
        Model (sentiment analysis model)
        text(string): to be analized
        max_length(int): the maximum length of tokens the model can hold

    Returns:
        prediction dictionary in a list [{'label': 'negative', 'score': 0.777593195438385}]
    """
    truncated_text = model.tokenizer.decode(model.tokenizer.encode(text, max_length=max_length, truncation=True))
    pred = model(truncated_text)
    return pred

def tokenize_function(tokenizer, text, truncation=True):
    """
    Tokenize the data for training of a LLM.

    Args:
        tokenizer of choice
        text(string): string to be converted to a token
    Returns:
        token list of the string ex: {'input_ids': [101, 13433, 7361, 2100, 102], 'attention_mask': [1, 1, 1, 1, 1]}

    """
    return tokenizer(text['text'], truncation=truncation)

def remap_labels(example):
    """
    Remaps labels from 0-4 to 0-2.
    
    Args:
        example(int): brings in a list from 0-4 that can be converted to 0-2
    
    """
    label = example['label']
    if label in [0, 1]:
        example['label'] = 0
    elif label == 2:
        example['label'] = 1
    elif label in [3, 4]:
        example['label'] = 2
    return example

def compute_metrics(eval_pred):
    """
    Compute the metrics whilst training the model
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_results = accuracy.compute(predictions=predictions, references=labels)
    f1_results = f1.compute(predictions=predictions, references=labels, average="weighted")  # or "macro", "micro"
    return {**accuracy_results, **f1_results}

def get_unique_filename(base_filename):
    """
    Generates a unique filename by appending a number if the base filename already exists.

    Args:
        base_filename (str): The base filename (e.g., "file").

    Returns:
        str: A unique filename (e.g., "file003").
    """

    counter = 1
    while True:
        filename = f"../data/models/{base_filename}{counter:03d}"  # Format as "file001", "file002", etc.
        if not os.path.exists(filename):
            return f"{base_filename}{counter:03d}"
        counter += 1

def stratified_dataset(dataset, label_column, sample_size_per_class, seed=42):
    """
    Takes a stratified sample from a Hugging Face Dataset.

    Args:
        dataset (Dataset): The Hugging Face Dataset.
        label_column (str): The name of the label column.
        sample_size_per_class (int): The number of samples to take from each class.
        seed (int): random seed for reproducibility.

    Returns:
        Dataset: A stratified sample of the dataset.
    """
    random.seed(seed)
    unique_labels = dataset.unique(label_column)
    stratified_indices = []

    for label in unique_labels:
        indices = [
            i for i, l in enumerate(dataset[label_column]) if l == label
        ]
        sampled_indices = random.sample(
            indices, min(len(indices), sample_size_per_class)
        )
        stratified_indices.extend(sampled_indices)

    return dataset.select(stratified_indices)


