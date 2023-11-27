import os
import random
import time

import numpy as np
import pandas as pd

import math

from data_utils import utils
from sgd import sgd
from q1c_neural import forward, forward_backward_prop


VOCAB_EMBEDDING_PATH = "data/lm/vocab.embeddings.glove.txt"
BATCH_SIZE = 50
NUM_OF_SGD_ITERATIONS = 40000
LEARNING_RATE = 0.3


def load_vocab_embeddings(path=VOCAB_EMBEDDING_PATH):
    result = []
    with open(path) as f:
        index = 0
        for line in f:
            line = line.strip()
            row = line.split()
            data = [float(x) for x in row[1:]]
            assert len(data) == 50
            result.append(data)
            index += 1
    return result


def load_data_as_sentences(path, word_to_num):
    """
    Converts the training data to an array of integer arrays.
      args: 
        path: string pointing to the training data
        word_to_num: A dictionary from string words to integers
      returns:
        An array of integer arrays. Each array is a sentence and each 
        integer is a word.
    """
    docs_data = utils.load_dataset(path)
    S_data = utils.docs_to_indices(docs_data, word_to_num)
    return docs_data, S_data


def convert_to_lm_dataset(S):
    """
    Takes a dataset that is a list of sentences as an array of integer arrays.
    Returns the dataset a bigram prediction problem. For any word, predict the
    next work. 
    IMPORTANT: we have two padding tokens at the beginning but since we are 
    training a bigram model, only one will be used.
    """
    in_word_index, out_word_index = [], []
    for sentence in S:
        for i in range(len(sentence)):
            if i < 2:
                continue
            in_word_index.append(sentence[i - 1])
            out_word_index.append(sentence[i])
    return in_word_index, out_word_index


def shuffle_training_data(in_word_index, out_word_index):
    combined = list(zip(in_word_index, out_word_index))
    random.shuffle(combined)
    return list(zip(*combined))


def int_to_one_hot(number, dim):
    res = np.zeros(dim)
    res[number] = 1.0
    return res


def lm_wrapper(in_word_index, out_word_index, num_to_word_embedding, dimensions, params):

    input_dim, hidden_dim, output_dim = dimensions

    data = np.zeros([BATCH_SIZE, input_dim])
    labels = np.zeros([BATCH_SIZE, output_dim])

    # Construct the data batch and run you backpropogation implementation
    ### YOUR CODE HERE
    #raise NotImplementedError

    num_random_indices = BATCH_SIZE

    # Choose random indices
    random_indices = np.random.choice(len(in_word_index), size=num_random_indices, replace=False)

    for idx, item in enumerate(random_indices):
        
        data[idx] = num_to_word_embedding[in_word_index[item]]
        labels[idx] = int_to_one_hot(out_word_index[item], output_dim)

    cost, grad = forward_backward_prop(data, labels, params, dimensions)
  
    
    ### END YOUR CODE

    cost /= BATCH_SIZE
    grad /= BATCH_SIZE
    return cost, grad

import re
def load_datasetq4(fname):
    docs = []
    with open(fname, 'r', encoding ='utf-8') as fd:
        cur = []
        for line in fd:
            # new sentence on -DOCSTART- or blank line
            if re.match(r"-DOCSTART-.+", line) or (len(line.strip()) == 0):
                if len(cur) > 0:
                    docs.append(cur)
                cur = []
            else:  # read in tokens
                cur.append(line.strip().split("\t",1))
        # flush running buffer
        docs.append(cur)
    return docs


def load_data_as_sentences_q4(file_path, word_to_num, encoding='utf-8'):
    docs_data = load_datasetq4(file_path)
    S_data = utils.docs_to_indices(docs_data, word_to_num)
    return S_data


def eval_neural_lm(eval_data_path, method):
    """
    Evaluate perplexity (use dev set when tuning and test at the end)
    """
    S_dev = load_data_as_sentences_q4(eval_data_path, word_to_num)
    in_word_index, out_word_index = convert_to_lm_dataset(S_dev)
    assert len(in_word_index) == len(out_word_index)
    num_of_examples = len(in_word_index)

    perplexity = 0
    ### YOUR CODE HERE
    for i in range(num_of_examples):
        data =  num_to_word_embedding[in_word_index[i]]
        label = out_word_index[i]
        ret = forward(data, label, params, dimensions)
        if method == 'two':
            perplexity+=np.log2(ret)
        else:
            perplexity+=np.log(ret)
    #raise NotImplementedError
    ### END YOUR CODE
    if method == 'two':
        perplexity = math.pow(2, -perplexity/num_of_examples)
    else:
        perplexity = np.exp(-perplexity/num_of_examples)
    
    return perplexity


if __name__ == "__main__":
    # Load the vocabulary
    vocab = pd.read_table("data/lm/vocab.ptb.txt",
                          header=None, sep="\s+", index_col=0, names=['count', 'freq'], )

    vocabsize = 2000
    num_to_word = dict(enumerate(vocab.index[:vocabsize]))
   
    num_to_word_embedding = load_vocab_embeddings()
    word_to_num = utils.invert_dict(num_to_word)

    _, S_train = load_data_as_sentences('data/lm/ptb-train.txt', word_to_num)
     
    in_word_index, out_word_index = convert_to_lm_dataset(S_train)
 
    assert len(in_word_index) == len(out_word_index)
    num_of_examples = len(in_word_index)

    random.seed(31415)
    np.random.seed(9265)
    in_word_index, out_word_index = shuffle_training_data(in_word_index, out_word_index)
    startTime = time.time()

    # Training should happen here
    # Initialize parameters randomly
    # Construct the params
    input_dim = 50
    hidden_dim = 50
    output_dim = vocabsize
    dimensions = [input_dim, hidden_dim, output_dim]
    params = np.random.randn((input_dim + 1) * hidden_dim + (
        hidden_dim + 1) * output_dim, )
   

    
    # run SGD
    params = sgd(
            lambda vec: lm_wrapper(in_word_index, out_word_index, num_to_word_embedding, dimensions, vec),
            params, LEARNING_RATE, NUM_OF_SGD_ITERATIONS, None, True, 1000)

    print(f"training took {time.time() - startTime} seconds")

    # Evaluate perplexity with dev-data

    
    perplexity_shakespeare_2 = eval_neural_lm('data/shakespeare.txt', '2') #9.332726040746131
    print(f"test perplexity with 2 for shakespeare : {perplexity_shakespeare_2}")

    perplexity_wikipedia_2 = eval_neural_lm('wikipedia_for_perplexity.txt', '2') #24.69498410960043
    print(f"test perplexity with 2 for wikipedia : {perplexity_wikipedia_2}")

    perplexity_shakespeare_ln = eval_neural_lm('data/shakespeare.txt', 'ln') #9.332726040746131
    print(f"test perplexity with ln for shakespeare : {perplexity_shakespeare_ln}")

    perplexity_wikipedia_ln = eval_neural_lm('wikipedia_for_perplexity.txt', 'ln') #24.69498410960043
    print(f"test perplexity with ln for wikipedia : {perplexity_wikipedia_ln}")

    #its same :)



    