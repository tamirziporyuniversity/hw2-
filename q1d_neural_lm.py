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


def eval_neural_lm(eval_data_path):
    """
    Evaluate perplexity (use dev set when tuning and test at the end)
    """
    _, S_dev = load_data_as_sentences(eval_data_path, word_to_num)
    in_word_index, out_word_index = convert_to_lm_dataset(S_dev)
    assert len(in_word_index) == len(out_word_index)
    num_of_examples = len(in_word_index)

    perplexity = 0
    ### YOUR CODE HERE
    for i in range(num_of_examples):
        data =  num_to_word_embedding[in_word_index[i]]
        label = out_word_index[i]
        ret = forward(data, label, params, dimensions)
        perplexity+=np.log2(ret)
    #raise NotImplementedError
    ### END YOUR CODE
    perplexity = math.pow(2, -perplexity/num_of_examples)
    #after running all: 113.31395087517399
    """
    #params: 104550
#train examples: 1118296
iter 1000: 5.629163
iter 2000: 5.671083
iter 3000: 5.651409
iter 4000: 5.648262
iter 5000: 5.624637
iter 6000: 5.567416
iter 7000: 5.551991
iter 8000: 5.552644
iter 9000: 5.568428
iter 10000: 5.589474
iter 11000: 5.571261
iter 12000: 5.548271
iter 13000: 5.500065
iter 14000: 5.434503
iter 15000: 5.389931
iter 16000: 5.397163
iter 17000: 5.401568
iter 18000: 5.390379
iter 19000: 5.344336
iter 20000: 5.315549
iter 21000: 5.278129
iter 22000: 5.276737
iter 23000: 5.289522
iter 24000: 5.263515
iter 25000: 5.244133
iter 26000: 5.241473
iter 27000: 5.231610
iter 28000: 5.222014
iter 29000: 5.185509
iter 30000: 5.158959
iter 31000: 5.122783
iter 32000: 5.135143
iter 33000: 5.114839
iter 34000: 5.125998
iter 35000: 5.079912
iter 36000: 5.081700
iter 37000: 5.053544
iter 38000: 5.026667
iter 39000: 5.030461
iter 40000: 5.025022
    """
    #after running from 35000: 113.68759154223653
    """
    iter 36000: 4.829140
iter 37000: 4.868955
iter 38000: 4.852504
iter 39000: 4.853829
iter 40000: 4.847035
    """
    return perplexity


if __name__ == "__main__":
    # Load the vocabulary
    vocab = pd.read_table("data/lm/vocab.ptb.txt",
                          header=None, sep="\s+", index_col=0, names=['count', 'freq'], )

    vocabsize = 2000
    num_to_word = dict(enumerate(vocab.index[:vocabsize]))
   
    num_to_word_embedding = load_vocab_embeddings()
    #print(num_to_word_embedding)
    word_to_num = utils.invert_dict(num_to_word)

    # Load the training data
    _, S_train = load_data_as_sentences('data/lm/ptb-train.txt', word_to_num)
    #print(word_to_num)
     
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
    print(f"#params: {len(params)}")
    print(f"#train examples: {num_of_examples}")

    
    # run SGD
    params = sgd(
            lambda vec: lm_wrapper(in_word_index, out_word_index, num_to_word_embedding, dimensions, vec),
            params, LEARNING_RATE, NUM_OF_SGD_ITERATIONS, None, True, 1000)

    print(f"training took {time.time() - startTime} seconds")

    # Evaluate perplexity with dev-data
    perplexity = eval_neural_lm('data/lm/ptb-dev.txt')
    print(f"dev perplexity : {perplexity}")

    # Evaluate perplexity with test-data (only at test time!)
    if os.path.exists('data/lm/ptb-test.txt'):
        perplexity = eval_neural_lm('data/lm/ptb-test.txt')
        print(f"test perplexity : {perplexity}")
    else:
        print("test perplexity will be evaluated only at test time!")


