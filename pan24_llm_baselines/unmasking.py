from collections import defaultdict
from random import randint
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

warnings.simplefilter('ignore', category=ConvergenceWarning)


def tokenize(text):
    return [text[i:i + 3] for i in range(0, len(text) - 2)]


def get_token_freqs(*token_lists):
    freqs = defaultdict(int)
    for tokens in token_lists:
        for t in tokens:
            freqs[t] += 1
    return freqs


def bootstrap_tokens(tokens, n_tokens):
    return [tokens[randint(0, len(tokens) - 1)] for _ in range(n_tokens)]


def create_chunks(tokens, chunk_size, n_chunks):
    return [bootstrap_tokens(tokens, chunk_size) for _ in range(n_chunks)]


def chunks_to_matrix(chunks, top_token_list):
    mat = []
    for c in chunks:
        freq = get_token_freqs(c)
        mat.append([freq[t] for t in top_token_list])

    return np.array(mat)


def deconstruct(x_left, x_right, rounds, n_delete=5, cv_folds=10, smoothing_kernel_size=3):
    X = np.vstack((x_left, x_right))
    y = np.zeros(len(x_left) + len(x_right))
    y[:len(x_left)] = 1.0
    X, y = shuffle(X, y)

    rounds = min(rounds, (X.shape[1] - 1) // n_delete)
    scores = np.zeros(rounds)
    for i in range(rounds):
        cv = cross_validate(LinearSVC(dual='auto'), X, y, cv=cv_folds, return_estimator=True)
        scores[i] = max(0.0, (cv['test_score'].mean() - .5) * 2)
        X = np.delete(X, np.argsort(cv['estimator'][0].coef_, axis=None)[::-1][:n_delete], axis=1)

    if smoothing_kernel_size:
        scores = np.convolve(scores, np.ones(smoothing_kernel_size) / smoothing_kernel_size, mode='valid')

    return scores


def score(text_left, text_right, rounds=35, top_n=200, cv_folds=10, n_delete=4, chunk_size=700, n_chunks=30):
    """
    Calculate normalized cumulative sum of the authorship unmasking curve points.

    :param text_left: input text2
    :param text_right: input text2
    :param rounds: number of deconstruction rounds
    :param top_n: number of top tokens to sample
    :param cv_folds: number of cross-validation folds
    :param n_delete: number of features to eliminate in each round
    :param chunk_size: size of bootstrapped chunks
    :param n_chunks: number of chunks to generate
    :return: score in [0, 1] indicating the "humanness" of the text
    """

    tokens_left = tokenize(text_left)
    tokens_right = tokenize(text_right)

    chunks_left = create_chunks(tokens_left, chunk_size, n_chunks)
    chunks_right = create_chunks(tokens_right, chunk_size, n_chunks)

    token_freqs = get_token_freqs(*chunks_left, *chunks_right)
    most_frequent = sorted(token_freqs.keys(), key=lambda x: token_freqs[x], reverse=True)[:top_n]
    x_left = chunks_to_matrix(chunks_left, most_frequent)
    x_right = chunks_to_matrix(chunks_right, most_frequent)

    scores = deconstruct(x_left, x_right, rounds, n_delete, cv_folds)
    return np.sum(scores) / len(scores)
