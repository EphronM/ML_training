#                                   OBJECTIVE


#   Estimate the similarity matrix using Cosine and measure the L1 error
#   between ground truth and estimated similarities.

#   Estimate the distance matrix using Euclidean and verify if the distance
#   estimates agree with the ground truth in principle..

#   Include bigram features and repeat the similarity matrix estimation.
#   Check if adding the bigram features helped decrease the similarity estimation error.


# -----------------------------------------------------------------------------------------------


#   Importing required libraries
import math
import pandas as pd


def generate_bigrams(data):
    unigrams, bigrams = [], []

    for sent in data:
        splitted = []
        splitted = sent.split()

        for word in splitted:
            if word not in unigrams:
                unigrams.append(word)

        for i in range(len(splitted)-1):
            bigrams.append(splitted[i]+' '+splitted[i+1])

    return unigrams, unigrams+bigrams


def jaccards_similarity(i, j, feature_matrix):
    intersection, total = 0, 0
    for i, j in zip(feature_matrix[i], feature_matrix[j]):
        if i == 1 and j == 1:
            intersection += 1
        if i == 1 or j == 1:
            total += 1
    score = int((intersection/total)*100)
    return score


def bigram_feature_matrix(bigrams):

    comb = []
    for sent in data:
        comb.append(sent.split())

    new_set = []
    for sent in comb:
        batch_set = []
        batch_set += sent

        for i in range(len(sent)-1):
            batch_set.append(sent[i]+' '+sent[i+1])
        new_set.append(batch_set)

    matrix = []
    for i in range(len(data)):
        matrix.append([new_set[i].count(w) for w in bigrams])
    return matrix


def cosine_similiarity(i, j, feature_matrix):
    numerator = 0
    a_vector = 0
    b_vector = 0

    for a, b in zip(feature_matrix[i], feature_matrix[j]):
        numerator += (a*b)
        a_vector += a**2
        b_vector += b**2

    denominator = math.sqrt(a_vector) * math.sqrt(b_vector)
    return round((numerator/denominator), 2)


def euclidean_dist(i, j, feature_matrix):
    dist = 0
    for a, b in zip(feature_matrix[i], feature_matrix[j]):
        dist += (a-b)**2
    return round((math.sqrt(dist)), 2)


def data_similarity_matrix(matrix, similarity, ideal_value):
    arr = []
    for i in range(len(data)):
        col = []
        for j in range(len(data)):
            if i == j:
                col.append(ideal_value)
            else:
                col.append(similarity(i, j, matrix))
        arr.append(col)
    return arr


def find_mae(matrix, human_matrix):
    diff = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            diff += abs(matrix[i][j] - human_matrix[i][j])
    mae_score = diff/25
    return mae_score


def matrix_normalise(matrix):
    new_matrix = []
    for i in range(len(matrix)):
        col = []
        for j in range(len(matrix)):
            col.append(round((matrix[i][j]*100), 2))
        new_matrix.append(col)
    return new_matrix


def transpose(l1, arr=[]):
    for i in range(len(l1[0])):
        row = []
        for item in l1:
            row.append(item[i])
        arr.append(row)
    return arr


if __name__ == '__main__':

    with open('dataset.txt', 'r') as file:
        data = file.read().split('\n')

    unigrams, bigrams = generate_bigrams(data)

    #  Extracting Unigram features to create the feature matrix

    uni_feature_matrix = []
    for i in range(len(data)):
        uni_feature_matrix.append([data[i].split().count(w) for w in unigrams])

    uni_feature_df = pd.DataFrame(uni_feature_matrix)  # convert into dataframe

    #   Unigram Jaccard similarity matrix
    uni_jaccard = pd.DataFrame(data_similarity_matrix(
        uni_feature_matrix, jaccards_similarity, 100))

    #   Unigram cosine similarity matrix after normalising
    uni_cosine = data_similarity_matrix(
        uni_feature_matrix, cosine_similiarity, 1)
    uni_cosine_normalized = pd.DataFrame(matrix_normalise(uni_cosine))

    #   unigram euclidean similarity matrix
    uni_euclidean = pd.DataFrame(data_similarity_matrix(
        uni_feature_matrix, euclidean_dist, 0))

    # Bigrams Feature matrix
    bi_feature_matrix = pd.DataFrame(bigram_feature_matrix(bigrams))

    #   Bigram Jaccard similarity matrix
    bi_jaccard = pd.DataFrame(data_similarity_matrix(
        bi_feature_matrix, jaccards_similarity, 100))

    #   Bigram cosine similarity matrix after normalising
    bi_cosine = data_similarity_matrix(
        bi_feature_matrix, cosine_similiarity, 1)
    bi_cosine_normalized = pd.DataFrame(matrix_normalise(bi_cosine))

    #   Bigram euclidean similarity matrix
    bi_euclidean = pd.DataFrame(data_similarity_matrix(
        bi_feature_matrix, euclidean_dist, 0))

    # Human similarity based on our personal opinions
    human_similarity = [[100, 0, 40, 30, 0],
                        [0, 100, 0, 20, 20],
                        [40, 0, 100, 0, 20],
                        [30, 20, 0, 100, 0],
                        [0, 20, 20, 0, 100]]

    human_df = pd.DataFrame(human_similarity)

    # L1 Error Calculation wrt human similarities

    print("MAE of uni_cosin >> ", find_mae(
        uni_cosine_normalized, human_similarity))
    print("MAE of uni_jaccard >> ", find_mae(uni_jaccard, human_similarity))
    print('\n')
    print("MAE of bi_cosin >> ", find_mae(
        bi_cosine_normalized, human_similarity))
    print("MAE of bi_jaccard >> ", find_mae(bi_jaccard, human_similarity))
#                                   OBJECTIVE


#   Estimate the similarity matrix using Cosine and measure the L1 error
#   between ground truth and estimated similarities.

#   Estimate the distance matrix using Euclidean and verify if the distance
#   estimates agree with the ground truth in principle..

#   Include bigram features and repeat the similarity matrix estimation.
#   Check if adding the bigram features helped decrease the similarity estimation error.


# -----------------------------------------------------------------------------------------------


#   Importing required libraries
import math
import pandas as pd


def generate_bigrams(data):
    unigrams, bigrams = [], []

    for sent in data:
        splitted = []
        splitted = sent.split()

        for word in splitted:
            if word not in unigrams:
                unigrams.append(word)

        for i in range(len(splitted)-1):
            bigrams.append(splitted[i]+' '+splitted[i+1])

    return unigrams, unigrams+bigrams


def jaccards_similarity(i, j, feature_matrix):
    intersection, total = 0, 0
    for i, j in zip(feature_matrix[i], feature_matrix[j]):
        if i == 1 and j == 1:
            intersection += 1
        if i == 1 or j == 1:
            total += 1
    score = int((intersection/total)*100)
    return score


def bigram_feature_matrix(bigrams):

    comb = []
    for sent in data:
        comb.append(sent.split())

    new_set = []
    for sent in comb:
        batch_set = []
        batch_set += sent

        for i in range(len(sent)-1):
            batch_set.append(sent[i]+' '+sent[i+1])
        new_set.append(batch_set)

    matrix = []
    for i in range(len(data)):
        matrix.append([new_set[i].count(w) for w in bigrams])
    return matrix


def cosine_similiarity(i, j, feature_matrix):
    numerator = 0
    a_vector = 0
    b_vector = 0

    for a, b in zip(feature_matrix[i], feature_matrix[j]):
        numerator += (a*b)
        a_vector += a**2
        b_vector += b**2

    denominator = math.sqrt(a_vector) * math.sqrt(b_vector)
    return round((numerator/denominator), 2)


def euclidean_dist(i, j, feature_matrix):
    dist = 0
    for a, b in zip(feature_matrix[i], feature_matrix[j]):
        dist += (a-b)**2
    return round((math.sqrt(dist)), 2)


def data_similarity_matrix(matrix, similarity, ideal_value):
    arr = []
    for i in range(len(data)):
        col = []
        for j in range(len(data)):
            if i == j:
                col.append(ideal_value)
            else:
                col.append(similarity(i, j, matrix))
        arr.append(col)
    return arr


def find_mae(matrix, human_matrix):
    diff = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            diff += abs(matrix[i][j] - human_matrix[i][j])
    mae_score = diff/25
    return mae_score


def matrix_normalise(matrix):
    new_matrix = []
    for i in range(len(matrix)):
        col = []
        for j in range(len(matrix)):
            col.append(round((matrix[i][j]*100), 2))
        new_matrix.append(col)
    return new_matrix


def transpose(l1, arr=[]):
    for i in range(len(l1[0])):
        row = []
        for item in l1:
            row.append(item[i])
        arr.append(row)
    return arr


if __name__ == '__main__':

    with open('dataset.txt', 'r') as file:
        data = file.read().split('\n')

    unigrams, bigrams = generate_bigrams(data)

    #  Extracting Unigram features to create the feature matrix

    uni_feature_matrix = []
    for i in range(len(data)):
        uni_feature_matrix.append([data[i].split().count(w) for w in unigrams])

    uni_feature_df = pd.DataFrame(uni_feature_matrix)  # convert into dataframe

    #   Unigram Jaccard similarity matrix
    uni_jaccard = pd.DataFrame(data_similarity_matrix(
        uni_feature_matrix, jaccards_similarity, 100))

    #   Unigram cosine similarity matrix after normalising
    uni_cosine = data_similarity_matrix(
        uni_feature_matrix, cosine_similiarity, 1)
    uni_cosine_normalized = pd.DataFrame(matrix_normalise(uni_cosine))

    #   unigram euclidean similarity matrix
    uni_euclidean = pd.DataFrame(data_similarity_matrix(
        uni_feature_matrix, euclidean_dist, 0))

    # Bigrams Feature matrix
    bi_feature_matrix = pd.DataFrame(bigram_feature_matrix(bigrams))

    #   Bigram Jaccard similarity matrix
    bi_jaccard = pd.DataFrame(data_similarity_matrix(
        bi_feature_matrix, jaccards_similarity, 100))

    #   Bigram cosine similarity matrix after normalising
    bi_cosine = data_similarity_matrix(
        bi_feature_matrix, cosine_similiarity, 1)
    bi_cosine_normalized = pd.DataFrame(matrix_normalise(bi_cosine))

    #   Bigram euclidean similarity matrix
    bi_euclidean = pd.DataFrame(data_similarity_matrix(
        bi_feature_matrix, euclidean_dist, 0))

    # Human similarity based on our personal opinions
    human_similarity = [[100, 0, 40, 30, 0],
                        [0, 100, 0, 20, 20],
                        [40, 0, 100, 0, 20],
                        [30, 20, 0, 100, 0],
                        [0, 20, 20, 0, 100]]

    human_df = pd.DataFrame(human_similarity)

    # L1 Error Calculation wrt human similarities

    print("MAE of uni_cosin >> ", find_mae(
        uni_cosine_normalized, human_similarity))
    print("MAE of uni_jaccard >> ", find_mae(uni_jaccard, human_similarity))
    print('\n')
    print("MAE of bi_cosin >> ", find_mae(
        bi_cosine_normalized, human_similarity))
    print("MAE of bi_jaccard >> ", find_mae(bi_jaccard, human_similarity))
