#                                   OBJECTIVE


#   Estimate the similarity matrices with Jaccard and Cosine,
#   but for the feature vectors.


# -----------------------------------------------------------------------------------------------


#   Importing required libraries
import math
import pandas as pd


def transpose(l1, arr=[]):
    for i in range(len(l1[0])):
        row = []
        for item in l1:
            row.append(item[i])
        arr.append(row)
    return arr


def jaccards_similarity(i, j):
    union, total = 0, 0
    for i, j in zip(feature_matrix[i], feature_matrix[j]):
        if i == 1 and j == 1:
            union += 1
        if i == 1 or j == 1:
            total += 1
    score = int((union/total)*100)
    return score


def cosine_similiarity(i, j):
    numerator = 0
    a_vector = 0
    b_vector = 0

    for a, b in zip(feature_matrix[i], feature_matrix[j]):
        numerator += (a*b)
        a_vector += a**2
        b_vector += b**2

    denominator = math.sqrt(a_vector) * math.sqrt(b_vector)
    return round((numerator/denominator), 2)


def data_similarity_matrix(similarity, ideal_value):
    arr = []
    for i in range(len(feature_matrix)):
        col = []
        for j in range(len(feature_matrix)):
            if i == j:
                col.append(ideal_value)
            else:
                col.append(similarity(i, j))
        arr.append(col)
    return arr


def matrix_normalise(matrix):
    new_matrix = []
    for i in range(len(matrix)):
        col = []
        for j in range(len(matrix)):
            col.append(round((matrix[i][j]*100), 2))
        new_matrix.append(col)
    return new_matrix


if __name__ == '__main__':

    with open('dataset.txt', 'r') as file:
        data = file.read().split('\n')

    words, data_matrixs = [], []

    for sent in data:
        for word in sent.split():
            if word not in words:
                words.append(word)

    for i in range(len(data)):
        data_matrixs.append([data[i].split().count(w) for w in words])

    feature_matrix = transpose(data_matrixs)

    #   Feature Vector Jaccard similarity
    print('\n')
    jaccard = pd.DataFrame(data_similarity_matrix(jaccards_similarity, 100))
    print('      Jaccard similarity for Feature Vector')
    print('\n')
    print(jaccard)
    print('\n')

    #   Feature Vector Cosine similarity
    cosine = data_similarity_matrix(cosine_similiarity, 1)
    new_cosine = pd.DataFrame(matrix_normalise(cosine))
    print('      Cosine similarity for Feature Vector')
    print('\n')
    print(new_cosine)
