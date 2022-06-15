#                                   OBJECTIVE


#   Estimate the Distribution of vowels, stopwords and punctuations
#   from a given Corpus of text file


# -----------------------------------------------------------------------------------------------


#   Importing required libraries

import string
import re
import matplotlib.pyplot as plt


def word_cleaner(word):  # Cleaning Puntuations
    cleaned = []
    for char in word:
        if char not in punts:
            if char not in numbers:
                cleaned.append(char)
    return ''.join(cleaned)


def vowels_counter(article, vowel_dict={}):  # Counting freq of vowels
    vowel_dict = {v: raw_data.lower().count(v) for v in vowels}
    return vowel_dict


def count_stopwords(cleaned_list, std_dict={}):  # Counting stopwords
    std_dict = {s: cleaned_list.lower().count(s) for s in stopwords}
    return std_dict


def count_puntuations(article, punt_dict={}):  # Counting Puntuations
    punt_dict = {p: article.count(p) for p in punts}
    return punt_dict


def words_caps(cleaned_list, cap_letter={}):
    for word in cleaned_list.split(' '):
        match = re.search('[A-Z]', word)
        if match:
            if word in cap_letter.keys():
                cap_letter[word] += 1
            else:
                cap_letter[word] = 1
    return cap_letter


if __name__ == "__main__":

    punts = list(string.punctuation)
    vowels = ['a', 'e', 'i', 'o', 'u']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

    with open("demo.txt", "r",  encoding="utf8") as file:       # Reading the text corpus
        raw_data = file.read()

    # Reading the textfile containing the stopwords
    with open('stop.txt', 'r') as stp:
        stopwords = stp.read().split('\n')

    cleaned_article = ' '.join([word_cleaner(word)              # converting the corpus into a single para
                               for word in raw_data.split()])

    stopwords_dict = count_stopwords(cleaned_article)
    vowels_dict = vowels_counter(raw_data)
    cap_word_dict = words_caps(cleaned_article)
    punt_count = count_puntuations(raw_data)

    dists = [stopwords_dict, vowels_dict, cap_word_dict, punt_count]
    labels = ['Stopwords', 'Vowels', 'Capital Words', 'Puntuations']

    for d in range(len(dists)):                                         # Plotting Dist
        plt.plot([i for i in range(len(dists[d]))], dists[d].values())
        plt.xlabel(f'{labels[d]}')
        plt.ylabel('Freq')
        plt.title(f'{labels[d]} Distribution')
        plt.show()

    # ploting vowels
    plt.bar(vowels_dict.keys(), vowels_dict.values())
    plt.xlabel('Vowels')
    plt.ylabel('Freq')
    plt.title('Top15 Vowel')
    plt.show()

    # ploting stopwords ( Plotting the top15 frequent appearing stopwords )

    top15_sw = sorted(stopwords_dict, key=stopwords_dict.get,
                      reverse=True)[:15]
    sw_values = [stopwords_dict[sw] for sw in top15_sw]

    plt.bar(top15_sw, sw_values)
    plt.xlabel('Stopwords')
    plt.ylabel('Freq')
    plt.title('Top15 Stopwords')
    plt.show()

    # ploting words with capital letter
    top15_cw = sorted(cap_word_dict, key=cap_word_dict.get, reverse=True)[:15]
    cw_values = [cap_word_dict[cw] for cw in top15_cw]

    plt.bar(top15_cw, cw_values)
    plt.xticks(rotation=90)
    plt.xlabel('Capital Word')
    plt.ylabel('Freq')
    plt.title('Top15 Capital letter')
    plt.show()

    # ploting puntuation symbols
    top15_pw = sorted(punt_count, key=punt_count.get, reverse=True)[:15]
    pw_values = [punt_count[pw] for pw in top15_pw]

    plt.bar(top15_pw, pw_values)
    plt.xlabel('Puntuation symbols')
    plt.ylabel('Freq')
    plt.title('Top15 Puntuation')
    plt.show()
