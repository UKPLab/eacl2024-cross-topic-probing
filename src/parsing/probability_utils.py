import itertools
import math
import string
from collections import Counter, defaultdict

import nltk
import numpy
import pandas
from nltk.corpus import stopwords


def filter_probabilities_cumsum(probabilities, threshold:float=0.9):
    sorted_probabilities = sorted(probabilities.items(), key=lambda ele: ele[1], reverse=True)
    cummulated_probabilities = numpy.array(sorted_probabilities).T[1].astype(numpy.float).cumsum()
    filtered_probabilities = [cummulated_probability for cummulated_probability in cummulated_probabilities if cummulated_probability <= threshold]
    selected_probabilities = dict(sorted_probabilities[:len(filtered_probabilities)])

    return selected_probabilities

def get_probabilities(entries, probability_threshold=0.9):
    counts = dict(Counter(entries))

    return counts


def join_series(series):
    return series.str.cat(sep=" ").lower()


def valid_ngram(ngram):
    if type(ngram) == str:
        return all([char not in string.punctuation for char in ngram]) and len(ngram) > 1
    has_stopwords = any(
        [word in stopwords.words("english") for word in ngram])
    has_puncation = any(
        [word in string.punctuation for word in ngram])

    return not has_stopwords and not has_puncation

def get_common_words_for_pair(pair):
    words = [set(nltk.word_tokenize(ele)) for ele in pair]

    return set.intersection(*words)


def get_distinct_words_for_pair(pair):
    words = [set(nltk.word_tokenize(ele)) for ele in pair]

    return set.difference(*words)


def get_word_probabilities(text_series, probability_threshold=0.9, force_joined_processing=False,  force_distinct_processing=False):

    concatenated_series = text_series.apply(" ".join)

    text = join_series(concatenated_series)
    words = nltk.word_tokenize(text)
    #words = [word for word in words if word not in stopwords.words("english") and word not in string.punctuation]

    if force_joined_processing:
        text = join_series(concatenated_series)
        words = nltk.word_tokenize(text)

        common_words = text_series.apply(get_common_words_for_pair)
        whitelist_words = set(itertools.chain.from_iterable(common_words.values))

    elif force_distinct_processing:
        text = join_series(concatenated_series)
        words = nltk.word_tokenize(text)

        distinct_words = text_series.apply(get_distinct_words_for_pair)
        whitelist_words = set(itertools.chain.from_iterable(distinct_words.values))
    else:
        whitelist_words = set(words)


    #bigrams = list(ngrams(words,2))
    #trigrams = list(ngrams(words,3))

    #words = words + [" ".join(ele) for ele in bigrams + trigrams]

    word_probabilities = get_probabilities(words, probability_threshold=probability_threshold)
    filtered_word_probabilities = {key:value for key, value in word_probabilities.items() if key in whitelist_words}
    return word_probabilities

def get_conditional_odds_ratio(samples: pandas.DataFrame, text_column:str, reference_column:str, probability_threshold:float=0.9, force_joined_processing=False, force_distinct_processing=False, is_numeric=False):

    if is_numeric:
        samples["label"] = pandas.qcut(samples["label"] , q=3).factorize()[0]

    word_reference_probabilities = {
        condition: get_word_probabilities(
            reference_frame[text_column], probability_threshold=probability_threshold, force_joined_processing=force_joined_processing, force_distinct_processing=force_distinct_processing
        )
        for condition, reference_frame in samples.groupby(reference_column)
    }

    reference_counts = {
        condition: sum(counts.values())
        for condition, counts in word_reference_probabilities.items()
    }

    odds_ratios = defaultdict(dict)
    for reference, probabilities in word_reference_probabilities.items():
        for word, word_reference_probability in probabilities.items():
            if valid_ngram(word):
                conditional_word_reference_probability = (word_reference_probability / sum(reference_counts.values())) / (reference_counts[reference] / sum(reference_counts.values()))
                inverse_summed = sum([
                    word_reference_probabilities[other_reference].get(word, min(word_reference_probabilities[other_reference].values()))
                    for other_reference in word_reference_probabilities.keys()
                    if other_reference != reference
                ])

                other_summed = sum(reference_counts.values()) - reference_counts[reference]

                inverse_probability = (inverse_summed / sum(reference_counts.values())) / (other_summed / sum(reference_counts.values()))

                if inverse_probability == 0:
                    continue
                odds_ratio = math.log10(
                    conditional_word_reference_probability / inverse_probability
                )
                odds_ratios[reference][word] = odds_ratio
    return odds_ratios

def get_conditional_odds_ratio_majority(samples: pandas.DataFrame, text_column:str, reference_column:str, probability_threshold:float=0.9, force_joined_processing=False, force_distinct_processing=False, is_numeric=False):

    if is_numeric:
        samples["label"] = pandas.qcut(samples["label"] , q=3).factorize()[0]

    word_reference_probabilities = {
        condition: get_word_probabilities(
            reference_frame[text_column], probability_threshold=probability_threshold, force_joined_processing=force_joined_processing, force_distinct_processing=force_distinct_processing
        )
        for condition, reference_frame in samples.groupby(reference_column)
    }

    reference_counts = {
        condition: sum(counts.values())
        for condition, counts in word_reference_probabilities.items()
    }

    odds_ratios = defaultdict(list)
    for reference, probabilities in word_reference_probabilities.items():
        for word, word_reference_probability in probabilities.items():
            if valid_ngram(word):
                conditional_word_reference_probability = (word_reference_probability / sum(reference_counts.values())) / (reference_counts[reference] / sum(reference_counts.values()))
                inverse_summed = sum([
                    word_reference_probabilities[other_reference].get(word, min(word_reference_probabilities[other_reference].values()))
                    for other_reference in word_reference_probabilities.keys()
                    if other_reference != reference
                ])

                other_summed = sum(reference_counts.values()) - reference_counts[reference]

                inverse_probability = (inverse_summed / sum(reference_counts.values())) / (other_summed / sum(reference_counts.values()))

                if inverse_probability == 0:
                    continue
                odds_ratio = math.log10(
                    conditional_word_reference_probability / inverse_probability
                )
                odds_ratios[word].append(odds_ratio)

    odds_ratios = {word: max(word_odd_ratios) for word, word_odd_ratios in odds_ratios.items()}

    return odds_ratios


def get_mutual_information(samples: pandas.DataFrame, text_column:str, reference_column:str, probability_threshold:float=0.9, force_joined_processing=False, force_distinct_processing=False, is_numeric=False):

    if is_numeric:
        samples["label"] = pandas.qcut(samples["label"] , q=3).factorize()[0]

    word_reference_probabilities = {
        condition: get_word_probabilities(
            reference_frame[text_column], probability_threshold=probability_threshold, force_joined_processing=force_joined_processing, force_distinct_processing=force_distinct_processing
        )
        for condition, reference_frame in samples.groupby(reference_column)
    }

    reference_probabilities = {
        condition: conditional_frame.shape[0] / samples.shape[0]
        for condition, conditional_frame in samples.groupby(reference_column)
    }

    word_probabilities = get_word_probabilities(
        samples[text_column], probability_threshold=probability_threshold, force_joined_processing=force_joined_processing, force_distinct_processing=force_distinct_processing
    )


    mutual_informations = defaultdict(dict)
    for reference, probabilities in word_reference_probabilities.items():
        for word, word_reference_probability in probabilities.items():
            if valid_ngram(word):
                word_probability = word_probabilities.get(word, min(word_probabilities.values()))
                reference_probability = reference_probabilities[reference]
                mutual_information = math.log10(word_reference_probability / (reference_probability * word_probability))
                mutual_informations[reference][word] = mutual_information

    return mutual_informations



def get_property_words(samples, property, force_joined_processing=False, force_distinct_processing=False, is_numeric=False, limit=False, majority=False):

    property_odds_ratios = get_conditional_odds_ratio(
        samples, text_column="inputs", reference_column=property, probability_threshold=1.0, force_joined_processing=force_joined_processing, force_distinct_processing=force_distinct_processing, is_numeric=is_numeric
    )

    property_words = {}

    for property, property_odds_ratio in property_odds_ratios.items():
        if limit:
            property_odds = {word: odds for word, odds in property_odds_ratio.items() > 0}
        else:
            property_odds = {word: odds for word, odds in property_odds_ratio.items()}

        property_words[property] = property_odds

    return property_words

def get_property_words_majority(samples, property, force_joined_processing=False, force_distinct_processing=False, is_numeric=False, limit=False):

    word_odds_ratios = get_conditional_odds_ratio_majority(
        samples, text_column="inputs", reference_column=property, probability_threshold=1.0, force_joined_processing=force_joined_processing, force_distinct_processing=force_distinct_processing, is_numeric=is_numeric
    )

    return word_odds_ratios


def get_property_words_mi(samples, property, force_joined_processing=False, force_distinct_processing=False, is_numeric=False, limit=False):

    mutual_informations = get_mutual_information(
        samples, text_column="inputs", reference_column=property, probability_threshold=1.0, force_joined_processing=force_joined_processing, force_distinct_processing=force_distinct_processing, is_numeric=is_numeric
    )

    property_words = {}

    for property, property_mutual_informations in mutual_informations.items():
        if limit:
            property_odds = {word: mutual_information for word, mutual_information in property_mutual_informations.items() > 0}
        else:
            property_odds = {word: mutual_information for word, mutual_information in property_mutual_informations.items()}

        property_words[property] = property_odds

    return property_words
