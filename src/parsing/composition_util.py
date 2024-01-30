import os
import re
import string
from collections import Counter

import pandas
import spacy
import yaml
from PyMultiDictionary import MultiDictionary
from nltk import TreebankWordTokenizer
from nltk.corpus import stopwords
from pandarallel import pandarallel
from sacremoses import MosesTokenizer
from sklearn.preprocessing import KBinsDiscretizer
from spacy.tokens import Doc
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Strip, NFC, NFKD, NFKC
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

from defs.control_task_types import CONTROL_TASK_TYPES
from defs.probe_task_types import PROBE_TASK_TYPES
from parsing.parsing_util import save_topic_folds, save_folds
from parsing.probability_utils import get_property_words_majority

dictionary = MultiDictionary()
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("negex")
normalizer = normalizers.Sequence([NFD(), NFC(), NFKD(), NFKC(), StripAccents(), Strip()])

def get_positions(span, subspan, nth_occurrence):
    all_start_positions = [i for i in range(len(span)) if span.startswith(subspan, i)]

    start = all_start_positions[nth_occurrence]
    end = start + len(subspan)

    return (start, end)


trebank_tokenizer = TreebankWordTokenizer()
bert_tokenizer = Whitespace()
tokenizer = MosesTokenizer()


def custom_tokenizer(text):
    tokens = tokenizer.tokenize(text, escape=False)
    return Doc(nlp.vocab, tokens)


nlp.tokenizer = custom_tokenizer
stop_words = stopwords.words("english")

def decontracted(phrase):
    # specific
    processed_phrase = re.sub(r"won\'t", "will not", phrase)
    processed_phrase = re.sub(r"can\'t", "can not", processed_phrase)
    processed_phrase = re.sub(r"cannot", "can not", processed_phrase)

    # general
    processed_phrase = re.sub(r"n\'t", " not", processed_phrase)
    processed_phrase = re.sub(r"\'re", " are", processed_phrase)
    processed_phrase = re.sub(r"\'s", " is", processed_phrase)
    processed_phrase = re.sub(r"\'d", " would", processed_phrase)
    processed_phrase = re.sub(r"\'ll", " will", processed_phrase)
    processed_phrase = re.sub(r"\'t", " not", processed_phrase)
    processed_phrase = re.sub(r"\'ve", " have", processed_phrase)
    processed_phrase = re.sub(r"\'m", " am", processed_phrase)
    processed_phrase = re.sub(r"″", "''", processed_phrase)
    processed_phrase = re.sub(r"’", "'", processed_phrase)


    if processed_phrase[-1] == "." and processed_phrase[-2] != " ":
        processed_phrase = processed_phrase[:-1] + " ."
    if processed_phrase[-1] == "!" and processed_phrase[-2] != " ":
        processed_phrase = processed_phrase[:-1] + " !"
    if processed_phrase[-1] == "?" and processed_phrase[-2] != " ":
        processed_phrase = processed_phrase[:-1] + " ?"


    processed_phrase = processed_phrase.replace("...", "").replace("..", "")
    processed_phrase = processed_phrase.replace("---", "").replace("--", "")
    processed_phrase = processed_phrase.replace("<br>", "").replace("</br>", "")
    processed_phrase = processed_phrase.replace(",", " , ")
    processed_phrase = processed_phrase.replace(";", " ; ")
    processed_phrase = processed_phrase.replace(":", " : ")
    processed_phrase = processed_phrase.replace("!", " ! ")
    processed_phrase = processed_phrase.replace("?", " ? ")
    processed_phrase = processed_phrase.replace("(", " ( ")
    processed_phrase = processed_phrase.replace(")", " ) ")
    processed_phrase = processed_phrase.replace("[", " [ ")
    processed_phrase = processed_phrase.replace("]", " ] ")
    processed_phrase = processed_phrase.replace("]", " ] ")
    processed_phrase = processed_phrase.replace("′", "'")
    processed_phrase = processed_phrase.replace('\xad', '')
    processed_phrase = processed_phrase.replace('\x90', '')
    processed_phrase = processed_phrase.encode("ascii", "ignore").decode()
    processed_phrase = processed_phrase.replace('\n', ' ')
    processed_phrase = processed_phrase.replace('\r', ' ')

    processed_phrase = processed_phrase.replace("'", " ' ")
    processed_phrase = processed_phrase.replace("   ", " ")
    processed_phrase = processed_phrase.replace("  ", " ")

    processed_phrase = nlp(processed_phrase).text
    processed_phrase = normalizer.normalize_str(processed_phrase)

    processed_phrase = re.sub(' +', ' ', processed_phrase)

    processed_phrase = processed_phrase.replace("1⁄2", "0.5")
    processed_phrase = processed_phrase.replace("1⁄4", "0.25")
    processed_phrase = processed_phrase.replace("1⁄5", "0.2")
    processed_phrase = processed_phrase.replace("1⁄8", "0.125")

    return processed_phrase


def scale_labels(labels):
    return (labels - labels.min()) / (labels.max() - labels.min())


def dump_config(config, name):
    yaml.dump(config, open(name, 'w'))



def save_probe(samples, name, path, fold_topics=None, fold_samples=None, sample_column="inputs", pairwise=False,
               token_level=False, max_samples=None):

    os.system("mkdir -p ../../" + path + "/" + name)

    num_labels = 1 if isinstance(samples["label"].values[0], float) else len(set(samples["label"]))
    num_inputs = len(samples[sample_column].values[0])

    generate_config(
        num_labels=num_labels,
        num_inputs=num_inputs,
        num_probe_folds=3,
        task=name,
        pairwise=pairwise,
        token_level=token_level,
        path=path
    )


    if fold_topics:

        if max_samples and samples.shape[0] > max_samples:
            samples = samples.sample(max_samples, random_state=0)

        save_topic_folds(
            samples=samples,
            fold_topics=fold_topics,
            data_path=path,
            task_name=name,
            sample_column=sample_column
        )

    elif fold_samples:
        if max_samples and samples.shape[0] > max_samples:
            samples = samples.sample(max_samples, random_state=0)

        save_folds(
            samples=samples,
            n_folds=3,
            data_path=path,
            task_name=name,
            sample_column=sample_column,
            fold_samples=fold_samples,
        )

    else:
        if max_samples and samples.shape[0] > max_samples:
            samples = samples.sample(max_samples, random_state=0)

        save_folds(
            samples=samples,
            n_folds=1,
            data_path=path,
            task_name=name,
            sample_column=sample_column,
        )

def generate_config(num_labels=2, num_inputs=2, num_probe_folds=4, task="ukp-a-similarity", pairwise=False,
                    token_level=False, path=""):
    base_config = {
        "probe_name": "probe-" + task,
        "num_probe_folds": num_probe_folds,
        "num_inputs": num_inputs,
        "num_labels": num_labels,
        "probes_samples_path": path + "/" + task + "/"
    }
    if pairwise and token_level:
        for probe_task_name, probe_task_type in [("bi", PROBE_TASK_TYPES.SENTENCE_PAIR_TOKENS_BI),
                                                 ("cross", PROBE_TASK_TYPES.SENTENCE_PAIR_TOKENS_CROSS)]:
            for control_task_name, control_task_type in [("none", CONTROL_TASK_TYPES.NONE),
                                                         ("perm", CONTROL_TASK_TYPES.PERMUTATION),
                                                         ("rand-weights", CONTROL_TASK_TYPES.RANDOM_WEIGHTS)]:
                config = base_config.copy()
                config["probe_task_type"] = probe_task_type.name
                config["control_task_type"] = control_task_type.name

                os.system("mkdir -p ../../" + path + "/" + task)

                dump_config(config,
                            name="../../" + path + "/" + task + "/config-" + probe_task_name + "-" + control_task_name + ".yaml")
    elif pairwise and not token_level:
        for probe_task_name, probe_task_type in [("bi", PROBE_TASK_TYPES.SENTENCE_PAIR_BI),
                                                 ("cross", PROBE_TASK_TYPES.SENTENCE_PAIR_CROSS)]:
            for control_task_name, control_task_type in [("none", CONTROL_TASK_TYPES.NONE),
                                                         ("perm", CONTROL_TASK_TYPES.PERMUTATION),
                                                         ("rand-weights", CONTROL_TASK_TYPES.RANDOM_WEIGHTS)]:
                config = base_config.copy()
                if probe_task_name == "cross":
                    config["num_inputs"] = 1
                else:
                    config["num_inputs"] = num_inputs

                config["probe_task_type"] = probe_task_type.name
                config["control_task_type"] = control_task_type.name

                dump_config(config,
                            name="../../" + path + "/" + task + "/config-" + probe_task_name + "-" + control_task_name + ".yaml")
    elif not pairwise and token_level:
        for control_task_name, control_task_type in [("none", CONTROL_TASK_TYPES.NONE),
                                                     ("perm", CONTROL_TASK_TYPES.PERMUTATION),
                                                     ("rand-weights", CONTROL_TASK_TYPES.RANDOM_WEIGHTS)]:
            config = base_config.copy()

            config["probe_task_type"] = PROBE_TASK_TYPES.SENTENCE_TOKENS.name
            config["control_task_type"] = control_task_type.name

            dump_config(config, name="../../" + path + "/" + task + "/config-" + control_task_name + ".yaml")
    else:
        for control_task_name, control_task_type in [("none", CONTROL_TASK_TYPES.NONE),
                                                     ("perm", CONTROL_TASK_TYPES.PERMUTATION),
                                                     ("rand-weights", CONTROL_TASK_TYPES.RANDOM_WEIGHTS)]:
            config = base_config.copy()

            config["probe_task_type"] = PROBE_TASK_TYPES.SENTENCE.name
            config["control_task_type"] = control_task_type.name

            dump_config(config, name="../../" + path + "/" + task + "/config-" + control_task_name + ".yaml")



def save_token_classification(samples, name, path, fold_topics=None, fold_samples=None, sample_column="inputs",
                              pairwise=False, token_level=False):
    samples.loc[:, "label"] = samples[sample_column].apply(
        lambda inputs: float(abs(len(re.findall(r'\w+', inputs[0])) - len(re.findall(r'\w+', inputs[1])))))
    samples.loc[:, "org-label"] = samples.loc[:, "label"]

    samples.loc[:, "label"] = scale_labels(samples.loc[:, "label"])

    save_probe(
        samples=samples,
        name=name,
        path=path,
        fold_topics=fold_topics, fold_samples=fold_samples,
        sample_column=sample_column,
        pairwise=pairwise,
        token_level=token_level
    )


def save_token_types_topic_maj(samples, name, path, fold_topics=None, fold_samples=None, sample_column="inputs",
                           pairwise=False,
                           token_level=False, max_samples=40000, is_numeric=False, force_joined_processing=False,
                           force_distinct_processing=False):

    if "rank" in name:
        is_numeric = True

    pandarallel.initialize(progress_bar=True, nb_workers=8, use_memory_fs=False)

    os.system("mkdir -p ../../" + path + name)
    samples[sample_column + "_tokenized"] = samples[sample_column].apply(lambda ele: [
        [token for token in nlp(sentence) if token.text in sentence] for sentence in ele
    ])

    topic_words = get_property_words_majority(samples, "topic", limit=False)


    word_samples = []
    for index, row in samples.iterrows():

        for input_index, tokens in enumerate(row[sample_column + "_tokenized"]):

            word_occurrences = {}

            for token in tokens:
                token_pos = token.pos_
                token_text = token.text.replace('``', '"').replace("''", '"')

                if token_text == " " or token_text in string.punctuation:
                    continue


                word_occurrence = word_occurrences.get(token_text, 0)
                word_occurrences[token_text] = word_occurrence + 1

                label = topic_words.get(token.text.lower(), 0)

                if label > -150000 and token.text not in string.punctuation:
                    start_token = token.idx
                    end_token = token.idx + len(token.text)

                    word_samples.append(
                        {
                            "inputs": ((token_text, input_index, start_token, end_token),),
                            "token_text": token_text,
                            "label": label,
                            "context": tuple(row[sample_column]),
                            "reference": tuple(row[sample_column]),
                            "topic": row["topic"],
                            "pos": token_pos
                        }
                    )

    word_samples = pandas.DataFrame(word_samples)
    word_samples["id"] = word_samples.index

    word_samples.loc[:, "org-label"] = word_samples.loc[:, "label"]

    word_samples["label"] = scale_labels(word_samples["label"])

    save_probe(
        samples=word_samples,
        name=name + "-reg-topic",
        path=path,
        fold_topics=fold_topics, fold_samples=fold_samples,
        sample_column=sample_column,
        pairwise=pairwise,
        token_level=token_level,
        max_samples=max_samples
    )
    estimator = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')

    word_samples["label"] = estimator.fit_transform(word_samples["org-label"].to_numpy().reshape(-1, 1)).reshape(1,-1)[0]
    word_samples["label"] = word_samples["label"].apply(int)

    save_probe(
        samples=word_samples,
        name=name + "-cls-topic",
        path=path,
        fold_topics=fold_topics, fold_samples=fold_samples,
        sample_column=sample_column,
        pairwise=pairwise,
        token_level=token_level,
        max_samples=max_samples
    )


def save_token_types_topics_majority(samples, name, path, fold_topics=None, fold_samples=None, sample_column="inputs",
                            pairwise=False,
                            token_level=False, max_samples=40000, is_numeric=False, force_joined_processing=False,
                            force_distinct_processing=False):

    if "rank" in name:
        is_numeric = True

    pandarallel.initialize(progress_bar=True, nb_workers=8, use_memory_fs=False)

    os.system("mkdir -p ../../" + path + name)
    samples[sample_column + "_tokenized"] = samples[sample_column].apply(lambda ele: [
        [token for token in nlp(sentence) if token.text in sentence] for sentence in ele
    ])

    topic_words = get_property_words_majority(samples, "topic", limit=False)


    word_samples = []
    for index, row in samples.iterrows():
        topic = row["topic"]

        for input_index, tokens in enumerate(row[sample_column + "_tokenized"]):

            word_occurrences = {}

            for token in tokens:
                token_pos = token.pos_
                token_text = token.text.replace('``', '"').replace("''", '"')

                if token_text == " " or token_text in string.punctuation:
                    continue


                word_occurrence = word_occurrences.get(token_text, 0)
                word_occurrences[token_text] = word_occurrence + 1

                label = topic_words.get(token.text.lower(), 0)

                if label > -150000 and token.text not in string.punctuation:
                    start_token = token.idx
                    end_token = token.idx + len(token.text)

                    word_samples.append(
                        {
                            "inputs": ((token_text, input_index, start_token, end_token),),
                            "token_text": token_text,
                            "label": label,
                            "context": tuple(row[sample_column]),
                            "reference": tuple(row[sample_column]),
                            "topic": row["topic"],
                            "pos": token_pos
                        }
                    )

    word_samples = pandas.DataFrame(word_samples)
    word_samples["id"] = word_samples.index

    word_samples.loc[:, "org-label"] = word_samples.loc[:, "label"]

    word_samples["label"] = scale_labels(word_samples["label"])

    save_probe(
        samples=word_samples,
        name=name + "-reg-topic-maj",
        path=path,
        fold_topics=fold_topics, fold_samples=fold_samples,
        sample_column=sample_column,
        pairwise=pairwise,
        token_level=token_level,
        max_samples=max_samples
    )
    estimator = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')

    word_samples["label"] = estimator.fit_transform(word_samples["org-label"].to_numpy().reshape(-1, 1)).reshape(1,-1)[0]
    word_samples["label"] = word_samples["label"].apply(int)

    save_probe(
        samples=word_samples,
        name=name + "-cls-topic-maj",
        path=path,
        fold_topics=fold_topics, fold_samples=fold_samples,
        sample_column=sample_column,
        pairwise=pairwise,
        token_level=token_level,
        max_samples=max_samples
    )

def save_pos(samples, name, path, fold_topics=None, fold_samples=None, sample_column="inputs", pairwise=False,
             token_level=False, max_samples=40000):
    os.system("mkdir -p ../../" + path + name)

    word_samples = []
    for index, row in samples.iterrows():

        topic = row["topic"]

        for input_index, sentence in enumerate(row[sample_column]):
            doc = nlp(sentence)

            token_occurrences = {}

            for token in doc:

                if token.text == " " or token.text == "" or token.text not in sentence:
                    continue

                token_occurrence = token_occurrences.get(token.text, 0)
                token_occurrences[token.text] = token_occurrence + 1

                start = token.idx
                end = token.idx + len(token.text)

                word_samples.append(
                    {
                        "inputs": ((token.text, input_index, start, end),),
                        "string_label": token.pos_,
                        "context": list(row[sample_column]),
                        "reference": tuple(row[sample_column]),
                        "topic": topic
                    }
                )

    word_samples = pandas.DataFrame(word_samples)

    word_samples["id"] = word_samples.index
    word_samples["label"] = pandas.factorize(word_samples["string_label"])[0]
    word_samples.loc[:, "org-label"] = word_samples.loc[:, "string_label"]

    save_probe(
        samples=word_samples,
        name=name,
        path=path,
        fold_topics=fold_topics, fold_samples=fold_samples,
        sample_column=sample_column,
        pairwise=pairwise,
        token_level=token_level,
        max_samples=max_samples
    )


def save_ner(samples, name, path, fold_topics=None, fold_samples=None, sample_column="inputs", pairwise=False,
             token_level=False, max_samples=40000):
    os.system("mkdir -p ../../" + path + name)

    word_samples = []
    for index, row in tqdm(samples.iterrows()):

        topic = row["topic"]

        for input_index, sentence in enumerate(row[sample_column]):
            doc = nlp(sentence)

            for entity in doc.ents:

                if True:# entity.label_ in ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART",
                                     #"LAW"]:

                    if entity.text not in sentence:
                        continue

                    start = list(doc)[entity.start].idx
                    end = start + len(entity.text)

                    word_samples.append(
                        {
                            "inputs": ((entity.text, input_index, start, end),),
                            "string_label": entity.label_,
                            "context": list(row[sample_column]),
                            "reference": tuple(row[sample_column]),
                            "topic": topic
                        }
                    )
    word_samples = pandas.DataFrame(word_samples)

    label_counts = Counter(word_samples["string_label"])
    selected_labels = [label for label, count in label_counts.items() if count > 20]
    word_samples = word_samples[word_samples["string_label"].isin(selected_labels)]

    word_samples["label"] = pandas.factorize(word_samples["string_label"])[0]
    word_samples.loc[:, "org-label"] = word_samples.loc[:, "string_label"]

    word_samples["id"] = word_samples.index
    save_probe(
        samples=word_samples,
        name=name,
        path=path,
        fold_topics=fold_topics, fold_samples=fold_samples,
        sample_column=sample_column,
        pairwise=pairwise,
        token_level=token_level,
        max_samples=max_samples
    )


def save_dependency(samples, name, path, fold_topics=None, fold_samples=None, sample_column="inputs", pairwise=False,
                    token_level=False, max_samples=40000):
    os.system("mkdir -p ../../" + path + name)

    word_samples = []
    for index, row in samples.iterrows():
        topic = row["topic"]

        for input_index, sentence in enumerate(row[sample_column]):
            doc = nlp(sentence)

            for token in doc:

                if token.text == " " or token.dep_ == "ROOT" or token.text not in sentence:
                    continue

                start_head = token.head.idx
                end_head = token.head.idx + len(token.head.text)

                start_token = token.idx
                end_token = token.idx + len(token.text)

                word_samples.append(
                    {
                        "inputs": ((token.head.text, input_index, start_head, end_head),(token.text, input_index, start_token, end_token)),
                        "string_label": token.dep_,
                        "context": list(row[sample_column]),
                        "reference": tuple(row[sample_column]),
                        "topic": topic
                    }
                )

    word_samples = pandas.DataFrame(word_samples)
    word_samples["id"] = word_samples.index

    label_counts = Counter(word_samples["string_label"])
    selected_labels = [label for label, count in label_counts.items() if count > 300]
    word_samples = word_samples[word_samples["string_label"].isin(selected_labels)]

    word_samples["label"] = pandas.factorize(word_samples["string_label"])[0]
    word_samples.loc[:, "org-label"] = word_samples.loc[:, "string_label"]

    save_probe(
        samples=word_samples,
        name=name,
        path=path,
        fold_topics=fold_topics, fold_samples=fold_samples,
        sample_column=sample_column,
        pairwise=pairwise,
        token_level=token_level,
        max_samples=max_samples
    )
