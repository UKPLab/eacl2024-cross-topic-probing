import glob
import re

import numpy
import pandas
from sklearn.model_selection import train_test_split

from parsing.composition_util import decontracted


def get_topic_splits(samples: pandas.DataFrame, n_folds: int = 4):
    fold_topics = []

    topics = list(sorted(samples["topic"].unique()))

    test_topic_splits = numpy.array_split(range(len(topics)), n_folds)

    target_length = len(test_topic_splits[0])

    for fold, test_topic_ids in enumerate(test_topic_splits):

        if len(test_topic_ids) != target_length:

            diff = target_length - len(test_topic_ids)

            topic_ids_to_exclude = test_topic_splits[fold - 1][-diff:]
            topics_to_exclude = [topics[test_topic_id] for test_topic_id in topic_ids_to_exclude]
        else:
            topics_to_exclude = []

        test_topics = [topics[test_topic_id] for test_topic_id in test_topic_ids]

        train_topics = sorted([topic for topic in topics if topic not in test_topics and topic not in topics_to_exclude])

        train_topics, dev_topics = train_test_split(train_topics, train_size=0.8, random_state=fold)

        fold_topics.append([train_topics, dev_topics, list(test_topics)])

    return fold_topics

def load_wtwt():

    data_path = "../../data/wtwt/wtwt_with_text.json"

    samples = pandas.read_json(data_path)

    samples = samples.reset_index()

    label_mapping = {
        "unrelated": 0,
        "comment": 1,
        "support": 2,
        "refute": 3,
    }

    samples.rename(columns={"text": "inputs", "merger":"topic", "stance":"label"}, inplace=True)
    samples["inputs"] = samples["inputs"].apply(lambda ele: re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', ' ', ele, flags=re.MULTILINE))
    samples["label"] = samples["label"].apply(lambda label: label_mapping[label])
    samples["inputs"] = samples["inputs"].apply(decontracted).apply(lambda sentence: (sentence,))
    samples = samples[["inputs", "topic", "label"]]

    topic_samples = samples.copy()[["inputs", "topic"]]
    topic_samples.columns = ["inputs", "topic"]

    samples["id"] = samples.index
    samples["context"] = ""

    return samples, topic_samples


def load_ukp_sent():
    data_path = "../../data/UKP_sentential_argument_mining/data"

    samples = pandas.concat([
        pandas.read_csv(file, delimiter="\t", quotechar="'")
        for file in glob.glob(data_path + "/*")
    ])

    samples["sentence"] = samples["sentence"].apply(decontracted)

    samples = samples.reset_index()

    label_mapping = {
        "NoArgument": 0,
        "Argument_for": 1,
        "Argument_against": 2,
    }

    topic_samples = samples.copy()[["sentence", "topic"]]
    topic_samples.columns = ["inputs", "topic"]

    samples["label"] = samples["annotation"].apply(lambda label: label_mapping[label])
    samples["inputs"] = samples["sentence"].apply(lambda sentence: (sentence,))

    samples = samples[["topic", "inputs", "label"]]
    samples["id"] = samples.index
    samples["context"] = ""
    return samples, topic_samples


DATA_LOADER = {
    "ukp-argmin": load_ukp_sent,
    "wtwt": load_wtwt,
}
