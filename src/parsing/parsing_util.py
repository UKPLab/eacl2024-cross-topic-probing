import itertools
import os
import random
from typing import List

import numpy
import pandas
from sklearn.model_selection import train_test_split


def get_samples_for_topic(samples: pandas.DataFrame, topics: List[str]) -> pandas.DataFrame:
    return samples[samples["topic"].str.lower().isin([topic.lower() for topic in topics])]

def get_fold_frame(samples: pandas.DataFrame, train_topics: List[str], dev_topics: List[str], test_topics: List[str],  sample_column: List[str]) -> pandas.DataFrame:
    train_samples = get_samples_for_topic(samples, train_topics)
    dev_samples = get_samples_for_topic(samples, dev_topics)
    test_samples = get_samples_for_topic(samples, test_topics)

    train_samples.loc[:,"set"] = "train"
    dev_samples.loc[:,"set"] = "dev"
    test_samples.loc[:,"set"] = "test"

    train_samples = train_samples
    dev_samples = dev_samples
    test_samples = test_samples

    return pandas.concat([train_samples, dev_samples, test_samples])

def get_fold_splits(samples: pandas.DataFrame, train_topics: List[str], dev_topics: List[str], test_topics: List[str],  sample_column: List[str]) -> pandas.DataFrame:
    train_samples = get_samples_for_topic(samples, train_topics)
    dev_samples = get_samples_for_topic(samples, dev_topics)
    test_samples = get_samples_for_topic(samples, test_topics)

    samples["set"] = ""
    samples.loc[train_samples["id"],"set"] = "train"
    samples.loc[dev_samples["id"],"set"] = "dev"
    samples.loc[test_samples["id"],"set"] = "test"

    return samples

def save_topic_folds(samples: pandas.DataFrame, fold_topics: List[List[str]], data_path: str, task_name: str, sample_column: List[str]):

    os.system("mkdir -p ../../" + data_path +"/" + task_name)
    samples = samples.sort_values("id")
    samples.loc[:, "id"] = samples["id"].apply(int)
    for i, (train_topics, dev_topics, test_topics) in enumerate(fold_topics):
        fold_frame = get_fold_splits(samples, train_topics, dev_topics, test_topics, sample_column)
        samples["set-" + str(i)] = fold_frame.drop_duplicates("id")["set"]

    relevant_columns = ["id"] + [sample_column, "context", "label", "topic"] + [col for col in samples.columns if "set-" in col]
    if "org-label" in fold_frame.columns:
        relevant_columns.append("org-label")

    samples[relevant_columns].to_csv("../../" + data_path +"/" + task_name + "/folds.csv", index=False)

def save_folds(samples: pandas.DataFrame, n_folds:int, data_path: str, task_name: str, sample_column: List[str], fold_samples=None):
    os.system("mkdir -p ../../" + data_path + "/" + task_name)
    samples = samples.sample(frac=1, random_state=0)
    samples.loc[:, "id"] = samples["id"].apply(int)



    if n_folds == 1:
        train_samples, test_samples = train_test_split(samples, train_size=0.8, random_state=0)
        train_samples, dev_samples = train_test_split(train_samples, train_size=0.8, random_state=0)

        train_samples["set-0"] = "train"
        dev_samples["set-0"] = "dev"
        test_samples["set-0"] = "test"

        fold_frame = pandas.concat([train_samples, dev_samples, test_samples])
        fold_frame[["id"]  + [sample_column, "context", "label", "set-0", "topic"]].to_csv("../../" + data_path + task_name + "/folds.csv", index=False)

    elif fold_samples:
        all_fold_frames = []
        for fold, (train_samples, dev_samples, test_samples) in enumerate(fold_samples):

            if "reference" not in samples.columns:
                reference_sample_column = sample_column
            elif type(samples[sample_column].iloc[0]) == list or type(samples[sample_column].iloc[0]) == tuple:
                reference_sample_column = "reference"
            elif "synonym-antonym" in task_name:
                reference_sample_column = "reference"
            else:
                reference_sample_column = sample_column

            if "synonym-antonym" in task_name:
                train_samples = set(itertools.chain.from_iterable(train_samples))
                dev_samples = set(itertools.chain.from_iterable(dev_samples))
                test_samples = set(itertools.chain.from_iterable(test_samples))

            elif type(train_samples) != set:
                train_samples = [tuple(ele) for ele in train_samples]
                dev_samples = [tuple(ele) for ele in dev_samples]
                test_samples = [tuple(ele) for ele in test_samples]

            elif type(train_samples) == set:
                train_samples = [tuple([ele]) for ele in train_samples]
                dev_samples = [tuple([ele]) for ele in dev_samples]
                test_samples = [tuple([ele]) for ele in test_samples]


            train_frame = samples[samples.apply(lambda row: row[reference_sample_column] in train_samples, axis=1)]
            dev_frame = samples[samples.apply(lambda row: row[reference_sample_column] in dev_samples, axis=1)]
            test_frame = samples[samples.apply(lambda row: row[reference_sample_column] in test_samples, axis=1)]

            train_frame["set-" + str(fold)] = "train"
            dev_frame["set-" + str(fold)] = "dev"
            test_frame["set-" + str(fold)] = "test"

            fold_frame = pandas.concat([train_frame, dev_frame, test_frame]).sort_values("id").drop_duplicates("id")
            all_fold_frames.append(fold_frame)

        samples = samples.sort_values("id")
        for fold, fold_frame in enumerate(all_fold_frames):
            samples["set-" + str(fold)] = fold_frame["set-" + str(fold)].values

        samples[["id"]  +  [sample_column, "context", "label", "topic"] + ["set-" + str(i) for i in range(len(fold_samples))]].to_csv("../../" + data_path + "/" + task_name  + "/folds.csv", index=False)
    else:
        all_fold_frames = []
        for fold, test_samples in enumerate(numpy.array_split(samples, n_folds)):
            train_samples = pandas.merge(samples, test_samples, indicator=True, how="outer")
            train_samples = train_samples[train_samples["_merge"] == "left_only"]
            train_samples, dev_samples = train_test_split(train_samples, train_size=0.8, random_state=fold)

            train_samples["set-" + str(fold)] = "train"
            dev_samples["set-" + str(fold)] = "dev"
            test_samples["set-" + str(fold)] = "test"

            fold_frame = pandas.concat([train_samples, dev_samples, test_samples]).sort_values("id").drop_duplicates("id")
            all_fold_frames.append(fold_frame)

        samples = samples.sort_values("id")
        samples = samples[samples["id"].isin(all_fold_frames[0]["id"])]
        for fold, fold_frame in enumerate(all_fold_frames):
            samples["set-" + str(fold)] = fold_frame["set-" + str(fold)]

        samples[["id"]  + [sample_column, "context", "label", "topic"] + ["set-" + str(i) for i in range(len(fold_samples))]].to_csv("../../" + data_path + "/" + task_name  + "/folds.csv", index=False)

def save_topic_step_folds(samples: pandas.DataFrame, fold_topics: List[List[str]], data_path: str, task_name: str, sample_column: List[str], step_size:float):
    samples.loc[:, "id"] = samples["id"].apply(int)
    for i, (train_topics, dev_topics, test_topics) in enumerate(fold_topics):

        step_samples = pandas.concat([
            topic_frame.sample(step_size, random_state=0) if topic in train_topics else topic_frame
            for topic, topic_frame in samples.groupby("topic")

        ])

        step_samples = step_samples.reset_index(drop=True)

        fold_frame = get_fold_frame(step_samples, train_topics, dev_topics, test_topics, sample_column)
        fold_frame[["id"] + [sample_column, "label", "set", "topic"]].to_csv("../../probes/" + task_name + "_fold_" + str(i) + ".csv", index=False)





def get_random_topic_folds(samples: pandas.DataFrame, n_folds: int = 4):
    fold_topics  = []

    topics = list(set(samples["topic"]))

    for fold, test_topics in enumerate(numpy.array_split(topics, n_folds)):

        random.seed(fold)

        train_topics = [topic for topic in topics if topic not in test_topics]

        train_topics, dev_topics = train_test_split(train_topics, train_size=0.8, random_state=fold)

        fold_topics.append([train_topics, dev_topics, list(test_topics)])

    return fold_topics


def get_close_topic_folds(samples: pandas.DataFrame, n_folds: int = 4):
    fold_topics  = []

    topics = list(set(samples["topic"]))


    for fold, test_topics in enumerate(numpy.array_split(topics, n_folds)):

        random.seed(fold)

        train_topics = [topic for topic in topics if topic not in test_topics]

        train_topics, dev_topics = train_test_split(list(set(train_topics)), train_size=0.8, random_state=fold)

        fold_topics.append([train_topics, dev_topics, list(test_topics)])

    return fold_topics