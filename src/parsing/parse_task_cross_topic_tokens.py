import argparse

from parsing.composition_util import save_token_types_topics_majority
from parsing.data_loader import DATA_LOADER, get_causes, get_topic_splits

data_path = "probes/"

causes, filtered_causes = get_causes()

token_level_probes = [
    (save_token_types_topics_majority, "token-types"),
]
sentence_level_probes = [
]

pairwise_probes = [
]


def parse_probes(samples, topic_samples, task):
    fold_topics = get_topic_splits(samples, n_folds=3)

    for parse_function, probe_name in token_level_probes:
        parse_function(
            samples=samples.copy(),
            name=task + "-" + probe_name + "-40",
            path=data_path,
            fold_topics=fold_topics,
            sample_column="inputs",
            token_level=True,
            pairwise=False,
            max_samples=40000
        )
        parse_function(
            samples=samples.copy(),
            name=task + "-" + probe_name + "-80",
            path=data_path,
            fold_topics=fold_topics,
            sample_column="inputs",
            token_level=True,
            pairwise=False,
            max_samples=80000
        )

def parse_in_topic_probes(samples, topic_samples, task):
    for parse_function, probe_name in token_level_probes:
        parse_function(
            samples=samples.copy(),
            name=task + "-" + probe_name + "-40",
            path=data_path,
            sample_column="inputs",
            token_level=True,
            pairwise=False,
            max_samples=40000
        )
        parse_function(
            samples=samples.copy(),
            name=task + "-" + probe_name + "-80",
            path=data_path,
            sample_column="inputs",
            token_level=True,
            pairwise=False,
            max_samples=80000
        )


PARSER = {
    "ukp-argmin": parse_probes,
    "all-news": parse_in_topic_probes,
    "20-news": parse_in_topic_probes,
    "wos": parse_in_topic_probes,
    "pstance": parse_probes,
    "semevalt6": parse_probes,
    "evi-sen": parse_probes,
    "evi-sen-big": parse_probes,
    "essay-type": parse_probes,
    "arg-q-rank": parse_probes,
    "arg-q": parse_probes,
    "arg-conv-rank": parse_probes,
    "wtwt": parse_probes,
}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='all-news')
    args = parser.parse_args()
    task = args.task

    samples, topic_samples = DATA_LOADER[task]()

    PARSER[task](samples, topic_samples, task)
