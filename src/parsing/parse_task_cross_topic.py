import argparse

from parsing.composition_util import save_ner, save_dependency, save_pos, save_token_types_topics_majority, save_probe
from parsing.data_loader import DATA_LOADER, get_topic_splits

data_path = "probes/"


token_level_probes = [
   (save_ner, "ner"), (save_dependency, "dependency"),  (save_pos, "pos"), (save_token_types_topics_majority, "token-types"),
]


def parse_probes(samples, task):
    fold_topics = get_topic_splits(samples, n_folds=3)


    for parse_function, probe_name in token_level_probes:
        parse_function(
            samples=samples.copy(),
            name=task + "-" + probe_name,
            path=data_path,
            fold_topics=fold_topics,
            sample_column="inputs",
            token_level=True,
            pairwise=False,
            max_samples=40000
        )

    save_probe(
        samples=samples.copy(),
        name=task,
        path=data_path,
        fold_topics=fold_topics,
        sample_column="inputs",
        pairwise=False,
        token_level=False
    )


PARSER = {
    "ukp-argmin": parse_probes,
}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='ukp-argmin')
    args = parser.parse_args()
    task = args.task

    samples, _ = DATA_LOADER[task]()

    PARSER[task](samples, task)
