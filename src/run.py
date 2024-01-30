import gc
import itertools
import os
from itertools import product
from multiprocessing.pool import Pool
from typing import Dict

import click
import pandas
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer
from utils.config_loading import load_config
from utils.data_loading import load_folds, batching_collate
from utils.experiment_util import init_random_weights, clean_up_runs, clean_up_amnesic_runs

from defs.control_task_types import CONTROL_TASK_TYPES
from defs.default_config import MLFLOW_URL
from defs.probe_task_types import PROBE_TASK_TYPES
from model.BartTransformer import BartTransformer
from model.ParallelSentenceTransformer import ParallelSentenceTransformer
from model.SpecificLayerPooling import SpecificLayerPooling
from probing_worker import ProbeWorker


def get_hyperparameters(hyperparameters:Dict):
    params = [dict(zip(hyperparameters, v)) for v in product(*hyperparameters.values())]
    params = [
        param
        for param in params
        if not (param["num_hidden_layers"] == 0 and param["hidden_dim"] > 0) and not (param["num_hidden_layers"] > 0 and param["hidden_dim"] == 0)
    ]
    return params


def run_scalar_mix_model(args):
    hyperparameter, config, word_embedding_dimension, probing_frames, mlflow_url, run_mdl, folds, run_amnesic, run_stacked_amnesic, run_rlace, run_default, run_mdl, sync_mdl, amnesic_property, amnesic_file, topic_splits = args

    if amnesic_file != "" or run_default == False or hyperparameter["scalar_mix"] == False:
        return

    hyperparameter["control_task_type"] = config["control_task_type"].name
    hyperparameter["probe_task_type"] = config["probe_task_type"].name
    hyperparameter["num_labels"] = config["num_labels"]
    hyperparameter["model_name"] = config["model_name"]
    hyperparameter["input_dim"] = int(config["num_inputs"] * word_embedding_dimension)
    hyperparameter["scalar_mix"] = True

    joined_frames = []

    for probing_frame in probing_frames:
        joined_frame = list(probing_frame.values())[0].copy()
        joined_frame["inputs_encoded"] = [
            [probing_frame[layer].loc[index,"inputs_encoded"] for layer in probing_frame.keys()]
            for index, row in joined_frame.iterrows()
        ]
        joined_frame["inputs_encoded"] = joined_frame["inputs_encoded"].apply(batching_collate)
        joined_frames.append(joined_frame)

    project_prefix = config["probes_samples_path"].split("/")[0]

    worker = ProbeWorker(
        probing_frames=joined_frames, hyperparameter=hyperparameter, run_mdl=run_mdl, project_prefix=project_prefix,
        probe_name=config["probes_samples_path"].split("/")[-2], mlflow_url=mlflow_url, folds=folds, sync_mdl=sync_mdl,
        topic_splits=topic_splits
    )
    worker.run_tasks(run_amnesic=False, run_stacked_amnesic=False, run_rlace=run_rlace, run_default=True, run_mdl=run_mdl)


def run_top_model(args):
    hyperparameter, config, word_embedding_dimension, probing_frames, mlflow_url, run_md, folds, run_amnesic, run_stacked_amnesic, run_rlace, run_default, run_mdl, sync_mdl, amnesic_property, amnesic_file, topic_splits = args
    hyperparameter = hyperparameter.copy()
    hyperparameter["control_task_type"] = config["control_task_type"].name
    hyperparameter["probe_task_type"] = config["probe_task_type"].name
    hyperparameter["num_labels"] = config["num_labels"]
    hyperparameter["model_name"] = config["model_name"]
    hyperparameter["input_dim"] = int(config["num_inputs"] * word_embedding_dimension)
    hyperparameter["scalar_mix"] = False
    joined_frames = []

    for probing_frame in probing_frames:
        last_key = list(probing_frame.keys())[-1]
        joined_frame = list(probing_frame.values())[0].copy()
        joined_frame["inputs_encoded"] = [
            probing_frame[last_key].loc[index,"inputs_encoded"]
            for index, row in joined_frame.iterrows()
        ]
        joined_frame["inputs_encoded"] = joined_frame["inputs_encoded"].apply(batching_collate)
        joined_frame["inputs_encoded"] = joined_frame["inputs_encoded"].apply(lambda ele: ele.flatten().unsqueeze(dim=0))
        joined_frames.append(joined_frame)

    project_prefix = config["probes_samples_path"].split("/")[0]

    worker = ProbeWorker(
        probing_frames=joined_frames, hyperparameter=hyperparameter, run_mdl=run_mdl, project_prefix=project_prefix,
        probe_name=config["probes_samples_path"].split("/")[-2], mlflow_url=mlflow_url, folds=folds, sync_mdl=sync_mdl,
        topic_splits=topic_splits
    )
    worker.run_tasks(run_amnesic=run_amnesic, run_stacked_amnesic=run_stacked_amnesic, run_rlace=run_rlace, run_default=run_default, run_mdl=run_mdl, amnesic_property=amnesic_property, amnesic_file=amnesic_file)



def run_param(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    run_scalar_mix_model(args)
    run_top_model(args)

    return None

def get_topic_split(frame, col):

    train_topics = list(frame[frame[col] == "train"]["topic"].unique())
    dev_topics = list(frame[frame[col] == "dev"]["topic"].unique())
    test_topics = list(frame[frame[col] == "test"]["topic"].unique())

    return [train_topics, dev_topics, test_topics]

def get_topic_splits(probing_frames, task_path):
    if "-in" in task_path:
        probing_frame = pandas.read_csv(task_path + "/folds.csv")
    else:
        probing_frame = list(probing_frames[0].values())[0]

    split_columns = sorted([col for col in probing_frame.columns if "set-" in col])

    topic_splits = [
        get_topic_split(probing_frame, col)
        for col in split_columns
    ]
    return topic_splits

model_layers = {
    "bert-base-uncased": 13,
    "bert-large-uncased": 13,
    "albert-base-v2": 13,
    "albert-large-v2": 25,
    "albert-xxlarge-v2": 13,
    "roberta-base": 13,
    "facebook/bart-base": 13,
    "facebook/bart-large": 25,
    "microsoft/deberta-base": 13,
    "microsoft/deberta-large": 25,
    "microsoft/deberta-xlarge": 48,
    "microsoft/deberta-v3-base": 13,
    "gpt2": 13,
    "google/electra-base-discriminator": 13,
}


all_tasks = ["wtwt", "ukp-argmin"]

@click.command()
@click.option('--config_file_path', type=str, default='../probes/ukp-argmin-token-types-80-cls-topic-maj/config-none.yaml')
@click.option('--task_name', type=str, default="probes-ukp-argmin-ner")
@click.option('--start_layer', type=int, default=0)
@click.option('--end_layer', type=int)
@click.option('--specific_layer', type=int)
@click.option('--n_cpus', type=int, default=1)
@click.option('--fold', type=int, default=-1)
@click.option('--num_hidden_layers', type=str, default="0")
@click.option('--seeds', type=str, default="0,1,2")
@click.option('--n_jobs', type=int, default=1)
@click.option('--batch_size', type=int)
@click.option('--pooling', type=str, default="mean")
@click.option('--control_task_type', type=str)
@click.option('--model_name', type=str, default="bert-base-uncased")
@click.option('--scalar_mix', type=bool, default=False)
@click.option('--fill_up', type=bool, default=False)
@click.option('--clean_run', type=bool, default=True)
@click.option('--topic', type=bool, default=False)
@click.option('--amnesic_property', type=str, default='')
@click.option('--amnesic_file', type=str, default='')
@click.option('--run_amnesic', type=bool, default=False)
@click.option('--run_stacked_amnesic', type=bool, default=False)
@click.option('--run_rlace', type=bool, default=False)
@click.option('--run_default', type=bool, default=True)
@click.option('--run_mdl', type=bool, default=True)
@click.option('--sync_mdl', type=bool, default=True)
def main(config_file_path:str, task_name,  start_layer, end_layer, specific_layer, n_cpus, fold, num_hidden_layers, seeds, n_jobs, batch_size, pooling, control_task_type, model_name, scalar_mix, fill_up, clean_run, topic, amnesic_property, amnesic_file,  run_amnesic, run_stacked_amnesic, run_rlace, run_default, run_mdl, sync_mdl):
    config = load_config(config_file_path)
    seeds = [int(seed) for seed in seeds.split(",")]
    if not end_layer:
        end_layer = model_layers.get(model_name, 13)


    if specific_layer is not None:
        start_layer = specific_layer
        end_layer = specific_layer + 1
        config["specific_layer"] = specific_layer
        config["hyperparameters"]["specific_layer"] = [specific_layer]

    config["hyperparameters"]["seed"] = seeds
    config["hyperparameters"]["scalar_mix"] = [scalar_mix]
    if scalar_mix:
        config["hyperparameters"]["mix_size"] = [end_layer - start_layer]

    if task_name:
        config["task_name"] = task_name
    else:
        config["task_name"] = [task for task in all_tasks if task in config_file_path][0]
        if "-in" in config_file_path:
            config["task_name"] += "-in"

    if model_name:
        config["model_name"] = model_name

    num_hidden_layers = [int(ele) for ele in num_hidden_layers.split(",")]

    if not num_hidden_layers is None:
        config["hyperparameters"]["num_hidden_layers"] = num_hidden_layers

    if not batch_size is None:
        config["hyperparameters"]["batch_size"] = [batch_size]
    if control_task_type:
        config["control_task_type"] = CONTROL_TASK_TYPES(control_task_type)

    if pooling == "cls":
        config["task_name"] = config["task_name"].replace("probe", "probe-cls")

    if topic:
        config["probe_task_type"] = PROBE_TASK_TYPES("TOPIC_" + config["probe_task_type"].name)

    layers = list(range(start_layer, end_layer))


    mlflow_url = MLFLOW_URL

    if fold > -1:
        folds = [fold]
    else:
        folds = range(config["num_probe_folds"])

    if clean_run:
        if amnesic_property != "":
            skip_run, folds_to_do = clean_up_amnesic_runs(mlflow_url, config, folds=folds,  project_prefix=config["probes_samples_path"].split("/")[0], amnesic_property=amnesic_property)
        else:
            skip_run, folds_to_do = clean_up_runs(mlflow_url, config, folds=folds, project_prefix=config["probes_samples_path"].split("/")[0])

        if skip_run:
            print("skip runs")
            return None

        folds = folds_to_do



    if "glove" in config["model_name"]:
        base_model = SentenceTransformer(config["model_name"])
        word_embedding_dimension = 300
    else:
        if "bart" in config["model_name"]:
            transformer = BartTransformer(config["model_name"], model_args={"output_hidden_states": True})
        else:
            transformer = Transformer(config["model_name"], model_args={"output_hidden_states": True})

        if config["control_task_type"] == CONTROL_TASK_TYPES.RANDOM_WEIGHTS and config["model_name"] in ["roberta-base", "microsoft/deberta-base", "microsoft/deberta-v3-base", "bert-base-uncased", "albert-base-v2", "google/electra-base-discriminator"]:
            transformer.auto_model.encoder.apply(init_random_weights)
        elif config["control_task_type"] == CONTROL_TASK_TYPES.RANDOM_WEIGHTS and config["model_name"] in ["facebook/bart-base"]:
            transformer.auto_model.encoder.apply(init_random_weights)
            transformer.auto_model.decoder.apply(init_random_weights)
        elif config["control_task_type"] == CONTROL_TASK_TYPES.RANDOM_WEIGHTS and config["model_name"] in ["gpt2"]:
            transformer.auto_model.h.apply(init_random_weights)


        word_embedding_dimension = transformer.get_word_embedding_dimension()

        pooling = SpecificLayerPooling(
            word_embedding_dimension=word_embedding_dimension,
            layers=layers, pooling_mode=pooling
        )
        base_model = ParallelSentenceTransformer(modules=[transformer,pooling])


    torch.set_num_threads(1)

    probing_frames = load_folds(
        base_path = "../" + config["probes_samples_path"],
        folds=folds,
        base_model=base_model,
        probe_task_type=config["probe_task_type"],
        control_task_type=config["control_task_type"],
        #sample_size=1000
    )

    topic_splits = get_topic_splits(probing_frames=probing_frames, task_path="../" + config["probes_samples_path"])

    if not topic_splits:
        sync_mdl = False

    del base_model

    if "glove" not in config["model_name"]:
        del transformer
        del pooling

    torch.cuda.empty_cache()
    gc.collect()

    hyperparameters = get_hyperparameters(config["hyperparameters"])

    params = list(itertools.chain.from_iterable(
        [
            (hyperparameter, config, word_embedding_dimension, probing_frames, mlflow_url, True, [fold], run_amnesic, run_stacked_amnesic, run_rlace, run_default, run_mdl, sync_mdl, amnesic_property, amnesic_file, topic_splits)
            for hyperparameter in hyperparameters
        ]
        for fold in folds
    ))
    if n_jobs == 1 or len(params) == 1:
        for param in params:
            run_param(param)

    else:
        pool = Pool(min(len(params), n_jobs))

        pool.map(run_param, params)




if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    main()
