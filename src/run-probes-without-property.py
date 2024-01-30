import os
import uuid
from collections import defaultdict

import click
from mlflow.tracking import MlflowClient

from defs.default_config import MLFLOW_URL
from utils.dropbox_util import download_dropbox_file


def get_params(run):
    config = run.data.params
    return (config["control_task_type"], config["probe_task_type"], config["model_name"], config["seed"], config["fold"], config["num_hidden_layers"])

@click.command()
@click.option('--amnesic_experiment', type=str, default="probes-amnesic-ukp-argmin-token-types-40-cls-topic-maj")
@click.option('--target_experiment', type=str, default="probes-ukp-argmin-ner")
@click.option('--model', type=str, default="bert-base-uncased")
def main(amnesic_experiment, target_experiment, model):
    client = MlflowClient(MLFLOW_URL)

    amnesic_experiment = client.get_experiment_by_name(amnesic_experiment)
    target_amnesic_experiment = client.get_experiment_by_name(target_experiment.replace("probes", "probes-amnesic"))
    target_experiment = client.get_experiment_by_name(target_experiment)

    amnesic_runs = client.search_runs(amnesic_experiment.experiment_id, filter_string="params.scalar_mix='False' and params.hidden_dim='0' and metrics.epoch > 18 and params.model_name='" + model + "'")
    target_amnesic_runs = client.search_runs(target_amnesic_experiment.experiment_id, filter_string="params.scalar_mix='False' and params.hidden_dim='0' and metrics.epoch > 18  and params.model_name='" + model + "'")


    mapped_runs = defaultdict(lambda: defaultdict(list))
    mapped_target_amnesic_runs = defaultdict(lambda: defaultdict(list))

    for run in amnesic_runs:
        run_params = get_params(run)

        if run.data.params["amnesic"] == "debias":
            mapped_runs[run_params]["debias"].append(run)
        if run.data.params["amnesic"] == "rand":
            mapped_runs[run_params]["rand"].append(run)

    for run in target_amnesic_runs:
        run_params = get_params(run)

        if "debias-" in run.data.params["amnesic"] or "rand-" in run.data.params["amnesic"]:
            mapped_target_amnesic_runs[run_params][run.data.params["amnesic"]].append(run)

    for params, runs in mapped_runs.items():
        for method in ["debias"]:
            control_task_type, probe_task_type, model_name, seed, fold, num_hidden_layers = params

            amnesic_property = method + "-" + amnesic_experiment.name

            if control_task_type in ["PERMUTATION", "RANDOM_WEIGHTS"]:
                continue

            if len(mapped_target_amnesic_runs[params][amnesic_property]) > 0:
                continue

            debias_runs = runs[method]

            if len(debias_runs) == 0:
                continue

            amnesic_dump_id = debias_runs[0].data.params["dump_id"]

            source_path = "/probes-amnesic-results/" + amnesic_dump_id + "/"
            destination_path = "./tmp/" + str(uuid.uuid4()) + "-amnesic-" + amnesic_dump_id + "/"
            file = "P_" + method + ".npy"

            os.system("mkdir -p " + destination_path)

            try:
                amnesic_file = download_dropbox_file(source_path, destination_path, file)
            except:
                continue

            amnesic_file = amnesic_file + file

            if control_task_type == "NONE":
                config_file_path = "../probes/" + target_experiment.name.replace("probes-", "") + "/config-none.yaml"
            elif control_task_type == "PERMUTATION":
                continue
            elif control_task_type == "RANDOM_WEIGHTS":
                continue
            else:
                continue

            command = "python3 run.py --seeds " + str(seed) + " --model_name " + model + " --fold " + str(fold) + " --num_hidden_layers " + num_hidden_layers + " --config_file_path " + config_file_path + " --clean_run True --scalar_mix True --run_amnesic False --run_default True --run_mdl True --amnesic_property " + amnesic_property + " --amnesic_file " + amnesic_file

            print(command)

            os.system(command)

            os.system("rm -rf " + destination_path)



if __name__ == "__main__":
    main()


