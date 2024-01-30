import glob
import os

import click
import torch


def run_probe(probe_name, model_path, seed, fold, topic=False, amnesic=False):

    probe_config_file = "../probes/" + probe_name + "/config-none.yaml"

    evolution_probe_name = "probes-evolution-" + probe_name

    command = "python3 run.py --topic " + str(topic) + " --task_name " + evolution_probe_name + " --n_jobs 4 --seeds " + str(seed) + " --fold " + str(fold) + " --num_hidden_layers 0 --config_file_path " + probe_config_file + " --scalar_mix False --model_name " + model_path + "/ --clean_run True --run_amnesic " + str(amnesic) + " --run_default True --run_mdl True --scalar_mix True"

    os.system(command)

    print("done")

@click.command()
@click.option('--task', type=str, default="wtwt")
@click.option('--model_name', type=str, default="microsoft/deberta-base")
@click.option('--seed', type=int, default=0)
@click.option('--fold', type=int, default=0)
def main(task, model_name, seed, fold):


    run_name = "evolution-" + task + "-fold_" + str(fold) + "-seed_" + str(seed) + "-" + model_name

    os.system("mkdir -p ./" + run_name.replace("/", "_"))

    model_files = glob.glob("./model_name/*.bin")

    for model_file in model_files:
        model_folder = "/".join(model_file.split("/")[:-1]) + "/"

        short_model_folder = "/".join(model_file.split("/")[:2])

        os.system("mv " + model_folder + "* " + short_model_folder)
        os.system("find " + short_model_folder + " -maxdepth 1 -mindepth 1 -type d -exec rm -rf '{}' \;")

        run_probe(task + "-ner", short_model_folder, seed, fold)
        run_probe(task, short_model_folder, seed, fold)
        run_probe(task + "-pos", short_model_folder, seed, fold)
        run_probe(task + "-dependency", short_model_folder, seed, fold)

        os.system("rm -rf ./" + run_name + "*")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
