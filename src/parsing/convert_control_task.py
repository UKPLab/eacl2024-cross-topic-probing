import glob
import os

import numpy
import pandas
import yaml


def load_config(path):
    file_stream = open(path, "r")
    config = yaml.safe_load(file_stream)
    return config

for dataset in ["wtwt", "ukp-argmin"]:
    data_path = "../../probes/" + dataset + "*"
    task_folders = list(glob.glob(data_path))

    for task_folder in task_folders:

        if "rand-y" in task_folder:
            continue

        print(task_folder)
        updated_task_folder = task_folder + "-rand-y"

        os.system("mkdir -p " + updated_task_folder)

        for config_file in glob.glob(task_folder + "/*.yaml"):
            config = load_config(config_file)
            config["probe_name"] = config["probe_name"] + "-rand-y"
            config["probes_samples_path"] = config["probes_samples_path"][:-1] + "-rand-y/"
            yaml.dump(config, open(config_file.replace(task_folder, task_folder + "-rand-y"), 'w'))

        samples = pandas.read_csv(task_folder + "/folds.csv")
        samples["label"] = numpy.random.permutation(samples["label"].values)
        samples.to_csv(updated_task_folder + "/folds.csv")