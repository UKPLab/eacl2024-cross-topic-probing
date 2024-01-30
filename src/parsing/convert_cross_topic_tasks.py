import glob
import os
import random

import pandas
import yaml


def load_config(path):
    file_stream = open(path, "r")
    config = yaml.safe_load(file_stream)
    return config

for dataset in ["ukp-argmin", "wtwt"]:
    data_path = "../../probes/" + dataset
    task_folders = list(glob.glob(data_path + "*40*")) + list(glob.glob(data_path + "*80*"))

    for task_folder in task_folders:
        
        print(task_folder)

        if "-in" in task_folder:
            continue

        updated_task_folder = task_folder.replace(dataset, dataset + "-in")

        os.system("mkdir -p " + updated_task_folder)

        for config_file in glob.glob(task_folder + "/*.yaml"):
            config = load_config(config_file)
            config["probe_name"] = config["probe_name"].replace(dataset, dataset + "-in")
            config["probes_samples_path"] = config["probes_samples_path"].replace(dataset, dataset + "-in")
            yaml.dump(config, open(config_file.replace(dataset, dataset + "-in"), 'w'))

        if os.path.exists(task_folder + "/folds.csv"):

            samples = pandas.read_csv(task_folder + "/folds.csv")
            updated_samples = samples.copy()

            sample_ids = list(samples["id"])

            random.Random(0).shuffle(sample_ids)

            start_position = 0

            for fold in range(3):
                train_frame = samples[samples["set-" + str(fold)] == "train"]
                dev_frame = samples[samples["set-" + str(fold)] == "dev"]
                test_frame = samples[samples["set-" + str(fold)] == "test"]

                train_length = train_frame.shape[0]
                dev_length = dev_frame.shape[0]
                test_length = test_frame.shape[0]

                test_ids = sample_ids[start_position:start_position+test_length]
                other_ids = [sample_id for sample_id in sample_ids if sample_id not in test_ids]

                train_ids = other_ids[:train_length]
                dev_ids = other_ids[train_length:]

                start_position = start_position + test_length

                updated_samples.loc[updated_samples["id"].isin(train_ids), "set-" + str(fold)] = "train"
                updated_samples.loc[updated_samples["id"].isin(dev_ids), "set-" + str(fold)] = "dev"
                updated_samples.loc[updated_samples["id"].isin(test_ids), "set-" + str(fold)] = "test"

            updated_samples.to_csv(updated_task_folder + "/folds.csv")
        else:
            for fold_file in glob.glob(task_folder + "/fold_*"):
                fold = int(fold_file.split("_")[-1].split(".")[0])

                samples = pandas.read_csv(fold_file)

                train_frame = samples[samples["set-" + str(fold)] == "train"]
                dev_frame = samples[samples["set-" + str(fold)] == "dev"]
                test_frame = samples[samples["set-" + str(fold)] == "test"]

                train_length = train_frame.shape[0]
                dev_length = dev_frame.shape[0]
                test_length = test_frame.shape[0]




