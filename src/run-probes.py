import glob
import os

import click


# python3 run-probes.py --task arg-conv-rank-in --model bert-base-uncased
@click.command()
@click.option('--task', type=str, default="ukp-argmin")
@click.option('--model', type=str, default="bert-base-uncased")
@click.option('--num_hidden_layers', type=str, default="0")
@click.option('--control_task_type', type=str, default="")
@click.option('--seeds', type=str, default="0,1,2")
@click.option('--in_filter', type=str, default="")
@click.option('--out_filter', type=str, default="")
@click.option('--scalar_mix', type=bool, default=True)
@click.option('--mdl', type=bool, default=True)
@click.option('--amnesic', type=bool, default=False)
@click.option('--specific_layer', type=int)
def main(task, model, num_hidden_layers, control_task_type, seeds, in_filter, out_filter, scalar_mix, mdl, amnesic, specific_layer):
    failed_runs = []

    for file in sorted(glob.glob("../probes/" + task + "*/*" + control_task_type + "*.yaml")):

        if "rand-y" not in in_filter and "rand-y" in file:
            continue

        if in_filter != "" or out_filter != "":
            if len(out_filter) > 0 and out_filter in file:
                continue

            if len(in_filter) > 0 and in_filter not in file:
                continue

        if task + "-in" in file or "const" in file:
            continue
        else:
            command = "python3 run.py --n_jobs 4 --seeds " + seeds + " --fold -1 --num_hidden_layers " + num_hidden_layers + " --config_file_path " + file + " --scalar_mix " +str(scalar_mix) + " --model_name " + model + " --clean_run True --run_amnesic " + str(amnesic) + " --run_default True --run_mdl " + str(mdl)

            if specific_layer is not None:
                command = command + " --specific_layer " + str(specific_layer)

            result = os.system(command)
            if result != 0:
                failed_runs.append(command)
    if len(failed_runs) > 0:
        file = open(task  + "_" + model.replace("/", "_") + "_fails.txt", "w+")
        file.writelines(failed_runs)

if __name__ == "__main__":
    main()


