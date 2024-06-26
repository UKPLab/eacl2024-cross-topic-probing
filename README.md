# Dive into the Chasm: Probing the Gap between In- and Cross-Topic Generalization
This repository includes the code for running the experiments to probe in the In- and Cross-Topic experimental settings.

Further details can be found in our publication _Dive into the Chasm: Probing the Gap between In- and Cross-Topic Generalization_


> **Abstract:** Pre-trained Language Models (PLMs) show impressive success when learning downstream tasks.
Pre-trained language models (LMs) perform well in In-Topic setups, where training and testing data come from the same topics. 
However, they face challenges in Cross-Topic scenarios where testing data is derived from distinct topics - such as Gun Control. 
This study analyzes various LMs with three probing-based experiments to shed light on the reasons behind the In- vs. Cross-Topic generalization gap.
Thereby, we demonstrate, for the first time, that generalization gaps and the robustness of the embedding space vary significantly across LMs. 
Additionally, we assess larger LMs and underscore the relevance of our analysis for recent models.
Overall, diverse pre-training objectives, architectural regularization, or data deduplication contribute to more robust LMs and diminish generalization gaps. 
Our research contributes to a deeper understanding and comparison of language models across different generalization scenarios.

Contact person: Andreas Waldis, andreas.waldis@live.com

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to e-mail us or report an issue if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

## Project structure

* `data/` -- directory for the data
* `probes/` -- directory for the probes
* `src` -- contains all necessary python files 

## Setup and Requirements

This repository requires Python3.6 or higher; further requirements can be found in the requirements.txt. 
Install them with the following command:

```
pip install -r requirements.txt
```

Further, it need the spacy model `en_core_web_sm`.
This repository requires a running MLFLOW instance for reporting and Dropbox as storage for computed models. Please define the corresponding URL and Dropbox auth-token in the file `src/defs/default_config.py`

## Data

We make use of the _UKP ArgMin_ dataset [Stab et al. 2018](https://aclanthology.org/D18-1402) and the _WTWT_ dataset [Conforti et al. 2020](conforti-etal-2020-will).
These datasets are freely available but you need to request _UKP ArgMin_ [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2345).
Once you have obtained both datasets, put them in the `data` folder.
**Note**, if you do not want to parse the data by your own, contact us and we will share the parsed probing dataset with you as soon you got data access.


## Generate probing tasks
After generating the probing tasks you will find them in the `probes` folder.

To generate the Cross-Topic probing task, run the following commands:

```
$ parse_task_cross_topic.py --task ukp-argmin 
$ parse_task_cross_topic.py --task wtwt
```

To generate the Cross-Topic topic information classification tasks, run the following commands:

```
$ parse_task_cross_topic_tokens.py --task ukp-argmin 
$ parse_task_cross_topic_tokens.py --task wtwt
```

To generate the In-Topic probing tasks, run the following commands:

```
$ convert_cross_topic_tasks.py
```

To generate the control tasks, run the following commands:

```
$ convert_control_task.py
```

## Experiments I

To run the first experiments on a model - like `bert-base-uncased` - run the following command.

```
$ run-probes.py --task ukp-argmin --model bert-base-uncased --seeds 0,1,2 --out_filter token-types --amnesic False
$ run-probes.py --task ukp-argmin-in --model bert-base-uncased --seeds 0,1,2 --out_filter token-types --amnesic False
$ run-probes.py --task wtwt --model bert-base-uncased --seeds 0,1,2 --out_filter token-types --amnesic False
$ run-probes.py --task wtwt-in --model bert-base-uncased --seeds 0,1,2 --out_filter token-types --amnesic False
```

## Experiments 2

To run the second experiment on a model - like `bert-base-uncased` - run the following command to fit the amnesic probe.

```
$ run-probes.py --task ukp-argmin --model bert-base-uncased --seeds 0,1,2 --in_filter token-types --amnesic True
$ run-probes.py --task ukp-argmin-in --model bert-base-uncased --seeds 0,1,2 --in_filter token-types --amnesic True
$ run-probes.py --task wtwt --model bert-base-uncased --seeds 0,1,2 --in_filter token-types --amnesic True
$ run-probes.py --task wtwt-in --model bert-base-uncased --seeds 0,1,2 --in_filter token-types --amnesic True
```

Afterward we can run probes again without topic information, for example, NER for Cross-Topic on _ukp-argmin_

```
$ run-probes-without-property.py --amnesic_experiment probes-amnesic-ukp-argmin-token-types-40-cls-topic-maj --target_experiment probes-ukp-argmin-ner
```

## Experiments 3

To run the third experiment, a fine-tuned model (fold=0, seed=0) - like `bert-base-uncased-ft-ukp-argmin-0-0` - runs the following command. Note you need to provide the fine-tuned model as a folder in the running directory.

```
$ evolution-dropbox.py --task ukp-argmin --model_name bert-base-uncased-ft-ukp-argmin-0-0 --seed 0 --fold 0 
```

## Citation
```
@inproceedings{waldis-etal-2024-dive,
    title = "Dive into the Chasm: Probing the Gap between In- and Cross-Topic Generalization",
    author = "Waldis, Andreas  and
      Hou, Yufang  and
      Gurevych, Iryna",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-eacl.146",
    pages = "2197--2214",
    abstract = "Pre-trained language models (PLMs) perform well in In-Topic setups, where training and testing data come from the same topics. However, they face challenges in Cross-Topic scenarios where testing data is derived from distinct topics. This paper analyzes various PLMs with three probing-based experiments to better understand the reasons behind such generalization gaps. For the first time, we demonstrate that the extent of these generalization gaps and the sensitivity to token-level interventions vary significantly across PLMs. By evaluating large language models (LLMs), we show the usefulness of our analysis for these recent models. Overall, we observe diverse pre-training objectives and architectural regularization contribute to more robust PLMs and mitigate generalization gaps. Our research contributes to a deeper understanding and comparison of language models across different generalization scenarios.",
}

```
