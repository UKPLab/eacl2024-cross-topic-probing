import copy
import glob
import itertools
import os
import random
from collections import defaultdict, Counter
from typing import List, Dict

import mlflow
import numpy
import pandas
import torch
from pandas import DataFrame
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import r2_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data_loading import parse_scalar_entries
from utils.seed_util import seed_all

from amnesic_probing.debias import classifier
from amnesic_probing.debias.debias import get_projection_to_intersection_of_nullspaces, get_rowspace_projection, \
    debias_by_specific_directions
from defs.schema import ProbingEntry
from model.probing_model import LinearProbingModel, ScalarMixProbingModel
from parsing.data_loader import get_topic_splits
from utils.dropbox_util import upload


class ProbeWorker:

    def __init__(self, hyperparameter: dict, probing_frames: Dict[int,DataFrame], probe_name: str, mlflow_url:str, run_mdl:bool, folds:List[int], project_prefix:str, topic_splits:List[List[str]] = None, sync_mdl=True):
        self.hyperparameter = hyperparameter
        self.probing_frames = probing_frames
        self.probe_name = probe_name
        self.batch_size = hyperparameter["batch_size"]
        self.n_layers = len(probing_frames)
        self.project_prefix = project_prefix
        self.setting = "in-topic" if "-in" in self.probe_name else "cross-topic"

        if run_mdl:
            self.mdl_method = "linear" if "-in" in self.probe_name else "cross-topic"
        else:
            self.mdl_method = "none"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gpus = 1 if torch.cuda.is_available() else 0
        self.mlflow_url = mlflow_url
        self.probing_model = LinearProbingModel if not self.hyperparameter["scalar_mix"] else ScalarMixProbingModel
        self.folds = folds
        self.parsed_folds = self.parse_folds()
        self.sync_mdl = sync_mdl

        if self.sync_mdl:
            if topic_splits:
                self.topic_splits = topic_splits
            else:
                self.topic_splits = self.load_topic_splits()

    def load_topic_splits(self):
        topic_splits = {}
        for probing_frame in self.probing_frames:
            including_folds = [int(col.split("-")[-1]) for col in probing_frame.columns if "set-" in col]
            splits = get_topic_splits(self.probing_frames[0], len(including_folds))

            for i, split in enumerate(splits):
                topic_splits[including_folds[i]] = split

        return topic_splits

    def get_fold_entries(self, fold, split, probing_frames):
        split_frame = probing_frames[probing_frames["set-" + str(fold)] == split]
        entries = parse_scalar_entries(split_frame)
        return entries

    def parse_folds(self):
        parsed_folds = {}
        for probing_frame in self.probing_frames:
            including_folds = [int(col.split("-")[-1]) for col in probing_frame.columns if "set-" in col]
            for fold in including_folds:
                parsed_folds[fold] = {
                    "train": self.get_fold_entries(fold, "train", probing_frame),
                    "dev": self.get_fold_entries(fold, "dev", probing_frame),
                    "test": self.get_fold_entries(fold, "test", probing_frame)
                }

        return parsed_folds


    def get_dataloader(self,entries , shuffle=False):
        dataloader = DataLoader(
                entries, batch_size=self.batch_size, shuffle=shuffle, num_workers=0
        )
        dataloader.collate_fn = self.batching_collate
        return dataloader

    def get_scalarmixin_weights(self, probing_model):
        scalar_mix = probing_model.layers[0]
        weights = torch.softmax(torch.tensor(scalar_mix.scalar_parameters), dim=0)
        weights = {
            "layer-weight-" + str(layer): float(weight.detach().cpu())
            for layer, weight in enumerate(weights)
        }

        return weights

    def train_run(self, train_entries, dev_entries, test_entries, log_dir, logger=None):
        train_dataloader = self.get_dataloader(train_entries, shuffle=True)
        dev_dataloader = self.get_dataloader(dev_entries, shuffle=False)
        test_dataloader = self.get_dataloader(test_entries, shuffle=False)

        self.hyperparameter["training_steps"] = len(train_dataloader) * 20
        self.hyperparameter["warmup_steps"] = self.hyperparameter["training_steps"] * self.hyperparameter["warmup_rate"]

        trainer = Trainer(
            logger=logger, max_epochs=20, gpus=self.gpus,
            num_sanity_val_steps=0, deterministic=False, gradient_clip_val=0.5,
            callbacks=[ModelCheckpoint(monitor="val_ref",  mode="max", dirpath=log_dir)]
        )

        probing_model = self.probing_model(
            hyperparameter=self.hyperparameter,
        )

        trainer.fit(model=probing_model, train_dataloader=train_dataloader, val_dataloaders=[dev_dataloader])

        trainer.test(ckpt_path="best", test_dataloaders=[test_dataloader])

        test_predictions = [(ele.inputs, ele.context, pred, ele.label) for ele, pred in zip(test_entries, probing_model.test_preds)]

        if type(probing_model) == ScalarMixProbingModel:
            mixin_weights = self.get_scalarmixin_weights(probing_model)
            mixin_weights["fraction"] = 1.0
        else:
            mixin_weights = {"fraction": 1.0}

        return pandas.DataFrame(test_predictions, columns=["inputs", "context", "pred", "ele.label"]), mixin_weights

    def train_mdl_run(self, train_entries, dev_online_entries, dev_entries, test_entries, log_dir, logger=None):
        train_dataloader = self.get_dataloader(train_entries, shuffle=True)
        dev_online_dataloader = self.get_dataloader(dev_online_entries, shuffle=False)
        dev_dataloader = self.get_dataloader(dev_entries, shuffle=False)
        test_dataloader = self.get_dataloader(test_entries, shuffle=False)

        self.hyperparameter["training_steps"] = len(train_dataloader) * 20
        self.hyperparameter["warmup_steps"] = self.hyperparameter["training_steps"] * self.hyperparameter["warmup_rate"]

        trainer = Trainer(
            logger=logger, max_epochs=20, gpus=self.gpus,
            num_sanity_val_steps=0, deterministic=False, gradient_clip_val=0.5,
            callbacks=[ModelCheckpoint(monitor="val_ref",  mode="max", dirpath=log_dir)]
        )

        probing_model = self.probing_model(
            hyperparameter=self.hyperparameter,
        )

        trainer.fit(model=probing_model, train_dataloader=train_dataloader, val_dataloaders=[dev_dataloader])

        dev_metrics = trainer.validate(ckpt_path="best", dataloaders=[dev_online_dataloader])
        test_metrics = trainer.test(ckpt_path="best", test_dataloaders=[test_dataloader])

        summed_loss = dev_metrics[0]["val loss sum"]

        if type(probing_model) == ScalarMixProbingModel:
            mixin_weights = self.get_scalarmixin_weights(probing_model)
            mixin_weights["fraction"] = 1.0
        else:
            mixin_weights = {}

        return summed_loss, test_metrics[0], mixin_weights


    def run_linear_task_fraction(self, fraction:int, entries:list, dev_entries:list, test_entries:list, log_dir:str=None):

        fraction_length = int(len(entries) * fraction)

        if fraction_length == 0:
            return 0, 0, 0, {}

        train_entries =entries[:fraction_length]
        dev_online_entries = entries[fraction_length:2*fraction_length]

        summed_loss, best_test_metrics, mixin_weights = self.train_mdl_run(train_entries, dev_online_entries, dev_entries, test_entries, log_dir + "/frac-" + str(fraction_length), logger=False)
        mixin_weights["fraction"] = fraction

        return summed_loss, best_test_metrics, len(dev_online_entries), mixin_weights

    def run_cross_topic_task_fraction(self, fraction:float, entries:list, dev_entries:list, test_entries:list,  train_topics:list, log_dir=None):
        num_topics = int(len(train_topics) / 2)
        selected_topics = train_topics[:num_topics]

        if num_topics == 0:
            n_training_samples = int(len(entries) / 2)
            train_entries = entries[:n_training_samples]
            dev_online_entries = entries[n_training_samples:]
        else:
            train_entries = [entry for entry in entries if entry.topic in selected_topics]
            dev_online_entries = [entry for entry in entries if entry.topic not in selected_topics and entry.topic in train_topics]

        grouped_train_entries = defaultdict(list)
        grouped_dev_entries = defaultdict(list)

        for train_entry in train_entries:
            grouped_train_entries[train_entry.topic].append(train_entry)

        for dev_entry in dev_online_entries:
            grouped_dev_entries[dev_entry.topic].append(dev_entry)

        for topic, topic_samples in grouped_train_entries.items():
            grouped_train_entries[topic] = topic_samples[:int(len(topic_samples) * fraction * 2)]

        for topic, topic_samples in grouped_dev_entries.items():
            grouped_dev_entries[topic] = topic_samples[:int(len(topic_samples) * fraction * 2)]


        train_entries = list(itertools.chain.from_iterable(grouped_train_entries.values()))
        dev_online_entries = list(itertools.chain.from_iterable(grouped_dev_entries.values()))

        min_length = min(len(train_entries), len(dev_online_entries))

        if min_length < num_topics or min_length == 0:
            return 0, 0, 0, {}

        summed_loss, best_test_metrics, mixin_weights = self.train_mdl_run(train_entries, dev_online_entries, dev_entries, test_entries, log_dir + "/frac-" + str(fraction), logger=False)
        mixin_weights["fraction"] = fraction

        return summed_loss, best_test_metrics, len(dev_online_entries), mixin_weights


    def get_mdl_train_entries(self, fold, is_cross_topic, train_entries, dev_entries, test_entries):
        if not self.sync_mdl:
            return train_entries

        all_entries = train_entries + dev_entries + test_entries

        cross_train_topics = self.topic_splits[fold][0]
        cross_n_train_topics = len(cross_train_topics) - (len(cross_train_topics) % 2)
        cross_train_topics = random.Random(fold).sample(cross_train_topics, cross_n_train_topics)
        cross_min_samples_per_topic = min(Counter([entry.topic for entry in all_entries if entry.topic in cross_train_topics]).values())
        cross_train_entries = list(itertools.chain.from_iterable([
            [entry for entry in all_entries if entry.topic in topic][:cross_min_samples_per_topic]
            for topic in cross_train_topics
        ]))

        if is_cross_topic:
            return cross_train_entries
        else:
            if len(cross_train_entries) > len(train_entries):
                return train_entries
            else:
                return random.Random(fold).sample(train_entries, len(cross_train_entries))


    def run_mdl_tasks(self, fold, fractions, log_dir, is_cross_topic, train_entries, dev_entries, test_entries, is_regression):

        mdl_train_entries = self.get_mdl_train_entries(fold, is_cross_topic, train_entries, dev_entries, test_entries)

        if is_regression:
            labels = [ele.label for ele in mdl_train_entries]

            dummy_model = DummyRegressor(strategy="mean")
            dummy_model.fit(labels, labels)
            samples_labels = dummy_model.predict(labels)

            uniform_code_length = float(torch.nn.MSELoss(reduction="sum")(torch.tensor(samples_labels), torch.tensor(labels)))
        else:
            uniform_code_length = len(mdl_train_entries) * numpy.log2(self.hyperparameter["num_labels"])

        fraction_losses = []
        collected_test_metrics = []
        fraction_lengths = []
        overall_test_metrics = []
        all_mixin_weights = []

        for fraction in fractions:

            if is_cross_topic:
                train_topics = list(sorted(set([entry.topic for entry in mdl_train_entries])))

                random.Random(fold).shuffle(train_topics)

                summed_loss, test_metrics, fraction_length, mixin_weights = self.run_cross_topic_task_fraction(
                    fraction=fraction, entries=mdl_train_entries, dev_entries=dev_entries, test_entries=test_entries,
                    train_topics=train_topics, log_dir=log_dir,
                )
            else:
                summed_loss, test_metrics, fraction_length, mixin_weights = self.run_linear_task_fraction(
                    fraction=fraction, entries=mdl_train_entries, dev_entries=dev_entries,
                    test_entries=test_entries, log_dir=log_dir
                )

            if fraction_length == 0:
                continue

            all_mixin_weights.append(mixin_weights)

            fraction_losses.append(summed_loss)

            if is_regression:

                collected_test_metrics.append({
                    "pearson": test_metrics["test pearson"]
                })
            else:
                collected_test_metrics.append({
                    "acc": test_metrics["test acc"],
                    "f1": test_metrics["test f1"]
                })

            test_metrics["fraction"] = fraction

            overall_test_metrics.append(test_metrics)

            fraction_lengths.append(fraction_length)

        os.system("rm -rf " + log_dir + "/frac*")

        first_portion_size = min([ele for ele in fraction_lengths if ele > 0])

        if is_regression:
            minimum_description_length = first_portion_size * (uniform_code_length / len(mdl_train_entries)) + sum(fraction_losses)
        else:
            minimum_description_length = first_portion_size * numpy.log2(self.hyperparameter["num_labels"]) + sum(fraction_losses)

        compression = uniform_code_length/minimum_description_length

        return uniform_code_length, minimum_description_length, compression, fraction_losses, fraction_lengths, collected_test_metrics, all_mixin_weights


    def run_amnesic_tasks(self, train_entries, dev_entries, test_entries, is_regression, fold, is_cross_topic, run_mdl):
        P, P_rand, dev_scores, test_scores, dev_rand_scores, test_rand_scores = self.get_amnesic_projections(train_entries, dev_entries, test_entries, is_regression)


        for P, method in [(P, "debias"), (P_rand, "rand")]:
            logger = MLFlowLogger(experiment_name=self.project_prefix + "-amnesic-" + self.probe_name, tracking_uri=self.mlflow_url)
            run_id = logger.run_id
            log_dir = "./lightning_logs/" + run_id

            self.hyperparameter["amnesic"] = method
            self.hyperparameter["dump_id"] = run_id

            transformed_train_entries, transformed_dev_entries, transformed_test_entries = self.remove_property(P, train_entries, dev_entries, test_entries)

            prediction_frame, mixin_weights = self.train_run(log_dir=log_dir, logger=logger, train_entries=transformed_train_entries, dev_entries=transformed_dev_entries, test_entries=transformed_test_entries)
            prediction_frame.to_csv(log_dir + "/preds_" + method + ".csv")


            if run_mdl:
                mlflow.set_tracking_uri(self.mlflow_url)

                uniform_code_length, minimum_description_length, compression, fraction_losses, fraction_lengths, collected_test_metrics, all_mixin_weights = self.run_mdl_tasks(
                    fractions=[1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024],
                    log_dir=log_dir,
                    fold=fold,
                    is_cross_topic=is_cross_topic,
                    train_entries=transformed_train_entries,
                    dev_entries=transformed_dev_entries,
                    test_entries=transformed_test_entries,
                    is_regression=is_regression
                )

                self.save_mdl_metrics(
                    run_id, uniform_code_length, minimum_description_length, compression, fraction_losses,
                    fraction_lengths, collected_test_metrics
                )

            numpy.save(open(log_dir + "/P_" + method + ".npy", "wb"), P)
            numpy.save(open(log_dir + "/amnesic" + method + "_dev_scores.npy", "wb"), dev_scores)
            numpy.save(open(log_dir + "/amnesic" + method + "_test_scores.npy", "wb"), test_scores)

            for file_path in glob.glob(log_dir + "/*"):
                upload(file_path, "/" + self.project_prefix + "-amnesic-results/" + run_id + "/" + file_path.split("/")[-1])

    def save_mdl_metrics(self, run_id, uniform_code_length, minimum_description_length, compression, fraction_losses, fraction_lengths, collected_test_metrics):
        mlflow.set_tracking_uri(self.mlflow_url)

        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_metric("uniform_length",uniform_code_length)
            mlflow.log_metric("minimum_description_length",minimum_description_length)
            mlflow.log_metric("compression",compression)

            for i, (fraction_loss, fraction_length, test_metrics) in enumerate(zip(fraction_losses, fraction_lengths, collected_test_metrics)):
                if i > 0:
                    for metric in test_metrics.keys():
                        mlflow.log_metric("z_test_" + str(i) + "_" + metric, test_metrics[metric], step=fraction_length)
                        mlflow.log_metric("z_loss_" + str(i) + "_" + metric, fraction_loss, step=fraction_length)

    def get_entries(self, fold):
        return self.parsed_folds[fold]["train"], self.parsed_folds[fold]["dev"], self.parsed_folds[fold]["test"]

    def run_amnesic_task(self, fold, is_cross_topic, run_mdl):
        is_regression = self.hyperparameter["num_labels"] == 1
        train_entries, dev_entries, test_entries = self.get_entries(fold)
        self.run_amnesic_tasks(
            train_entries=train_entries, dev_entries=dev_entries, test_entries=test_entries,
            is_regression=is_regression, fold=fold, is_cross_topic=is_cross_topic, run_mdl=run_mdl
        )

    def run_stacked_amnesic_task(self, fold, is_cross_topic, run_mdl):
        is_regression = self.hyperparameter["num_labels"] == 1
        train_entries, dev_entries, test_entries = self.get_entries(fold)
        self.run_stacked_amnesic_tasks(
            train_entries=train_entries, dev_entries=dev_entries, test_entries=test_entries,
            is_regression=is_regression, fold=fold, is_cross_topic=is_cross_topic, run_mdl=run_mdl
        )

    def run_rlace_task(self, fold, is_cross_topic, run_mdl):
        is_regression = self.hyperparameter["num_labels"] == 1
        train_entries, dev_entries, test_entries = self.get_entries(fold)
        self.run_rlace_tasks(
            train_entries=train_entries, dev_entries=dev_entries, test_entries=test_entries,
            is_regression=is_regression, fold=fold, is_cross_topic=is_cross_topic, run_mdl=run_mdl
        )

    def run_default_task(self, fold, run_mdl=False, is_cross_topic=False):
        logger = MLFlowLogger(experiment_name=self.project_prefix + "-" + self.probe_name, tracking_uri=self.mlflow_url)
        run_id = logger.run_id
        log_dir = "./lightning_logs/" + run_id

        self.hyperparameter["amnesic"] = ""
        self.hyperparameter["dump_id"] = run_id

        is_regression = self.hyperparameter["num_labels"] == 1

        train_entries, dev_entries, test_entries = self.get_entries(fold)

        prediction_frame, mixin_weights = self.train_run(train_entries=train_entries, dev_entries=dev_entries, test_entries=test_entries, log_dir=log_dir, logger=logger)
        prediction_frame.to_csv(log_dir +"/preds.csv")


        if run_mdl:
            uniform_code_length, minimum_description_length, compression, fraction_losses, fraction_lengths, collected_test_metrics, all_mixin_weights = self.run_mdl_tasks(
                fractions=[1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024],
                log_dir=log_dir,
                fold=fold,
                is_cross_topic=is_cross_topic,
                train_entries=train_entries,
                dev_entries=dev_entries,
                test_entries=test_entries,
                is_regression=is_regression
            )
            self.save_mdl_metrics(
                run_id, uniform_code_length, minimum_description_length, compression, fraction_losses,
                fraction_lengths, collected_test_metrics
            )

        if type(self.probing_model) == ScalarMixProbingModel.__class__ and self.hyperparameter["scalar_mix"]:
            all_mixin_weights.append(mixin_weights)
            pandas.DataFrame(all_mixin_weights).sort_values("fraction").to_csv(log_dir + "/mixin_weights.csv", index=False)

        for file_path in glob.glob(log_dir + "/*"):
            upload(file_path, "/" + self.project_prefix + "-results/" + run_id + "/" + file_path.split("/")[-1])

        os.system("rm -rf " + log_dir)

        return "Done"

    def run_default_task_without_property(self, fold, run_mdl=False, is_cross_topic=False, amnesic_property=None, amnesic_file=None):
        logger = MLFlowLogger(experiment_name=self.project_prefix + "-amnesic-" + self.probe_name, tracking_uri=self.mlflow_url)
        run_id = logger.run_id
        log_dir = "./lightning_logs/" + run_id

        self.hyperparameter["amnesic"] = amnesic_property
        self.hyperparameter["dump_id"] = run_id

        P = numpy.load(amnesic_file)

        is_regression = self.hyperparameter["num_labels"] == 1

        train_entries, dev_entries, test_entries = self.get_entries(fold)
        transformed_train_entries, transformed_dev_entries, transformed_test_entries = self.remove_property(P, train_entries, dev_entries, test_entries)

        prediction_frame, mixin_weights = self.train_run(train_entries=transformed_train_entries, dev_entries=transformed_dev_entries, test_entries=transformed_test_entries, log_dir=log_dir, logger=logger)
        prediction_frame.to_csv(log_dir +"/preds.csv")

        if run_mdl:
            uniform_code_length, minimum_description_length, compression, fraction_losses, fraction_lengths, collected_test_metrics, all_mixin_weights = self.run_mdl_tasks(
                fractions=[1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024],
                log_dir=log_dir,
                fold=fold,
                is_cross_topic=is_cross_topic,
                train_entries=transformed_train_entries,
                dev_entries=transformed_dev_entries,
                test_entries=transformed_test_entries,
                is_regression=is_regression
            )

            self.save_mdl_metrics(
                run_id, uniform_code_length, minimum_description_length, compression, fraction_losses,
                fraction_lengths, collected_test_metrics
            )

        for file_path in glob.glob(log_dir + "/*"):
            upload(file_path, "/" + self.project_prefix + "-amnesic-results/" + run_id + "/" + file_path.split("/")[-1])

        os.system("rm -rf " + log_dir)

        return "Done"


    def dot(self, a, b):
        a_tensor = torch.FloatTensor(a).to(self.device)
        b_tensor = torch.FloatTensor(b).to(self.device)
        return torch.mm(a_tensor, b_tensor).detach().cpu().numpy()


    def get_classification_debiasing_projection(self, classifier_class, cls_params: Dict, num_classifiers: int, input_dim: int,
                                                is_autoregressive: bool, X_test: numpy.ndarray, Y_test: numpy.ndarray,
                                                min_score: float, X_train: numpy.ndarray, Y_train: numpy.ndarray, X_dev: numpy.ndarray,
                                                Y_dev: numpy.ndarray, best_iter_diff=0.01) -> (numpy.ndarray, list, list, list, tuple):
        """
        :param classifier_class: the sklearn classifier class (SVM/Perceptron etc.)
        :param cls_params: a dictionary, containing the params for the sklearn classifier
        :param num_classifiers: number of iterations (equivalent to number of dimensions to remove)
        :param input_dim: size of input vectors
        :param is_autoregressive: whether to train the ith classiifer on the data projected to the nullsapces of w1,...,wi-1
        :param min_score: above this threshold, ignore the learned classifier
        :param X_train: ndarray, training vectors
        :param Y_train: ndarray, training labels (protected attributes)
        :param X_dev: ndarray, eval vectors
        :param Y_dev: ndarray, eval labels (protected attributes)
        :param best_iter_diff: float, diff from majority, used to decide on best iteration
        :return: P, the debiasing projection; rowspace_projections, the list of all rowspace projection;
                Ws, the list of all calssifiers.
        """

        X_train_cp = X_train.copy()
        X_dev_cp = X_dev.copy()
        X_test_cp = X_test.copy()
        rowspace_projections = []
        Ws = []
        all_projections = []
        best_projection = None
        iters_under_threshold = 0
        prev_acc = -99.
        iters_no_change = 0

        dev_scores = []
        test_scores = []

        pbar = tqdm(range(num_classifiers))
        for i in pbar:
            clf = classifier.SKlearnClassifier(classifier_class(**cls_params))
            dev_score = clf.train_network(X_train_cp, Y_train, X_dev_cp, Y_dev)
            test_score = clf.model.score(X_test_cp, Y_test)
            pbar.set_description("iteration: {}, accuracy: {}".format(i, dev_score))

            dev_scores.append(dev_score)
            test_scores.append(test_score)

            if iters_under_threshold >= 3:
                print('3 iterations under the minimum accuracy.. stopping the process')
                break

            if dev_score <= min_score and best_projection is not None:
                iters_under_threshold += 1
                continue

            if round(prev_acc, 3) == round(dev_score, 3):
                iters_no_change += 1
            else:
                iters_no_change = 0

            if iters_no_change >= 3:
                print('3 iterations with no accuracy change.. topping the process')
                break

            prev_acc = dev_score

            W = clf.get_weights()
            Ws.append(W)
            P_rowspace_wi = get_rowspace_projection(W)  # projection to W's rowspace
            rowspace_projections.append(P_rowspace_wi)

            if is_autoregressive:
                """
                to ensure numerical stability, explicitly project to the intersection of the nullspaces found so far
                 (instead of doing X = P_iX, which is problematic when w_i is not exactly orthogonal to w_i-1,...,w1,
                  due to e.g inexact argmin calculation).
                """
                # use the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
                # N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))

                P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
                all_projections.append(P)

                # project
                X_train_cp = self.dot(X_train, P)
                X_dev_cp = self.dot(X_dev, P)
                X_test_cp = self.dot(X_test, P)

                # the first iteration that gets closest performance (or less) to majority
                if (dev_score - min_score) <= best_iter_diff and best_projection is None:
                    print('projection saved timestamp: {}'.format(i))
                    best_projection = (P, i + 1)

        """
        calculae final projection matrix P=PnPn-1....P2P1
        since w_i.dot(w_i-1) = 0, P2P1 = I - P1 - P2 (proof in the paper); this is more stable.
        by induction, PnPn-1....P2P1 = I - (P1+..+PN). We will use instead Ben-Israel's formula to increase stability,
        i.e., we explicitly project to intersection of all nullspaces (this is not critical at this point; I-(P1+...+PN)
        is roughly as accurate as this)
        """

        P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

        if best_projection is None:
            print('projection saved timestamp: {}'.format(num_classifiers))
            print('using all of the iterations as the final projection')
            best_projection = (P, num_classifiers)

        return best_projection, dev_scores, test_scores

    def remove_property_from_entry(self, P, entry):
        new_entry = copy.deepcopy(entry)

        P_dim = P.shape[1]
        entry_dim = entry.inputs_encoded.shape[1]
        if P_dim == entry_dim:
            new_entry.inputs_encoded = torch.FloatTensor(self.dot(P, new_entry.inputs_encoded.T).T)
        else:
            elements = [new_entry.inputs_encoded[:, i * P_dim: (i+1) * P_dim] for i in range(int(entry_dim/P_dim))]
            updated_elements = [torch.FloatTensor(self.dot(P, element.T).T)[0] for element in elements]
            new_entry.inputs_encoded = torch.stack(updated_elements).flatten().unsqueeze(0)
        return new_entry

    def remove_property_from_entries(self, P, entries):
        updated_entries = [self.remove_property_from_entry(P, entry) for entry in entries]
        return updated_entries

    def remove_property(self, P, train_entries, dev_entries, test_entries):
        train_entries = self.remove_property_from_entries(P, train_entries)
        dev_entries = self.remove_property_from_entries(P, dev_entries)
        test_entries = self.remove_property_from_entries(P, test_entries)

        return train_entries, dev_entries, test_entries

    def get_random_projection(self, n_coord, classifier_class, cls_params, X_train, Y_train, X_dev, Y_dev, X_test, Y_test,):
        dim = X_train.shape[1]
        rand_directions = [numpy.random.rand(1, dim) - 0.5 for _ in range(n_coord)]


        dev_scores = []
        test_scores = []

        for i in range(1, len(rand_directions)):
            P = debias_by_specific_directions(rand_directions[:i], dim)

            X_train_cp = self.dot(X_train, P)
            X_dev_cp = self.dot(X_dev, P)
            X_test_cp = self.dot(X_test, P)

            clf = classifier.SKlearnClassifier(classifier_class(**cls_params))
            dev_acc = clf.train_network(X_train_cp, Y_train, X_dev_cp, Y_dev)
            test_acc = clf.model.score(X_test_cp, Y_test)

            dev_scores.append(dev_acc)
            test_scores.append(test_acc)


        rand_direction_p = debias_by_specific_directions(rand_directions, dim)

        return rand_direction_p, dev_scores, test_scores


    def ignore_label(self, samples, labels, labels_to_ignore):
        indices_to_ignore = []

        for label_to_ignore in labels_to_ignore:
            indices_to_ignore = indices_to_ignore + list(numpy.where(labels == label_to_ignore)[0])

            samples = numpy.delete(samples, indices_to_ignore, axis=0)
            labels = numpy.delete(labels, indices_to_ignore, axis=0)

        return samples, labels


    def get_amnesic_projections(self, train_entries, dev_entries, test_entries, is_regression=False):

        x_train = numpy.array([entry.inputs_encoded.flatten().numpy() for entry in train_entries])
        y_train = numpy.array([entry.label for entry in train_entries])
        x_dev = numpy.array([entry.inputs_encoded.flatten().numpy() for entry in dev_entries])
        y_dev = numpy.array([entry.label for entry in dev_entries])
        x_test = numpy.array([entry.inputs_encoded.flatten().numpy() for entry in test_entries])
        y_test = numpy.array([entry.label for entry in test_entries])

        train_label_counter = Counter(y_train)
        dev_label_counter = Counter(y_dev)
        test_label_counter = Counter(y_test)

        train_label_to_ignore = [k for k,v in train_label_counter.items() if v == 1]
        dev_label_to_ignore = [k for k,v in dev_label_counter.items() if v == 1]
        test_label_to_ignore = [k for k,v in test_label_counter.items() if v == 1]

        if not is_regression:

            label_to_ignore = train_label_to_ignore + dev_label_to_ignore + test_label_to_ignore

            x_train, y_train = self.ignore_label(x_train, y_train, label_to_ignore)
            x_dev, y_dev = self.ignore_label(x_dev, y_dev, label_to_ignore)
            x_test, y_test = self.ignore_label(x_test, y_test, label_to_ignore)

        if is_regression:

            dummy_model = DummyRegressor(strategy="mean")
            dummy_model.fit(x_train, y_train)
            samples_labels = dummy_model.predict(x_dev)

            majority_score = r2_score(y_dev, samples_labels)

            clf = SGDRegressor
            params = {'warm_start': True, 'loss': 'squared_error',  'max_iter': 500, 'random_state': self.hyperparameter["seed"],
                      'early_stopping': True}

            #best_projection, dev_scores, test_scores = self.get_regression_debias_projection(
            #    X_train=x_train, Y_train=y_train, X_dev=x_dev, Y_dev=y_dev, X_test=x_test, Y_test=y_test,
            #    num_classifiers=30
            #)
        else:

            dummy_classifier = DummyClassifier(strategy="most_frequent")
            dummy_classifier.fit(x_train, y_train)
            samples_labels = dummy_classifier.predict(x_dev)

            majority_score = accuracy_score(y_dev, samples_labels)

            clf = SGDClassifier
            params = {'warm_start': True, 'loss': 'log', 'n_jobs': -1, 'max_iter': 500, 'random_state': self.hyperparameter["seed"],
                        'early_stopping': True}

        best_projection, dev_scores, test_scores = self.get_classification_debiasing_projection(
            classifier_class=clf, cls_params=params, input_dim=x_train.shape[1], is_autoregressive=True,
            X_train=x_train, Y_train=y_train, X_dev=x_dev, Y_dev=y_dev, X_test=x_test, Y_test=y_test,
            min_score=majority_score, num_classifiers=20
        )

        P = best_projection[0]
        P_rand, dev_rand_scores, test_rand_scores = self.get_random_projection(20, X_train=x_train, Y_train=y_train, X_dev=x_dev, Y_dev=y_dev, X_test=x_test, Y_test=y_test, classifier_class=clf, cls_params=params,)

        return P, P_rand, dev_scores, test_scores, dev_rand_scores, test_rand_scores


    def run_tasks(self, run_default=True, run_mdl=False, run_amnesic=False, run_rlace=False, run_stacked_amnesic=False, amnesic_file="", amnesic_property=""):
        seed_all(self.hyperparameter["seed"])
        is_cross_topic = self.setting == "cross-topic"
        for fold in self.folds:
            self.hyperparameter["fold"] = fold
            if run_amnesic and self.hyperparameter["num_hidden_layers"] == 0:
                self.run_amnesic_task(fold=fold, run_mdl=run_mdl, is_cross_topic=is_cross_topic)
            if run_stacked_amnesic:
                self.run_stacked_amnesic_task(fold=fold, run_mdl=run_mdl, is_cross_topic=is_cross_topic)
            if run_default and amnesic_property != "":
                self.run_default_task_without_property(fold=fold, run_mdl=run_mdl, is_cross_topic=is_cross_topic, amnesic_file=amnesic_file, amnesic_property=amnesic_property)
            elif run_default:
                self.run_default_task(fold=fold, run_mdl=run_mdl, is_cross_topic=is_cross_topic)


    def batching_collate(self, batch:List[ProbingEntry]):

        if not self.hyperparameter["scalar_mix"]:
            encoded_inputs = torch.stack([
                torch.flatten(element.inputs_encoded)
                for element in batch
            ])
        else:
            encoded_inputs = torch.stack([
                element.inputs_encoded
                for element in batch
            ]).transpose(1,0)

        if type(batch[0].label) == int:
            labels = torch.LongTensor(
                [element.label for element in batch]
            )
        else:
            labels = torch.Tensor(
                [element.label for element in batch]
            )

        return encoded_inputs, labels
