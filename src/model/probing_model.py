from collections import defaultdict
from typing import Dict, List

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from torch.nn import ModuleList
from transformers import get_linear_schedule_with_warmup

from model.ScalarMix import ScalarMix


class SkeletonProbingModel(LightningModule):
    def __init__(self, hyperparameter):
        super().__init__()
        self.hyperparameter = hyperparameter
        if self.hyperparameter["num_labels"] == 1:
            self.loss = torch.nn.SmoothL1Loss()
            self.metrics = {
                "pearson": torchmetrics.PearsonCorrcoef()
            }
        else:
            self.loss = torch.nn.CrossEntropyLoss()
            self.metrics = {
                "acc": torchmetrics.Accuracy(),
                "f1": torchmetrics.F1(average="macro", num_classes=self.hyperparameter["num_labels"]),
            }
            if self.hyperparameter["num_labels"] > 2:
                self.f1_all = torchmetrics.F1(average="none", num_classes=self.hyperparameter["num_labels"])
                self.acc_all = torchmetrics.Accuracy(average="none", num_classes=self.hyperparameter["num_labels"])

        self.best_val_metrics = defaultdict(lambda: -1)
        self.best_test_metrics = defaultdict(lambda: -1)
        self.test_preds = []



    def run_val_step(self, batch, prefix):
        x, y = batch
        pred = self(x)
        if self.hyperparameter["num_labels"] > 1:
            pred = torch.nn.Softmax()(pred)
            loss = self.loss(pred, y)
        else:
            loss = self.loss(pred, y.unsqueeze(dim=1))
        loss_sum = loss * y.shape[0]
        self.log(prefix + " loss", loss, on_epoch=True, prog_bar=True)
        return loss, loss_sum, pred, y

    def training_step(self, batch, batch_index):
        x, y = batch
        pred = self(x)
        if self.hyperparameter["num_labels"] > 1:
            pred = torch.nn.Softmax()(pred)
            loss = self.get_loss(pred, y)
        else:
            loss = self.get_loss(pred, y.unsqueeze(dim=1))
        return loss

    def on_train_start(self):
        if self.logger:
            self.logger.log_hyperparams(self.hyperparameter)

    def validation_step(self, batch, batch_index):
       # set = "val" if dataloader_index == 0 else "test"
        return self.run_val_step(batch, "val")

    def test_step(self, batch, batch_index):
        return self.run_val_step(batch, "test")




    def training_epoch_end(self, losses):
        self.log("train loss", losses[0]["loss"])

    def test_epoch_end(self, test_step_outputs):

        losses_sum = [ele[1] for ele in test_step_outputs]
        preds = [ele[2] for ele in test_step_outputs]
        truths = [ele[3] for ele in test_step_outputs]

        pred_labels = torch.cat(preds).detach().cpu()
        truth_labels = torch.cat(truths).detach().cpu()

        summed_loss = torch.stack(losses_sum).sum()


        metric_results = {}

        for metric, func in self.metrics.items():
            if metric == "pearson":
                pred_labels = pred_labels.squeeze(dim=1)

            metric_result = func(pred_labels, truth_labels)
            metric_results["test " + metric] = float(metric_result)

        if self.hyperparameter["num_labels"] > 2:
            for label, value in enumerate(self.f1_all.cpu()(pred_labels.cpu(), truth_labels.cpu())):
                metric_results["z_test f1-" + str(label)] = float(value)
            for label, value in enumerate(self.acc_all.cpu()(pred_labels.cpu(), truth_labels.cpu())):
                metric_results["z_test acc-" + str(label)] = float(value)

        for metric, result in metric_results.items():
            self.best_test_metrics[metric] = result
            self.log(metric, result, on_epoch=True, prog_bar=True)

        self.best_test_metrics["summed_loss"] = float(summed_loss.detach().cpu())
        if self.hyperparameter["num_labels"] > 2:
            self.test_preds = pred_labels.argmax(dim=1).detach().cpu().numpy()
        else:
            self.test_preds = pred_labels.detach().cpu().numpy()


    def validation_epoch_end(self, validation_step_outputs):
        self.process_validation_results(validation_step_outputs)

    def process_validation_results(self, validation_step_outputs):

        losses_sum = [ele[1] for ele in validation_step_outputs]
        preds = [ele[2] for ele in validation_step_outputs]
        truths = [ele[3] for ele in validation_step_outputs]

        pred_labels = torch.cat(preds).detach().cpu()
        truth_labels = torch.cat(truths).detach().cpu()

        summed_loss = torch.stack(losses_sum).sum()


        metric_results = {}

        for metric, func in self.metrics.items():
            if metric == "pearson":
                pred_labels = pred_labels.squeeze(dim=1)

            metric_result = func(pred_labels, truth_labels)

            metric_results[metric] = metric_result

        if "pearson" in metric_results:
            ref_metric = float(metric_results["pearson"].detach().cpu())
        else:
            ref_metric = float(metric_results["f1"].detach().cpu())

        if ref_metric >= self.best_val_metrics["ref"] and self.current_epoch > 2:
            self.best_val_metrics["summed_loss"] = float(summed_loss.detach().cpu())
            self.best_val_metrics["ref"] = ref_metric
            self.best_val_metrics["epoch"] = ref_metric
                
            for metric, result in metric_results.items():
                self.best_val_metrics[metric] = float(result.detach().cpu())
                    
        #if set == "test" and best_epoch:
        #    for metric, result in metric_results.items():
        #        self.best_test_metrics[metric] = float(result.detach().cpu())
        #        self.best_test_metrics["summed_loss"] = float(summed_loss.detach().cpu())

        for metric, metric_result in metric_results.items():
            self.log("val " + metric, metric_result,  on_epoch=True, prog_bar=True)

        if self.hyperparameter["num_labels"] > 2:
            for label, value in enumerate(self.f1_all.cpu()(pred_labels.cpu(), truth_labels.cpu())):
                self.log("z_val f1-" + str(label), value, on_epoch=True, prog_bar=False)
            for label, value in enumerate(self.acc_all.cpu()(pred_labels.cpu(), truth_labels.cpu())):
                self.log("z_val acc-" + str(label), value,  on_epoch=True, prog_bar=False)

        self.log("val loss sum", summed_loss,  on_epoch=True, prog_bar=True)

        self.log_custom_metrics(metric_results, "val")

        self.log("val_ref", ref_metric)


    def configure_optimizers(self):
        optimizer = self.hyperparameter["optimizer"](self.parameters(), lr=self.hyperparameter["learning_rate"])

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hyperparameter["warmup_steps"],
            num_training_steps=self.hyperparameter["training_steps"]
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )
        return optimizer


class LinearProbingModel(SkeletonProbingModel):
    def __init__(self, hyperparameter: Dict, ):

        super().__init__(hyperparameter)


        self.layers = ModuleList([])
        self.layers.append(torch.nn.Dropout(self.hyperparameter["dropout"]))

        input_dim = self.hyperparameter["input_dim"]
        if self.hyperparameter["num_hidden_layers"] > 0:
            for i in range(self.hyperparameter["num_hidden_layers"]):
                self.layers.append(torch.nn.Linear(input_dim, self.hyperparameter["hidden_dim"]))
                self.layers.append(torch.nn.LeakyReLU())
                self.layers.append(torch.nn.Dropout(self.hyperparameter["dropout"]))

                input_dim = self.hyperparameter["hidden_dim"]


        self.layers.append(torch.nn.Linear(input_dim, self.hyperparameter["num_labels"]))


    def forward(self, x:torch.Tensor):
        for l in self.layers:
            x = l(x)
        pred = x
        return pred

    def get_loss(self, pred, y):
        return self.loss(pred, y)

    def log_custom_metrics(self, metric_results, set):
        pass



class AmnesicLinearProbingModel(SkeletonProbingModel):
    def __init__(self, hyperparameter: Dict, projections: Dict):

        super().__init__(hyperparameter)


        input_dim = self.hyperparameter["input_dim"]
        self.layers = ModuleList([])

        for projection in projections:
            projection_layer = torch.nn.Linear(in_features=input_dim, out_features=input_dim, bias=False)
            projection_layer._parameters["weight"] = torch.from_numpy(projection).float()
            for _, param in projection_layer._parameters.items():
                if param is not None:
                    param.requires_grad = False
            self.layers.append(projection_layer)

        self.layers.append(torch.nn.Dropout(self.hyperparameter["dropout"]))

        if self.hyperparameter["num_hidden_layers"] > 0:
            for i in range(self.hyperparameter["num_hidden_layers"]):
                self.layers.append(torch.nn.Linear(input_dim, self.hyperparameter["hidden_dim"]))
                self.layers.append(torch.nn.LeakyReLU())
                self.layers.append(torch.nn.Dropout(self.hyperparameter["dropout"]))

                input_dim = self.hyperparameter["hidden_dim"]



        self.layers.append(torch.nn.Linear(input_dim, self.hyperparameter["num_labels"]))


    def forward(self, x:torch.Tensor):
        for l in self.layers:
            x = l(x)
        pred = x
        return pred


    def layer_output(self, x:torch.Tensor, target_layer:int):
        for i, l in enumerate(self.layers):
            if i >= target_layer:
                break
            else:
                x = l(x)
        return x

    def get_loss(self, pred, y):
        return self.loss(pred, y)

    def log_custom_metrics(self, metric_results, set):
        pass


class ScalarMixProbingModel(SkeletonProbingModel):
    def __init__(self, hyperparameter: Dict):
        super().__init__(hyperparameter)

        self.mix_size = hyperparameter["mix_size"]

        self.layers = ModuleList([])
        self.layers.append(
            ScalarMix(
                mixture_size=self.mix_size,
            )
        )

        input_dim = self.hyperparameter["input_dim"]
        if self.hyperparameter["num_hidden_layers"] > 0:
            for i in range(self.hyperparameter["num_hidden_layers"]):
                self.layers.append(torch.nn.Linear(input_dim, self.hyperparameter["hidden_dim"]))
                self.layers.append(torch.nn.LeakyReLU())
                self.layers.append(torch.nn.Dropout(self.hyperparameter["dropout"]))

                input_dim = self.hyperparameter["hidden_dim"]

        self.layers.append(torch.nn.Linear(input_dim, self.hyperparameter["num_labels"]))

    def forward(self, x:List[torch.Tensor]):
        for l in self.layers:
            x = l(x)
        pred = x
        return pred

    def get_loss(self, pred, y):
        return self.loss(pred, y)

    def log_custom_metrics(self, metric_results, set):
        pass

        #if set == "val":
        #    scalar_mix = self.layers[0]
        #    weights = torch.softmax(torch.tensor(scalar_mix.scalar_parameters), dim=0)
        #    for layer, weight in enumerate(weights):
        #        weight = float(weight.detach().cpu())
        #       self.log("z_layer-" + str(layer) + "-weight", weight,  on_epoch=True, prog_bar=False)

