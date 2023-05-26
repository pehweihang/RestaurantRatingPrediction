# =========================================================================
# Copyright (C) 2023. FuxiCTR Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import logging
import sys

import numpy as np
import torch
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.models import BaseModel
from torch import nn
from tqdm import tqdm


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class FinalMLP(BaseModel):
    def __init__(
        self,
        feature_map,
        model_id="FinalMLP",
        gpu=-1,
        learning_rate=1e-3,
        embedding_dim=10,
        mlp1_hidden_units=[64, 64, 64],
        mlp1_hidden_activations="ReLU",
        mlp1_dropout=0,
        mlp1_batch_norm=False,
        mlp2_hidden_units=[64, 64, 64],
        mlp2_hidden_activations="ReLU",
        mlp2_dropout=0,
        mlp2_batch_norm=False,
        use_fs=True,
        fs_hidden_units=[64],
        fs1_context=[],
        fs2_context=[],
        num_heads=1,
        embedding_regularizer=None,
        net_regularizer=None,
        **kwargs
    ):
        super(FinalMLP, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs
        )
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        feature_dim = embedding_dim * feature_map.num_fields
        self.mlp1 = MLP_Block(
            input_dim=feature_dim,
            output_dim=None,
            hidden_units=mlp1_hidden_units,
            hidden_activations=mlp1_hidden_activations,
            output_activation=None,
            dropout_rates=mlp1_dropout,
            batch_norm=mlp1_batch_norm,
        )
        self.mlp2 = MLP_Block(
            input_dim=feature_dim,
            output_dim=None,
            hidden_units=mlp2_hidden_units,
            hidden_activations=mlp2_hidden_activations,
            output_activation=None,
            dropout_rates=mlp2_dropout,
            batch_norm=mlp2_batch_norm,
        )
        self.use_fs = use_fs
        if self.use_fs:
            self.fs_module = FeatureSelection(
                feature_map,
                feature_dim,
                embedding_dim,
                fs_hidden_units,
                fs1_context,
                fs2_context,
            )
        self.fusion_module = InteractionAggregation(
            mlp1_hidden_units[-1],
            mlp2_hidden_units[-1],
            output_dim=1,
            num_heads=num_heads,
        )
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.loss_fn = RMSELoss()
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        flat_emb = self.embedding_layer(X).flatten(start_dim=1)
        if self.use_fs:
            feat1, feat2 = self.fs_module(X, flat_emb)
        else:
            feat1, feat2 = flat_emb, flat_emb
        y_pred = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def fit(
        self,
        data_generator,
        epochs=1,
        validation_data=None,
        max_gradient_norm=10.0,
        **kwargs
    ):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        logging.info(
            "Start training: {} batches/epoch".format(self._steps_per_epoch)
        )
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self.train_epoch(data_generator)
            if self._stop_training:
                break
            else:
                logging.info(
                    "************ Epoch={} end ************".format(
                        self._epoch_index + 1
                    )
                )
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

    def eval_step(self):
        logging.info(
            "Evaluation @epoch {} - batch {}: ".format(
                self._epoch_index + 1, self._batch_index + 1
            )
        )
        val_logs = self.evaluate(
            self.valid_gen, metrics=self._monitor.get_metrics()
        )
        self.train()

    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []
            if self._verbose > 0:
                data_generator = tqdm(
                    data_generator, disable=False, file=sys.stdout
                )
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(
                    return_dict["y_pred"].data.cpu().numpy().reshape(-1)
                )
                y_true.extend(
                    self.get_labels(batch_data).data.cpu().numpy().reshape(-1)
                )
                if self.feature_map.group_id is not None:
                    group_id.extend(
                        self.get_group_id(batch_data).numpy().reshape(-1)
                    )
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None
            val_logs = {"RMSE": np.sqrt(((y_true - y_pred) ** 2).mean())}
            logging.info(
                "[Metrics] "
                + " - ".join(
                    "{}: {:.6f}".format(k, v) for k, v in val_logs.items()
                )
            )
            return val_logs

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        loss = self.loss_fn(return_dict["y_pred"], y_true)
        return loss

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        loss = self.get_total_loss(batch_data)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0
        self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(
                data_generator, disable=False, file=sys.stdout
            )
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss = self.train_step(batch_data)
            train_loss += loss.item()
            if self._total_steps % self._eval_steps == 0:
                logging.info(
                    "Train loss: {:.6f}".format(train_loss / self._eval_steps)
                )
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break


class FeatureSelection(nn.Module):
    def __init__(
        self,
        feature_map,
        feature_dim,
        embedding_dim,
        fs_hidden_units=[],
        fs1_context=[],
        fs2_context=[],
    ):
        super(FeatureSelection, self).__init__()
        self.fs1_context = fs1_context
        if len(fs1_context) == 0:
            self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs1_ctx_emb = FeatureEmbedding(
                feature_map,
                embedding_dim,
                required_feature_columns=fs1_context,
            )
        self.fs2_context = fs2_context
        if len(fs2_context) == 0:
            self.fs2_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs2_ctx_emb = FeatureEmbedding(
                feature_map,
                embedding_dim,
                required_feature_columns=fs2_context,
            )
        self.fs1_gate = MLP_Block(
            input_dim=embedding_dim * max(1, len(fs1_context)),
            output_dim=feature_dim,
            hidden_units=fs_hidden_units,
            hidden_activations="ReLU",
            output_activation="Sigmoid",
            batch_norm=False,
        )
        self.fs2_gate = MLP_Block(
            input_dim=embedding_dim * max(1, len(fs2_context)),
            output_dim=feature_dim,
            hidden_units=fs_hidden_units,
            hidden_activations="ReLU",
            output_activation="Sigmoid",
            batch_norm=False,
        )

    def forward(self, X, flat_emb):
        if len(self.fs1_context) == 0:
            fs1_input = self.fs1_ctx_bias.repeat(flat_emb.size(0), 1)
        else:
            fs1_input = self.fs1_ctx_emb(X).flatten(start_dim=1)
        gt1 = self.fs1_gate(fs1_input) * 2
        feature1 = flat_emb * gt1
        if len(self.fs2_context) == 0:
            fs2_input = self.fs2_ctx_bias.repeat(flat_emb.size(0), 1)
        else:
            fs2_input = self.fs2_ctx_emb(X).flatten(start_dim=1)
        gt2 = self.fs2_gate(fs2_input) * 2
        feature2 = flat_emb * gt2
        return feature1, feature2


class InteractionAggregation(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert (
            x_dim % num_heads == 0 and y_dim % num_heads == 0
        ), "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(
            torch.Tensor(
                num_heads * self.head_x_dim * self.head_y_dim, output_dim
            )
        )
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        xy = torch.matmul(
            torch.matmul(
                head_x.unsqueeze(2),
                self.w_xy.view(self.num_heads, self.head_x_dim, -1),
            ).view(-1, self.num_heads, self.output_dim, self.head_y_dim),
            head_y.unsqueeze(-1),
        ).squeeze(-1)
        output += xy.sum(dim=1)
        return output
