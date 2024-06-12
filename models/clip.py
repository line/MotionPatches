"""
Copyright 2024 LY Corporation
LY Corporation licenses this file to you under the CC BY-NC 4.0
(the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:
    https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""

import numpy as np
import timm
import torch
import transformers
from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        return self.layer_norm(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, trainable: bool = True) -> None:
        super().__init__()
        self.text_model = transformers.AutoModel.from_pretrained(model_name)

        for param in self.text_model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state

        return last_hidden_state[:, self.target_token_idx, :]


class MotionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        trainable: bool = True,
        patch_size=16,
    ) -> None:
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            img_size=(224, patch_size * 5),
        )

        for param in self.model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, x):
        return self.model(x)


class ClipModel(nn.Module):
    def __init__(
        self,
        motion_encoder_alias="vit_base_patch16_224_in21k",
        text_encoder_alias="distilbert-base-uncased",
        motion_encoder_pretrained: bool = True,
        motion_encoder_trainable: bool = True,
        text_encoder_trainable: bool = True,
        motion_embedding_dims: int = 768,
        text_embedding_dims: int = 768,
        projection_dims: int = 256,
        dropout: float = 0.5,
        logit: float = 0.07,
        patch_size: int = 16,
    ) -> None:
        super().__init__()

        motion_encoder = MotionEncoder(
            model_name=motion_encoder_alias,
            pretrained=motion_encoder_pretrained,
            trainable=motion_encoder_trainable,
            patch_size=patch_size,
        )
        text_encoder = TextEncoder(
            model_name=text_encoder_alias, trainable=text_encoder_trainable
        )

        self.motion_encoder = motion_encoder
        self.text_encoder = text_encoder

        self.motion_projection = ProjectionHead(
            embedding_dim=motion_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / logit)))

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def encode_motion(self, motion):
        motion_features = self.motion_encoder(motion)
        motion_embeddings = self.motion_projection(motion_features)
        return motion_embeddings

    def encode_text(self, text):
        text_features = self.text_encoder(
            input_ids=text["input_ids"], attention_mask=text["attention_mask"]
        )

        text_embeddings = self.text_projection(text_features)

        return text_embeddings

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits, torch.arange(len(logits), device=logits.device)
        )

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        motion_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + motion_loss) / 2.0

    def forward(self, motion, text, return_loss=False):
        motion_embeds = self.encode_motion(motion)
        text_embeds = self.encode_text(text)

        # normalized features
        motion_embeds = motion_embeds / motion_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, motion_embeds.t()) * logit_scale
        logits_per_motion = logits_per_text.T

        if return_loss:
            return self.clip_loss(logits_per_text)
        else:
            return motion_embeds, text_embeds
