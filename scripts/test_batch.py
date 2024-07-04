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

import logging
import os
import sys
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from datasets import TextMotionPatchDataset
from models.clip import ClipModel

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name="test_config", config_path="../conf")
def main(cfg: DictConfig) -> None:
    saved_cfg = OmegaConf.load(pjoin(cfg.checkpoints_dir, ".hydra/config.yaml"))
    print(OmegaConf.to_yaml(saved_cfg))
    test_dataloader = prepare_test_dataset(saved_cfg)
    model, tokenizer = prepare_test_model(saved_cfg)
    eval(saved_cfg, test_dataloader, model, tokenizer)


def prepare_test_dataset(cfg):
    mean = np.load(pjoin(cfg.dataset.data_root, "Mean_raw.npy"))
    std = np.load(pjoin(cfg.dataset.data_root, "Std_raw.npy"))

    if cfg.eval.eval_train:
        test_split_file = pjoin(cfg.dataset.data_root, "train.txt")
    else:
        test_split_file = pjoin(cfg.dataset.data_root, "test.txt")
    test_dataset = TextMotionPatchDataset(
        cfg,
        mean,
        std,
        test_split_file,
        eval_mode=True,
        patch_size=cfg.train.patch_size,
        fps=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=16
    )
    return test_dataloader


def prepare_test_model(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    motion_encoder_alias = cfg.model.motion_encoder
    text_encoder_alias = cfg.model.text_encoder
    motion_embedding_dims: int = 768
    text_embedding_dims: int = 768
    projection_dims: int = 256

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_alias)

    model = ClipModel(
        motion_encoder_alias=motion_encoder_alias,
        text_encoder_alias=text_encoder_alias,
        motion_embedding_dims=motion_embedding_dims,
        text_embedding_dims=text_embedding_dims,
        projection_dims=projection_dims,
        patch_size=cfg.train.patch_size,
    )

    if cfg.eval.use_best_model:
        model_path = pjoin(cfg.checkpoints_dir, "best_model.pt")
    else:
        model_path = pjoin(cfg.checkpoints_dir, "last_model.pt")

    print(model_path)
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.to(device)

    return model, tokenizer


def eval(cfg, test_dataloader, model, tokenizer=None, verbose=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_imgs_feat = []
    all_captions_feat = []

    all_img_idxs = []
    all_captions = []

    step = 0
    with torch.no_grad():
        model.eval()
        test_pbar = tqdm(test_dataloader, leave=False)
        for batch in test_pbar:
            step += 1
            texts, motions, m_length, img_indexs = batch
            motions = motions.to(device)

            texts_token = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(device)

            motion_features = model.encode_motion(motions)
            text_features = model.encode_text(texts_token)

            # normalized features
            motion_features = motion_features / motion_features.norm(
                dim=1, keepdim=True
            )
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            for i in range(motion_features.size(0)):
                all_imgs_feat.append(motion_features[i].cpu().numpy())
                all_captions_feat.append(text_features[i].cpu().numpy())

                all_captions.append(texts[i])
                all_img_idxs.append(img_indexs[i].item())

    all_captions = np.array(all_captions)

    all_imgs_feat = np.vstack(all_imgs_feat)
    all_captions_feat = np.vstack(all_captions_feat)

    stats_t2m = [0, 0, 0, 0, 0, 0]
    stats_m2t = [0, 0, 0, 0, 0, 0]

    batch_size = 32
    num_batch = all_imgs_feat.shape[0] // batch_size
    for batch_idx in tqdm(range(num_batch)):
        all_captions_feat_batch = all_captions_feat[batch_size*batch_idx:batch_size*(batch_idx+1)]
        all_imgs_feat_batch = all_imgs_feat[batch_size*batch_idx:batch_size*(batch_idx+1)]
        all_captions_batch = all_captions[batch_size*batch_idx:batch_size*(batch_idx+1)]

        dataset_pair = dict()
        for img_idx, caption in enumerate(all_captions_batch):
            dataset_pair[img_idx] = np.where(all_captions_batch == caption)[0]

        # match test queries to target motions, get nearest neighbors
        sims_t2m = 100 * all_captions_feat_batch.dot(all_imgs_feat_batch.T)

        t2m_r1 = 0
        # Text->Motion
        ranks = np.zeros(sims_t2m.shape[0])
        for index, score in enumerate(sims_t2m):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in dataset_pair[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        for j, k in enumerate([1, 2, 3, 5, 10]):
            # Compute metrics
            r = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
            stats_t2m[j] += r

        stats_t2m[-1] += np.median(ranks) + 1

        # match motions queries to target texts, get nearest neighbors
        sims_m2t = sims_t2m.T

        m2t_r1 = 0
        # Motion->Text
        ranks = np.zeros(sims_m2t.shape[0])
        for index, score in enumerate(sims_m2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in dataset_pair[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        for j, k in enumerate([1, 2, 3, 5, 10]):
            # Compute metrics
            r = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
            stats_m2t[j] += r

        stats_m2t[-1] += np.median(ranks) + 1

    for j, k in enumerate([1, 2, 3, 5, 10]):
        # Compute metrics
        if verbose:
            log.info(f't2m_recall_top{k}_correct_composition: {stats_t2m[j]/num_batch}')
    if verbose:
        log.info(f't2m_recall_median_correct_composition: {stats_t2m[-1]/num_batch}')

    for j, k in enumerate([1, 2, 3, 5, 10]):
        # Compute metrics
        if verbose:
            log.info(f'm2t_recall_top{k}_correct_composition: {stats_m2t[j]/num_batch}')
    if verbose:
        log.info(f'm2t_recall_median_correct_composition: {stats_m2t[-1]/num_batch}')


if __name__ == "__main__":
    main()
