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

import codecs as cs
import random
from os.path import join as pjoin

import cv2
import numpy as np
import torch
from einops import rearrange
from torch.utils import data
from tqdm import tqdm


class TextMotionPatchDataset(data.Dataset):
    def __init__(
        self,
        cfg,
        mean,
        std,
        split_file,
        eval_mode=False,
        patch_size=16,
        fps=None,
    ):
        self.cfg = cfg
        self.eval_mode = eval_mode
        self.max_motion_length = cfg.dataset.max_motion_length
        self.patch_size = patch_size
        self.fps = fps

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(cfg.dataset.motion_dir, name + ".npy"))
                if len(motion.shape) != 3:
                    continue
                if np.isnan(motion).any():
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(cfg.dataset.text_dir, name + ".txt")) as f:
                    for index, line in enumerate(f.readlines()):
                        if eval_mode and index >= 1:
                            continue
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            n_motion = motion[
                                int(f_tag * cfg.dataset.fps) : int(
                                    to_tag * cfg.dataset.fps
                                )
                            ]

                            new_name = (
                                random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                            )
                            while new_name in data_dict:
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                            data_dict[new_name] = {
                                "motion": n_motion,
                                "length": len(n_motion),
                                "text": [text_dict],
                            }
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        )

        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = name_list

        if self.cfg.dataset.dataset_name == "KIT-ML":
            self.kinematic_chain = [
                [0, 11, 12, 13, 14, 15],
                [0, 16, 17, 18, 19, 20],
                [0, 1, 2, 3, 4],
                [3, 5, 6, 7],
                [3, 8, 9, 10],
            ]
        else:
            self.kinematic_chain = [
                [0, 2, 5, 8, 11],
                [0, 1, 4, 7, 10],
                [0, 3, 6, 9, 12, 15],
                [9, 14, 17, 19, 21],
                [9, 13, 16, 18, 20],
            ]

        for key, item in tqdm(data_dict.items()):
            motion = data_dict[key]["motion"]
            if self.cfg.dataset.dataset_name == "KIT-ML" and self.fps is not None:
                motion = self._subsample_to_20fps(motion, self.cfg.dataset.fps)

            motion = (motion - self.mean[np.newaxis, ...]) / self.std[np.newaxis, ...]

            motion = self.use_kinematic(motion)

            data_dict[key]["pre_motion"] = motion
            data_dict[key]["length"] = motion.shape[0]

    def real_len(self):
        return len(self.data_dict)

    def _subsample_to_20fps(self, orig_ft, orig_fps):
        T, n_j, _ = orig_ft.shape
        out_fps = 20.0
        # Matching the sub-sampling used for rendering
        if int(orig_fps) % int(out_fps):
            sel_fr = np.floor(orig_fps / out_fps * np.arange(int(out_fps))).astype(int)
            n_duration = int(T / int(orig_fps))
            t_idxs = []
            for i in range(n_duration):
                t_idxs += list(i * int(orig_fps) + sel_fr)
            if int(T % int(orig_fps)):
                last_sec_frame_idx = n_duration * int(orig_fps)
                t_idxs += [
                    x + last_sec_frame_idx for x in sel_fr if x + last_sec_frame_idx < T
                ]
        else:
            t_idxs = np.arange(0, T, orig_fps / out_fps, dtype=int)

        ft = orig_ft[t_idxs, :, :]
        return ft

    def use_kinematic(self, motion):
        if self.patch_size == 16:
            motion_ = np.zeros(
                (motion.shape[0], len(self.kinematic_chain) * 16, motion.shape[2]),
                float,
            )
            for i_frames in range(motion.shape[0]):
                for i, kinematic_chain in enumerate(self.kinematic_chain):
                    joint_parts = motion[i_frames, kinematic_chain]
                    joint_parts = joint_parts.reshape(1, -1, 3)
                    joint_parts = cv2.resize(
                        joint_parts, (16, 1), interpolation=cv2.INTER_LINEAR
                    )
                    motion_[i_frames, 16 * i : 16 * (i + 1)] = joint_parts[0]

        else:
            raise NotImplementedError

        return motion_

    def __len__(self):
        return self.real_len() * self.cfg.dataset.times

    def __getitem__(self, item):
        idx = item % self.real_len()
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["pre_motion"], data["length"], data["text"]
        # Randomly select a caption
        if self.eval_mode:
            caption = text_list[0]["caption"]
        else:
            text_data = random.choice(text_list)
            caption = text_data["caption"]

        max_motion_length = self.max_motion_length
        if m_length >= self.max_motion_length:
            idx = (
                random.randint(0, len(motion) - max_motion_length)
                if not self.eval_mode
                else 0
            )
            motion = motion[idx : idx + max_motion_length]
            m_length = max_motion_length
        else:
            if self.cfg.preprocess.padding:
                padding_len = max_motion_length - m_length
                D = motion.shape[1]
                C = motion.shape[2]
                padding_zeros = np.zeros((padding_len, D, C), dtype=np.float32)
                motion = np.concatenate((motion, padding_zeros), axis=0)

        motion = torch.tensor(motion).float()
        motion = rearrange(motion, "t j c -> c t j")

        return caption, motion, m_length, item
