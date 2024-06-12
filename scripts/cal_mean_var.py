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

import os
from os.path import join as pjoin

import numpy as np
from tqdm import tqdm

for dataset in ["HumanML3D", "KIT-ML"]:
    file_list = os.listdir(f"data/{dataset}/new_joints/")

    data_list = []
    for file in tqdm(file_list):
        data = np.load(pjoin(f"data/{dataset}/new_joints/", file))
        if np.isnan(data).any():
            print(file)
            continue
        if len(data.shape) != 3:
            print(file, data.shape)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)

    np.save(pjoin(f"data/{dataset}/", "Mean_raw.npy"), Mean)
    np.save(pjoin(f"data/{dataset}/", "Std_raw.npy"), Std)
