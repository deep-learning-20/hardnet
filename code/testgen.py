import argparse

import numpy as np

import os

import shutil

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from tqdm import tqdm

import warnings

from lib.dataset import MegaDepthDataset
from lib.exceptions import NoGradientError
from lib.loss import loss_function
from lib.model import D2Net


training_dataset = MegaDepthDataset(
    scene_list_path='megadepth_utils/train_scenes.txt',
    preprocessing='caffe',
    pairs_per_scene=32
)

training_dataloader = DataLoader(
    training_dataset,
    batch_size=8,
    num_workers=1
)

training_dataset.build_dataset()

print("dataset size is ", len(training_dataset))
data=[]
for i in tqdm(range(16)):
    data.append(training_dataset.__getitem__(i))

import pickle

writefile = open("/cluster/scratch/rthakur/dl/test/testgen.pkl", "wb")
pickle.dump(data, writefile)
writefile.close()

