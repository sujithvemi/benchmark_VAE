# import argparse
import logging
import os
import time

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from pythae.data.datasets import DatasetOutput
from pythae.models import VQVAE, VQVAEConfig, VAEConfig, VAE
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder
from pythae.models.nn.benchmarks.utils import ResBlock
from pythae.trainers import BaseTrainer, BaseTrainerConfig
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

# PATH = os.path.dirname(os.path.abspath("."))

class Encoder_Conv_VAE_new_dSprites(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 256, 256)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.n_channels, 32, 4, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, 4, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, 4, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            # nn.Conv2d(256, 512, 4, 2, padding=1),
            # nn.BatchNorm2d(512),
            # nn.PReLU(),
            nn.Conv2d(256, 512, 2, 1, padding=0),
            nn.BatchNorm2d(512),
            nn.PReLU(),
        )

        self.lin = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
        )
        self.embedding = nn.Linear(256, args.latent_dim)
        self.log_var = nn.Linear(256, args.latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        output = ModelOutput(
            embedding=self.embedding(self.lin(h1)),
            log_covariance=self.log_var(self.lin(h1))
        )
        return output

class Decoder_Conv_AE_new_dSprites(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.input_dim = (1, 256, 256)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        self.fc1 = nn.Sequential(
            nn.Linear(args.latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 1, padding=0),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.ConvTranspose2d(256, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.ConvTranspose2d(32, self.n_channels, 4, 2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        h1 = self.fc2(self.fc1(z)).reshape(z.shape[0], 512, 1, 1)
        output = ModelOutput(reconstruction=self.deconv_layers(h1))

        return output

class NewDSprites(Dataset):
    def __init__(self, data_file_path=None, transforms=None):
        self.imgs_arr = np.load(data_file_path, allow_pickle=True)["imgs"]
        # if is_train:
        #     self.imgs_arr = self.imgs_arr[train_idxs, :]
        # else:
        #     self.imgs_path = self.imgs_path[60000:]
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs_arr)

    def __getitem__(self, idx):
        img = np.expand_dims(self.imgs_arr[idx, :, :], axis=-1)
        if self.transforms is not None:
            img = self.transforms(img)
        return DatasetOutput(data=img)

img_transforms = transforms.Compose([transforms.ToTensor()])

dataset = NewDSprites(
    data_file_path="../../data/new_dSprites_256x256.npz",
    transforms=img_transforms,
)

model_config = VAEConfig(
    input_dim=(1, 256, 256), latent_dim=10, reconstruction_loss="bce"
)

encoder = Encoder_Conv_VAE_new_dSprites(model_config)
decoder = Decoder_Conv_AE_new_dSprites(model_config)

model = VAE(model_config=model_config, encoder=encoder, decoder=decoder)

print(model)

training_config = BaseTrainerConfig(
    output_dir='conv_vae_new_dSprites_v2',
    # train_dataloader_num_workers=8,
    # eval_dataloader_num_workers=8,
    learning_rate=1e-4,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    steps_saving=2,
    optimizer_cls="AdamW",
    optimizer_params={"weight_decay": 0.05},
    scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 5, "factor": 0.5},
    # no_cuda=False,
    num_epochs=50
)

from pythae.trainers.training_callbacks import WandbCallback

callbacks = []
wandb_cb = WandbCallback()
wandb_cb.setup(
    training_config,
    model_config=model_config,
    project_name="multimodal_llm_robustness_exp",
    entity_name="sujithvemi-Synechron",
)

callbacks.append(wandb_cb)

from pythae.pipelines import TrainingPipeline
pipeline = TrainingPipeline(
model=model,
training_config=training_config)

pipeline(
    train_data=dataset,
    callbacks=callbacks
)