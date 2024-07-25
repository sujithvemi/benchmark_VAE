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

class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(device=torch.device("cuda"))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(device=torch.device("cuda"))
        )
        self.ds = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2, padding=0),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        skip_op = self.ds(x)
        conv1_op = self.conv1(x)
        conv2_op = self.conv2(conv1_op)
        # print("conv1_op device: ", conv1_op.get_device())
        # print("conv2_op device: ", conv2_op.get_device())
        # print("skip_op device: ", skip_op.get_device())
        return nn.PReLU(device=torch.device("cuda"))(conv2_op.to("cuda:0") + skip_op.to("cuda:0"))
    
class DeconvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(device=torch.device("cuda"))
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(device=torch.device("cuda"))
        )
        self.us = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 1, 2, padding=0, output_padding=1),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        # x.to("cuda")
        skip_op = self.us(x)
        conv1_op = self.conv1(x)
        conv2_op = self.conv2(conv1_op)
        return nn.PReLU(device=torch.device("cuda"))(conv2_op.to("cuda:0") + skip_op.to("cuda:0"))

class Encoder_Res_VAE_new_dSprites(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 256, 256)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        self.conv_layers = nn.Sequential(
            ConvResBlock(1, 32),
            ConvResBlock(32, 64),
            ConvResBlock(64, 128),
            ConvResBlock(128, 256),
            ConvResBlock(256, 512),
            ConvResBlock(512, 512),
            ConvResBlock(512, 512),
            ConvResBlock(512, 1024)
        )
        self.lin1 = nn.Linear(1024, 512)
        self.lin2 = nn.Linear(512, 256)
        self.embedding = nn.Linear(256, args.latent_dim)
        self.log_var = nn.Linear(256, args.latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        output = ModelOutput(
            embedding=self.embedding(self.lin2(self.lin1(h1))),
            log_covariance=self.log_var(self.lin2(self.lin1(h1)))
        )
        return output

class Decoder_Res_AE_new_dSprites(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.input_dim = (1, 256, 256)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        self.fc1 = nn.Linear(args.latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.deconv_layers = nn.Sequential(
            DeconvResBlock(1024, 512),
            DeconvResBlock(512, 512),
            DeconvResBlock(512, 512),
            DeconvResBlock(512, 256),
            DeconvResBlock(256, 128),
            DeconvResBlock(128, 64),
            DeconvResBlock(64, 32),
            DeconvResBlock(32, 1),
        )

    def forward(self, z: torch.Tensor):
        h1 = self.fc3(self.fc2(self.fc1(z))).reshape(z.shape[0], 1024, 1, 1)
        output = ModelOutput(reconstruction=self.deconv_layers(h1))

        return output

class NewDSprites(Dataset):
    def __init__(self, data_file_path=None, transforms=None):
        self.imgs_arr = np.load(data_file_path, mmap_mode="r")
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs_arr)

    def __getitem__(self, idx):
        img = np.expand_dims(self.imgs_arr[idx, :, :], axis=-1)
        img_copy = np.array(img, copy=True)
        # print(np.unique(img_copy))
        if self.transforms is not None:
            img_copy = self.transforms(img_copy)
        return DatasetOutput(data=img_copy)

img_transforms = transforms.Compose([transforms.ToTensor()])

dataset = NewDSprites(
    data_file_path="../../data/new_dSprites_256x256_imgs.npy",
    transforms=img_transforms,
)

model_config = VAEConfig(
    input_dim=(1, 256, 256),
    latent_dim=10,
    # reconstruction_loss="bce"
)

encoder = Encoder_Res_VAE_new_dSprites(model_config)
decoder = Decoder_Res_AE_new_dSprites(model_config)

model = VAE(model_config=model_config, encoder=encoder, decoder=decoder).to("cuda:0")

print(model)

# for i in model.named_parameters():
#     print(f"{i[0]} -> {i[1].device}")

training_config = BaseTrainerConfig(
    output_dir='res_vae_new_dSprites_v1',
    # train_dataloader_num_workers=8,
    # eval_dataloader_num_workers=8,
    learning_rate=1e-3,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    steps_saving=5,
    optimizer_cls="Adam",
    # optimizer_params={"weight_decay": 0.05},
    scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 5, "factor": 0.1},
    no_cuda=False,
    num_epochs=100
)

from pythae.trainers.training_callbacks import WandbCallback

callbacks = []
wandb_cb = WandbCallback()
wandb_cb.setup(
    training_config,
    model_config=model_config,
    project_name="multimodal_llm_robustness_exp_resnets",
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