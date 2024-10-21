import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.Functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from pythae.data.datasets import DatasetOutput
from pythae.models import VAEConfig, VAE, BetaVAEConfig, BetaVAE, BetaTCVAEConfig, BetaTCVAE
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder
from pythae.models.nn.benchmarks.utils import ResBlock
from pythae.trainers import BaseTrainer, BaseTrainerConfig
# from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import argparse

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
            # nn.LeakyReLU()
            nn.PReLU(device="cuda")
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU()
            nn.PReLU(device="cuda")
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
        return nn.PReLU(device="cuda")(conv2_op.to("cuda:0") + skip_op.to("cuda:0"))
    
class DeconvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU()
            nn.PReLU(device="cuda")
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU()
            nn.PReLU(device="cuda")
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
        return nn.PReLU(device="cuda")(conv2_op.to("cuda:0") + skip_op.to("cuda:0"))

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
            embedding=self.embedding(F.prelu(self.lin2(F.prelu(self.lin1(h1))))),
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
        h1 = F.prelu(self.fc3(F.prelu(self.fc2(F.prelu(self.fc1(z)))))).reshape(z.shape[0], 1024, 1, 1)
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


# for i in model.named_parameters():
#     print(f"{i[0]} -> {i[1].device}")

from pythae.trainers.training_callbacks import WandbCallback

from pythae.pipelines import TrainingPipeline

def multivae_pipelines(vae_type):
    if vae_type == "vanilla_vae":
        vae_config = VAEConfig(
            input_dim=(1, 256, 256),
            latent_dim=8,
            # reconstruction_loss="bce"
        )
        
        vae_dataset = NewDSprites(
	    data_file_path="../../data/new_dSprites_256x256_imgs.npy",
	    transforms=img_transforms,
	)
        
        training_config = BaseTrainerConfig(
            output_dir='res_vanilla_vae_multivae_train_v2',
            # train_dataloader_num_workers=8,
            # eval_dataloader_num_workers=8,
            learning_rate=1e-4,
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            steps_saving=15,
            optimizer_cls="Adam",
            # optimizer_params={"weight_decay": 0.05},
            scheduler_cls="ReduceLROnPlateau",
            scheduler_params={"patience": 5, "factor": 0.5},
            no_cuda=False,
            num_epochs=150
        )

        vae_encoder = Encoder_Res_VAE_new_dSprites(vae_config)
        vae_decoder = Decoder_Res_AE_new_dSprites(vae_config)

        vae_model = VAE(model_config=vae_config, encoder=vae_encoder, decoder=vae_decoder).to("cuda:0")
        
        print("Training Vanilla VAE")
        print(vae_model)
        
        callbacks = []
        wandb_cb = WandbCallback()
        wandb_cb.setup(
            training_config,
            model_config=vae_config,
            project_name="multimodal_llm_robustness_exp_resnets",
            entity_name="sujithvemi-Synechron",
        )

        callbacks.append(wandb_cb)
        
        pipeline = TrainingPipeline(
            model=vae_model,
            training_config=training_config
        )

        pipeline(
            train_data=vae_dataset,
            callbacks=callbacks
        )
    elif vae_type == "beta_vae":
        beta_vae_config = BetaVAEConfig(
            input_dim=(1, 256, 256),
            latent_dim=8,
            beta = 4,
            # reconstruction_loss="bce"
        )
        
        beta_vae_dataset = NewDSprites(
	    data_file_path="../../data/new_dSprites_256x256_imgs.npy",
	    transforms=img_transforms,
	)
        
        training_config = BaseTrainerConfig(
            output_dir='res_beta_vae_multivae_train_v2_corrected_model',
            # train_dataloader_num_workers=8,
            # eval_dataloader_num_workers=8,
            learning_rate=1e-4,
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            steps_saving=15,
            optimizer_cls="Adam",
            # optimizer_params={"weight_decay": 0.05},
            scheduler_cls="ReduceLROnPlateau",
            scheduler_params={"patience": 5, "factor": 0.5},
            no_cuda=False,
            num_epochs=150
        )

        beta_vae_encoder = Encoder_Res_VAE_new_dSprites(beta_vae_config)
        beta_vae_decoder = Decoder_Res_AE_new_dSprites(beta_vae_config)

        beta_vae_model = BetaVAE(model_config=beta_vae_config, encoder=beta_vae_encoder, decoder=beta_vae_decoder).to("cuda:0")
        
        print("Training Beta VAE")
        print(beta_vae_model)
        
        callbacks = []
        wandb_cb = WandbCallback()
        wandb_cb.setup(
            training_config,
            model_config=beta_vae_config,
            project_name="multimodal_llm_robustness_exp_resnets",
            entity_name="sujithvemi-Synechron",
        )

        callbacks.append(wandb_cb)
        
        pipeline = TrainingPipeline(
            model=beta_vae_model,
            training_config=training_config
        )

        pipeline(
            train_data=beta_vae_dataset,
            callbacks=callbacks
        )
    elif vae_type == "beta_tcvae":
        beta_tcvae_config = BetaTCVAEConfig(
            input_dim=(1, 256, 256),
            latent_dim=8,
            beta = 6,
            # reconstruction_loss="bce"
        )
        
        beta_tcvae_dataset = NewDSprites(
	    data_file_path="../../data/new_dSprites_256x256_imgs.npy",
	    transforms=img_transforms,
	)
        
        training_config = BaseTrainerConfig(
            output_dir='res_beta_tcvae_multivae_train_v2_corrected_model',
            # train_dataloader_num_workers=8,
            # eval_dataloader_num_workers=8,
            learning_rate=1e-4,
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            steps_saving=15,
            optimizer_cls="Adam",
            # optimizer_params={"weight_decay": 0.05},
            scheduler_cls="ReduceLROnPlateau",
            scheduler_params={"patience": 5, "factor": 0.5},
            no_cuda=False,
            num_epochs=150
        )

        beta_tcvae_encoder = Encoder_Res_VAE_new_dSprites(beta_tcvae_config)
        beta_tcvae_decoder = Decoder_Res_AE_new_dSprites(beta_tcvae_config)

        beta_tcvae_model = BetaTCVAE(model_config=beta_tcvae_config, encoder=beta_tcvae_encoder, decoder=beta_tcvae_decoder).to("cuda:0")
        
        print("Training Beta TCVAE")
        print(beta_tcvae_model)
        
        callbacks = []
        wandb_cb = WandbCallback()
        wandb_cb.setup(
            training_config,
            model_config=beta_tcvae_config,
            project_name="multimodal_llm_robustness_exp_resnets",
            entity_name="sujithvemi-Synechron",
        )

        callbacks.append(wandb_cb)
        
        pipeline = TrainingPipeline(
            model=beta_tcvae_model,
            training_config=training_config
        )

        pipeline(
            train_data=beta_tcvae_dataset,
            callbacks=callbacks
        )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    args = parser.parse_args()
    model_name = args.model
    multivae_pipelines(model_name)
    #with Pool(3) as p:
     #   p.map(multivae_pipelines, ["vanilla_vae", "beta_vae", "beta_tcvae"])
