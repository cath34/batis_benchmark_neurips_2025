"""
main training script
To run: python train.py args.config=$CONFIG_FILE_PATH
"""

import os
import hydra
import sys
from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict, cast
import pytorch_lightning as pl
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from src.dataset.dataloader import EbirdVisionDataset
from src.transforms.transforms import get_transforms

#from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils.config_utils import load_opts
from src.dataset.dataloader import get_subset
from src.trainer.utils import get_target_size, get_nb_bands, get_scheduler, init_first_layer_weights, \
    load_from_checkpoint
from src.losses.metrics import get_metrics

from src.trainer.trainer import EbirdDataModule
from src.utils.compute_normalization_stats import *

# Define your Mean-Variance ResNet18 Model.
class ResNet18MeanVariance(nn.Module):
    def __init__(self, output_dim, opts, use_sigmoid_mean=True):
        """
        Args:
            output_dim (int): Number of output components (N).
            use_sigmoid_mean (bool): Whether to apply a sigmoid activation on the mean predictions.
        """
        super(ResNet18MeanVariance, self).__init__()
        self.use_sigmoid_mean = use_sigmoid_mean
        self.resnet = models.resnet18(pretrained=True)
        self.opts = opts 

        if len(self.opts.data.bands) != 3 or len(self.opts.data.env) > 0:
            self.bands = self.opts.data.bands + self.opts.data.env
            orig_channels = self.resnet.conv1.in_channels
            weights = self.resnet.conv1.weight.data.clone()
            self.resnet.conv1 = nn.Conv2d(get_nb_bands(self.bands), 64, kernel_size=(7, 7), stride=(2, 2),
                                             padding=(3, 3), bias=False, )
            # assume first three channels are rgb
            if self.opts.experiment.module.pretrained:
                # self.model.conv1.weight.data[:, :orig_channels, :, :] = weights
                self.resnet.conv1.weight.data = init_first_layer_weights(get_nb_bands(self.bands), weights)
        
        self.feature_extractor = nn.Sequential(
            *list(self.resnet.children())[:-2],       # keep conv layers
            nn.Dropout2d(p=0.2)         # MC‑Dropout (epistemic)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        d = self.resnet.fc.in_features

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, 2*output_dim)
        )

    def forward(self, x):
        # Get combined output vector (shape: [batch_size, 2*output_dim])

        feat = self.feature_extractor(x)
        h = self.pool(feat)
        out = self.head(h)
        
        mean, raw_variance = torch.chunk(out, chunks=2, dim=1)
        
        # Optionally apply sigmoid to the mean if predictions are meant to be in [0,1]
        if self.use_sigmoid_mean:
            mean = torch.sigmoid(mean)
        # Exponentiate the log variance to get a positive variance value

         # Clamp the log_variance to a specific range for numerical stability.
        max_var = mean * (1 - mean)
        variance = max_var * torch.sigmoid(raw_variance)
        log_variance = torch.log(variance + 1e-5)

        return mean, variance, log_variance
        # Double the output dimension: one half for mean and one half for log-variance.
        #self.resnet.fc = nn.Linear(512, 2 * output_dim)
    
    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_path: str,
        output_dim: int,
        opts,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        use_sigmoid_mean: bool = True,
    ) -> "ResNet18MeanVariance":
        """
        Factory method that instantiates **and** populates a `ResNet18MeanVariance`
        from a checkpoint file.

        Args
        ----
        ckpt_path : str
            Path to the .ckpt / .pt / .pth file.
        output_dim : int
            Number of output components (same value you used at training time).
        opts : Namespace | dict‑like
            The same options/config object you passed when the model was created.
        map_location : str | torch.device, optional
            Where to load the weights (default: "cpu").
        strict : bool, optional
            Passed to `load_state_dict`.  If `False`, missing/unexpected keys are
            only warned about (handy when the checkpoint has extra heads, etc.).
        use_sigmoid_mean : bool, optional
            Forwarded to the constructor.

        Returns
        -------
        model : ResNet18MeanVariance
            A model instance with weights loaded.
        """
        # 1) read the file --------------------------------------------------------
        ckpt = torch.load(ckpt_path, map_location=map_location)

        # 2) peel off wrappers Lightning, ignite, etc. ---------------------------
        #    If you saved `trainer.save_checkpoint(...)` you’ll have "state_dict";
        #    if you did `torch.save(model.state_dict())` you already have it.
        state_dict = ckpt.get("state_dict", ckpt)

        # 3) strip the "module." prefix if the model was DataParallel’ed ---------
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        # 4) re‑create the architecture -----------------------------------------
        model = cls(
            output_dim=output_dim,
            opts=opts,
            use_sigmoid_mean=use_sigmoid_mean,
        )

        # 5) load weights --------------------------------------------------------
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)

        if not strict:  # handy debug printout
            if missing:
                print(f"[load_from_checkpoint] ⚠️  Missing keys: {missing}")
            if unexpected:
                print(f"[load_from_checkpoint] ⚠️  Unexpected keys: {unexpected}")

        return model
    
# Define the Gaussian negative log likelihood loss function.
def gaussian_nll_loss(mean, log_variance, target):
    """
    Computes the Gaussian negative log likelihood loss.
    
    For each element, the loss is:
        loss = 0.5 * log_variance + 0.5 * ((target - mean)^2 / exp(log_variance))
        
    The constant term (0.5 * log(2π)) is omitted, as it does not affect optimization.

    Args:
        mean (Tensor): Predicted means.
        log_variance (Tensor): Predicted log variances.
        target (Tensor): Ground truth targets.
        
    Returns:
        loss (Tensor): Mean loss over the batch.
    """

    loss = 0.5 * log_variance + 0.5 * ((target - mean)**2 / torch.exp(log_variance))
    return loss.mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    config_path = parser.parse_args().config
    
    base_dir = os.getcwd()

    config_path = os.path.join(base_dir, config_path)
    default_config = os.path.join(base_dir, "configs/defaults.yaml")

    config = load_opts(config_path, default=default_config)
    global_seed = 1234
    print(global_seed)
    print("SUP")

    config.variables.bioclim_means, config.variables.bioclim_std, config.variables.ped_means, config.variables.ped_std = compute_means_stds_env_vars(
            root_dir=config.data.files.base,
            train_csv=config.data.files.train,
            env=config.data.env,
            env_data_folder=config.data.files.env_data_folder,
            output_file_means=config.data.files.env_means,
            output_file_std=config.data.files.env_stds
        )
    
    config.variables.rgbnir_means, config.variables.rgbnir_std = compute_means_stds_images(
            root_dir=config.data.files.base,
            train_csv=config.data.files.train,
            output_file_means=config.data.files.rgbnir_means,
            output_file_std=config.data.files.rgbnir_stds)
    
    pl.seed_everything(global_seed)

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.save_preds_path, exist_ok=True)

    with open(os.path.join(config.save_path, "config.yaml"), "w") as fp:
        OmegaConf.save(config=config, f=fp)
    fp.close()

    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(config.trainer))
    print(trainer_args)

    print(config.experiment.module.freeze)

    freeze_backbone = config.experiment.module.freeze
    subset = get_subset(config.data.target.subset, config.data.total_species)
    target_size = get_target_size(config, subset)
    print(target_size)
    print("Predicting ", target_size, "species")

    target_type = config.data.target.type
    learning_rate = config.experiment.module.lr
    sigmoid_activation = nn.Sigmoid()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def enable_dropout(m):                         # keep dropout live at test‑time
        if isinstance(m, nn.Dropout): m.train()

    print(config.experiment.module.lr)
    model = ResNet18MeanVariance(output_dim=755, opts=config)
    model = ResNet18MeanVariance.load_from_checkpoint(ckpt_path="HetReg_ZA_3/last.ckpt", output_dim=755, opts=config)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.experiment.module.lr, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     factor=config.scheduler.reduce_lr_plateau.factor,
                                                     patience=config.scheduler.reduce_lr_plateau.lr_schedule_patience
                                                     )

    print(config.scheduler.name)

    opts = config

    
    dm = EbirdDataModule(config)
    dm.prepare_data()
    dm.setup()

    test_loader = dm.test_dataloader()
    num_epochs = config.max_epochs
    print("Everything ok so far")

    opts = config
    T = 20 
    for epoch in range(1):
        # ------------------------------
        # Test phase (optional)
        # ------------------------------
        model.eval()
        model.apply(enable_dropout)

        test_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                for dropout_idx in range(T):
                    print(dropout_idx)
                    x = batch['sat'].to(device).squeeze(1)
                    y = batch['target'].to(device)
                    hotspot_id = batch['hotspot_id']

                    mean, variance, log_variance = model(x)

                    pred = mean.type_as(y)
                    var_arr = variance.type_as(y)

                    if opts.save_preds_path != "":
                        preds_path = opts.save_preds_path
                        path_dropout = f"{preds_path}/{dropout_idx}"
                        if not os.path.isdir(path_dropout):
                            os.makedirs(path_dropout)
                        for i, elem in enumerate(pred):
                            np.save(os.path.join(path_dropout, batch["hotspot_id"][i] + ".npy"),
                                elem.cpu().detach().numpy())
                        for i, elem in enumerate(var_arr):
                            np.save(os.path.join(path_dropout, batch["hotspot_id"][i] + "_var.npy"),
                                elem.cpu().detach().numpy())

        print(f"Epoch {epoch+1} - Test Loss: {test_loss:.4f}")
    
    var = 1
    print(device)

if __name__ == "__main__":
    print("COUCOUUUU")
    main()

""" 
T = 20                                         # MC samples (paper used 50) :contentReference[oaicite:3]{index=3}

epistemic, aleatoric = [], []
with torch.no_grad():
    for _ in range(T):
        mu_t, logvar_t = model(x.cuda())       # stochastic because of dropout
        epistemic.append(mu_t)
        aleatoric.append(torch.exp(logvar_t))  # σ²
    epistemic = torch.stack(epistemic)         # [T, B, N]
    aleatoric = torch.stack(aleatoric)         # [T, B, N]

# ----- predictive statistics -----
mu_pred  = epistemic.mean(0)                           # posterior mean
var_epi  = epistemic.var(0)                            # epistemic part
var_alea = aleatoric.mean(0)                           # aleatoric part
var_pred = var_epi + var_alea                          # total variance
"""