import os
import json
import glob
import torch
import random
import argparse

torch.set_float32_matmul_precision('high')

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from torch import nn
from torchmetrics.classification import BinaryAccuracy
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pytorch_lightning as pl

from core.data import MyDataset
from core.model import CDG

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

# constants
NW = 0
WD = 1e-4
LR = 3e-4
EPOCHS = 200
PATIENT = 3
SAVE_TOP_K = 1
BATCH_SIZE = 1
EARLY_STOP = 7

# S/Z and HSIC
KEY_CONCEPTS = 4
CONFUSE_CONCEPTS = 12
CONCEPT_DIM = 16
HSIC_CHUNK_SIZE = 512
HSIC_CACHE_GB = 8
HSIC_REFERENCE_SEED = 42
LAMBDA_CLS = 1.0
LAMBDA_REC = 1.0
LAMBDA_KL = 1.0
LAMBDA_HSIC = 1e-3
ACCUMULATE_GRAD_BATCHES = 2

TASKS = [
    ['C16_CTRANSPATH', './data/data_c16_ctranspath.csv', 768],
    ['C16_CONCH', './data/data_c16_conch.csv', 512],
    ['C16_UNI', './data/data_c16_uni.csv', 1024]
]

# BACKBONES = ['TransMIL']
BACKBONES = [
    'MaxMIL', 
    'ABMIL', 
    'DSMIL',
    'TransMIL',
    'CLAM-SB', 
    'CLAM-MB', 
    'DTFD-AFS', 
    ]


class ExternalPatchSampler:
    """
    One freshly sampled PFM patch per external TRAINING slide, per batch.
    """

    def __init__(self, train_dataset, cache_gb=8, seed=42):
        self.paths = list(train_dataset.x)
        self.names = [os.path.splitext(os.path.basename(path))[0] for path in self.paths]
        if len(set(self.names)) != len(self.names):
            raise ValueError("Training slide names must be unique for HSIC exclusion.")
        if len(self.paths) < 3:
            raise ValueError("HSIC requires at least three training slides.")
        self.name_set = set(self.names)
        self.cache = {}
        self.cache_bytes = 0
        self.max_cache_bytes = int(cache_gb * 1024 ** 3)
        self.fixed_patches = {}
        self.generator = torch.Generator().manual_seed(seed)

        self.fixed_seed = seed
        estimated_bytes = sum(os.path.getsize(path) for path in self.paths)
        print("HSIC: {} training slides, feature files {:.2f} GB, CPU cache limit {:.2f} GB.".format(len(self.paths), estimated_bytes / 1024 ** 3, cache_gb))
        if estimated_bytes > self.max_cache_bytes:
            print("HSIC: uncached slides will be read from disk each batch; all patches remain eligible.")

    def load_features(self, path):
        if path in self.cache:
            return self.cache[path]
        x = torch.load(path, map_location='cpu')
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Expected nonempty [patches, features] tensor: {}".format(path))
        size = x.numel() * x.element_size()
        # Keep admitted files instead of LRU scan-thrashing across every slide.
        if self.cache_bytes + size <= self.max_cache_bytes:
            self.cache[path] = x
            self.cache_bytes += size
        return x

    def sample(self, current_name, deterministic=False):
        if not deterministic and current_name not in self.name_set:
            raise ValueError("Current training slide is absent from the HSIC training sampler.")
        patches = []
        for index, (name, path) in enumerate(zip(self.names, self.paths)):
            if name == current_name:
                continue
            if deterministic and path in self.fixed_patches:
                patch = self.fixed_patches[path]
            else:
                x = self.load_features(path)
                generator = torch.Generator().manual_seed(self.fixed_seed + index) if deterministic else self.generator
                patch_index = torch.randint(x.shape[0], (1,), generator=generator).item()
                # Clone one row so fixed references do not retain uncached slides.
                patch = x[patch_index].clone()
                if deterministic:
                    self.fixed_patches[path] = patch
            patches.append(patch)
        return torch.stack(patches)

    def clear(self):
        self.cache.clear()
        self.fixed_patches.clear()
        self.cache_bytes = 0


# pytorch lightning module

class Net(pl.LightningModule):
    def __init__(self, backbone, dim, reference_sampler=None):
        super().__init__()

        self.reference_sampler = reference_sampler
        self.save_hyperparameters({'backbone': backbone, 'dim': dim, 'lambda_cls': LAMBDA_CLS, 'lambda_rec': LAMBDA_REC, 'lambda_kl': LAMBDA_KL, 'lambda_hsic': LAMBDA_HSIC, 'hsic_chunk_size': HSIC_CHUNK_SIZE, 'hsic_cache_gb': HSIC_CACHE_GB, 'hsic_reference_seed': HSIC_REFERENCE_SEED, 'hsic_kernel': 'normalized_biased_rbf_reference_median', 'accumulate_grad_batches': ACCUMULATE_GRAD_BATCHES})
        self.model = CDG(backbone=backbone, input_dim=dim, key_concepts=KEY_CONCEPTS, confuse_concepts=CONFUSE_CONCEPTS, concept_dim=CONCEPT_DIM, hsic_chunk_size=HSIC_CHUNK_SIZE, lambda_cls=LAMBDA_CLS, lambda_rec=LAMBDA_REC, lambda_kl=LAMBDA_KL, lambda_hsic=LAMBDA_HSIC)

        self.accuracy = BinaryAccuracy()

    def forward(self, x):
        op = self.model(x)
        return op

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=LR, weight_decay=WD)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=PATIENT, verbose=True, mode='max')
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'valid_acc'
        }

    def reference_features(self, x, names, deterministic=False):
        if len(names) != 1:
            raise ValueError("HSIC training requires BATCH_SIZE=1.")
        if self.reference_sampler is None:
            raise RuntimeError("Attach a training-only ExternalPatchSampler before HSIC training/validation.")
        return self.reference_sampler.sample(names[0], deterministic=deterministic).to(device=x.device, dtype=x.dtype)

    def training_step(self, train_batch, batch_idx):
        x, y, mark, names = train_batch

        reference_x = self.reference_features(x, names) if self.model.lambda_hsic != 0 else None
        outputs = self.model.compute_losses(x, y, mark, reference_x=reference_x)

        self.log('train_loss', outputs['loss'], on_epoch=False, prog_bar=True, logger=True, batch_size=len(y))
        self.log('train_loss_cls', outputs['loss_cls'], on_epoch=False, logger=True, batch_size=len(y))
        self.log('train_loss_rec', outputs['loss_rec'], on_epoch=False, logger=True, batch_size=len(y))
        self.log('train_loss_kl', outputs['loss_kl'], on_epoch=False, logger=True, batch_size=len(y))
        self.log('train_loss_hsic', outputs['loss_hsic'], on_epoch=False, logger=True, batch_size=len(y))
        self.log('train_hsic_weighted', self.model.lambda_hsic * outputs['loss_hsic'], on_epoch=False, logger=True, batch_size=len(y))
        self.log('train_s_std', outputs['s'].detach().flatten(start_dim=1).std(dim=0, unbiased=False).mean(), on_epoch=False, logger=True, batch_size=len(y))
        self.log('train_z_std', outputs['z'].detach().flatten(start_dim=1).std(dim=0, unbiased=False).mean(), on_epoch=False, logger=True, batch_size=len(y))
        self.log('train_acc', self.accuracy(outputs['probs'], y), on_epoch=True, prog_bar=True, logger=True, batch_size=len(y))
        return outputs['loss']

    def validation_step(self, val_batch, batch_idx):
        x, y, mark, names = val_batch
        reference_x = self.reference_features(x, names, deterministic=True) if self.model.lambda_hsic != 0 else None
        outputs = self.model.compute_losses(x, y, mark, reference_x=reference_x)
        pd = outputs['probs']
        self.log('valid_loss', outputs['loss'], on_epoch=True, logger=True, batch_size=len(y))
        self.log('valid_loss_hsic', outputs['loss_hsic'], on_epoch=True, logger=True, batch_size=len(y))
        self.log('valid_acc', self.accuracy(pd, y), on_epoch=True, prog_bar=True, logger=True, batch_size=len(y))

    def predict_step(self, batch, batch_idx):
        x, y, mark, _ = batch
        outputs = self.model.compute_losses(x, y, mark)
        pd = outputs['probs']
        return pd, y


# main

def custom_collate_func(batch):
    x = torch.concat([item[0] for item in batch], axis=0)
    y = torch.tensor([item[1] for item in batch])
    mark = [item[2] for item in batch]
    name = [item[3] for item in batch]
    return x, y, mark, name


def train_one_times(backbone, dim, csv_path, task_name):
    train_dataset = MyDataset(csv_path, data_type='train')
    valid_dataset = MyDataset(csv_path, data_type='valid')
    test_dataset = MyDataset(csv_path, data_type='test')

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NW,
        drop_last=False,
        collate_fn=custom_collate_func
        )
    
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NW, 
        drop_last=False,
        collate_fn=custom_collate_func
        )
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NW, 
        drop_last=False,
        collate_fn=custom_collate_func
        )

    reference_sampler = ExternalPatchSampler(train_dataset, cache_gb=HSIC_CACHE_GB, seed=HSIC_REFERENCE_SEED) if LAMBDA_HSIC != 0 else None
    if reference_sampler is not None:
        for dataset in (valid_dataset, test_dataset):
            names = {os.path.splitext(os.path.basename(path))[0] for path in dataset.x}
            if reference_sampler.name_set & names:
                raise ValueError("Train and validation/test slide identities overlap.")
    model = Net(backbone, dim, reference_sampler=reference_sampler)

    save_dir = './logs{}/{}'.format(BATCH_SIZE, task_name)

    logger = TensorBoardLogger(name=backbone, save_dir=save_dir)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=SAVE_TOP_K, 
        monitor='valid_acc', 
        mode='max', 
        save_last=True, 
        filename='{epoch}-{valid_acc:.4f}'
        )
    
    earlystop_callback = EarlyStopping(
        monitor='valid_acc', 
        mode='max', 
        min_delta=0.00, 
        patience=EARLY_STOP
        )

    # training: Lightning handles backward, optimizer steps and gradient accumulation.
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1, 
        max_epochs=EPOCHS, 
        logger=logger, 
        check_val_every_n_epoch=1, 
        callbacks=[checkpoint_callback, earlystop_callback], 
        deterministic=True, 
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES
        )
    
    trainer.fit(model, train_loader, valid_loader)

    predictions = trainer.predict(dataloaders=test_loader, ckpt_path='best')

    preds = torch.concat([item[0] for item in predictions]).float().numpy().tolist()
    labels = torch.concat([item[1] for item in predictions]).float().numpy().tolist()

    results = {
        'img': test_loader.dataset.x,
        'pred': preds,
        'label': labels,
    }
    results_json = json.dumps(results)
    with open(os.path.join(trainer.log_dir, 'results.json'), 'w+') as f:
        f.write(results_json)

    if reference_sampler is not None:
        reference_sampler.clear()


def main():
    for task in TASKS:
        task_name, csv_path, dim = task
        for backbone in BACKBONES:
            print(backbone, task_name, csv_path, dim)
            train_one_times(backbone, dim=dim, csv_path=csv_path, task_name=task_name)

if __name__ == '__main__':
    main()
