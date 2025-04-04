import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
import sys

# Complete suppression of all warnings during import and model creation
original_showwarning = warnings.showwarning

def silent_showwarning(*args, **kwargs):
    pass

# Completely disable all warnings
warnings.showwarning = silent_showwarning

# Suppress warnings
warnings.filterwarnings('ignore')
# Specifically suppress pl_bolts UnderReviewWarning
warnings.filterwarnings('ignore', category=UserWarning, module='pl_bolts')
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PL_DISABLE_WARNINGS"] = "1"

# Import everything silently
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from utils.utils import grab_image_augmentations, grab_wids, grab_arg_from_checkpoint, prepend_paths, re_prepend_paths
from utils.ssl_online_custom import SSLOnlineEvaluator

from datasets.ContrastiveTabularDataset import ContrastiveTabularDataset
from datasets.PPMIDataset import PPMIDataset
from datasets.ImagingAndTabularDataset import ImagingAndTabularDataset
from datasets.ImageDataset import ImageDataset

# Silence all output related to models
import contextlib
import io
null_output = io.StringIO()

with contextlib.redirect_stdout(null_output), contextlib.redirect_stderr(null_output):
    from models.MultimodalSimCLR import MultimodalSimCLR
    from models.SimCLR import SimCLR
    from models.BYOL_Bolt import BYOL
    from models.BarlowTwins import BarlowTwins
    from models.SCARF import SCARF



def load_datasets(hparams):
  if hparams.datatype == 'multimodal':
    transform = grab_image_augmentations(hparams.img_size, hparams.target)
    hparams.transform = transform.__repr__()
    train_dataset = ContrastiveImagingAndTabularDataset(
      hparams.data_train_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
      hparams.data_train_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot,
      hparams.labels_train, hparams.img_size, hparams.live_loading)
    val_dataset = ContrastiveImagingAndTabularDataset(
      hparams.data_val_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
      hparams.data_val_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot,
      hparams.labels_val, hparams.img_size, hparams.live_loading)
    hparams.input_size = train_dataset.get_input_size()
  elif hparams.datatype == 'imaging':
    transform = grab_image_augmentations(hparams.img_size, hparams.target, hparams.crop_scale_lower)
    hparams.transform = transform.__repr__()
    train_dataset = ContrastiveImageDataset(
      data=hparams.data_train_imaging, labels=hparams.labels_train, 
      transform=transform, delete_segmentation=hparams.delete_segmentation, 
      augmentation_rate=hparams.augmentation_rate, img_size=hparams.img_size, live_loading=hparams.live_loading)
    val_dataset = ContrastiveImageDataset(
      data=hparams.data_val_imaging, labels=hparams.labels_val, 
      transform=transform, delete_segmentation=hparams.delete_segmentation, 
      augmentation_rate=hparams.augmentation_rate, img_size=hparams.img_size, live_loading=hparams.live_loading)
  elif hparams.datatype == 'tabular':
    train_dataset = ContrastiveTabularDataset(hparams.data_train_tabular, hparams.labels_train, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
    val_dataset = ContrastiveTabularDataset(hparams.data_val_tabular, hparams.labels_val, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
    hparams.input_size = train_dataset.get_input_size()
  elif hparams.datatype == 'ppmi':
    train_dataset = PPMIDataset(
      processed_data_dir=os.path.join(hparams.data_base, 'processed-data'),
      img_size=hparams.input_size,
      live_loading=True,
      train=True,
      augmentation=hparams.transform_train,
      augmentation_rate=hparams.augmentation_rate,
      one_hot_tabular=hparams.one_hot,
      corruption_rate=hparams.corruption_rate
    )
    val_dataset = PPMIDataset(
      processed_data_dir=os.path.join(hparams.data_base, 'processed-data'),
      img_size=hparams.input_size,
      live_loading=True,
      train=False,
      augmentation=hparams.transform_val,
      augmentation_rate=hparams.augmentation_rate,
      one_hot_tabular=hparams.one_hot,
      corruption_rate=hparams.corruption_rate
    )
    hparams.input_size = train_dataset.get_input_size()
  else:
    raise Exception(f'Unknown datatype {hparams.datatype}')
  return train_dataset, val_dataset


def select_model(hparams, train_dataset):
  if hparams.datatype == 'multimodal' or hparams.datatype == 'ppmi':
    model = MultimodalSimCLR(hparams)
  elif hparams.datatype == 'imaging':
    if hparams.loss.lower() == 'byol':
      model = BYOL(**hparams)
    elif hparams.loss.lower() == 'simsiam':
      model = SimSiam(**hparams)
    elif hparams.loss.lower() == 'swav':
      if not hparams.resume_training:
        model = SwAV(gpus=1, nmb_crops=(2,0), num_samples=len(train_dataset),  **hparams)
      else:
        model = SwAV(**hparams)
    elif hparams.loss.lower() == 'barlowtwins':
      model = BarlowTwins(**hparams)
    else:
      model = SimCLR(hparams)
  elif hparams.datatype == 'tabular':
    model = SCARF(hparams)
  else:
    raise Exception(f'Unknown datatype {hparams.datatype}')
  return model


def pretrain(hparams, logger):
  """
  Train code for pretraining or supervised models. 
  
  IN
  hparams:      All hyperparameters
  logger:       PyTorch Lightning logger (CSVLogger)
  """
  pl.seed_everything(hparams.seed)

  # Load appropriate dataset
  train_dataset, val_dataset = load_datasets(hparams)
  
  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=True, persistent_workers=(hparams.num_workers > 0))

  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, persistent_workers=(hparams.num_workers > 0))

  # Create log directory
  logdir = os.path.join(hparams.data_base, 'runs', hparams.datatype, 'checkpoints')
  os.makedirs(logdir, exist_ok=True)
  
  model = select_model(hparams, train_dataset)
  
  callbacks = []

  if hparams.online_mlp:
    model.hparams.classifier_freq = float('Inf')
    callbacks.append(SSLOnlineEvaluator(z_dim = model.pooled_dim, hidden_dim = hparams.embedding_dim, num_classes = hparams.num_classes, swav = False, multimodal = (hparams.datatype=='multimodal')))
  callbacks.append(ModelCheckpoint(filename='checkpoint_last_epoch_{epoch:02d}', dirpath=logdir, save_on_train_epoch_end=True, auto_insert_metric_name=False))
  callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  trainer = Trainer.from_argparse_args(
    hparams, 
    accelerator="gpu",
    devices=1,
    num_sanity_val_steps=0,  # Disable sanity checking
    callbacks=callbacks, 
    logger=logger,
    max_epochs=hparams.max_epochs,
    check_val_every_n_epoch=hparams.check_val_every_n_epoch,
    limit_train_batches=hparams.limit_train_batches,
    limit_val_batches=hparams.limit_val_batches,
    enable_progress_bar=hparams.enable_progress_bar
  )

  if hparams.resume_training:
    trainer.fit(model, train_loader, val_loader, ckpt_path=hparams.checkpoint)
  else:
    trainer.fit(model, train_loader, val_loader)