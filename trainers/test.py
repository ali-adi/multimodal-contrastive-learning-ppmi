from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from datasets.ImageDataset import ImageDataset
from datasets.TabularDataset import TabularDataset
from models.Evaluator import Evaluator
from utils.utils import grab_arg_from_checkpoint
import os


def test(hparams, logger=None):
    """
    Tests trained models. 
    
    IN
    hparams: All hyperparameters
    logger: PyTorch Lightning logger (CSVLogger)
    """
    pl.seed_everything(hparams.seed)
    
    if hparams.datatype == 'imaging' or hparams.datatype == 'multimodal':
        test_dataset = ImageDataset(hparams.data_test_eval_imaging, hparams.labels_test_eval_imaging, hparams.delete_segmentation, 0, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading)
        
        print(test_dataset.transform_val.__repr__())
    elif hparams.datatype == 'tabular':
        test_dataset = TabularDataset(hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular, hparams.eval_one_hot, hparams.field_lengths_tabular)
        hparams.input_size = test_dataset.get_input_size()
    elif hparams.datatype == 'ppmi':
        from datasets.PPMIDataset import PPMIDataset
        test_dataset = PPMIDataset(
            processed_data_dir=os.path.join(hparams.data_base, 'processed-data'),
            img_size=hparams.input_size,
            live_loading=True,
            train=False,
            augmentation=hparams.transform_test,
            augmentation_rate=hparams.augmentation_rate,
            one_hot_tabular=hparams.one_hot,
            corruption_rate=hparams.corruption_rate
        )
    else:
        raise Exception('argument datatype must be set to imaging, tabular, multimodal, or ppmi')
    
    drop = ((len(test_dataset)%hparams.batch_size)==1)

    test_loader = DataLoader(
        test_dataset,
        num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
        pin_memory=True, shuffle=False, drop_last=drop, persistent_workers=(hparams.num_workers > 0))

    hparams.dataset_length = len(test_loader)

    model = Evaluator(hparams)
    model.freeze()
    trainer = Trainer.from_argparse_args(
        hparams,
        accelerator="gpu",
        devices=1,
        logger=logger
    )
    trainer.test(model, test_loader, ckpt_path=hparams.checkpoint)