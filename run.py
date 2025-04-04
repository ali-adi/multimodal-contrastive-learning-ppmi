import os 
import sys
import time
import random
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import warnings
import logging

# Suppress all warnings
warnings.filterwarnings('ignore')
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PL_DISABLE_WARNINGS"] = "1"

from trainers.pretrain import pretrain
from trainers.evaluate import evaluate
from trainers.test import test
from trainers.generate_embeddings import generate_embeddings
from utils.utils import grab_arg_from_checkpoint, prepend_paths, re_prepend_paths

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Multimodal Contrastive Learning')
    
    # Model Configuration
    parser.add_argument('--model', type=str, default='resnet50', help='Model architecture')
    parser.add_argument('--datatype', type=str, default='ppmi', help='Dataset type')
    parser.add_argument('--input_size', type=int, default=224, help='Input size for images')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the embedding space')
    parser.add_argument('--projection_dim', type=int, default=128, help='Dimension of the projection head output')
    parser.add_argument('--imaging_pretrain_checkpoint', type=str, default=None, help='Path to pretrained imaging model checkpoint')
    parser.add_argument('--encoder_num_layers', type=int, default=3, help='Number of layers in the tabular encoder')
    parser.add_argument('--pretrained_imaging_strategy', type=str, default='frozen', choices=['frozen', 'unfrozen'], help='Strategy for pretrained imaging model')
    parser.add_argument('--init_strat', type=str, default='normal', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='Weight initialization strategy')
    parser.add_argument('--finetune_strategy', type=str, default='unfrozen', choices=['frozen', 'unfrozen'], help='Strategy for finetuning pretrained models')
    parser.add_argument('--online_mlp', action='store_true', help='Whether to use online MLP')
    parser.add_argument('--pooled_dim', type=int, default=2048, help='Dimension of the pooled embedding')
    
    # Training Parameters
    parser.add_argument('--pretrain', action='store_true', help='Whether to pretrain')
    parser.add_argument('--run_eval', action='store_true', help='Whether to run evaluation')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of epochs')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='Number of epochs between validation runs')
    parser.add_argument('--limit_train_batches', type=float, default=1.0, help='Limit the number of training batches per epoch (float between 0 and 1)')
    parser.add_argument('--limit_val_batches', type=float, default=1.0, help='Limit the number of validation batches per epoch (float between 0 and 1)')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'anneal'], help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--cosine_anneal_mult', type=int, default=1, help='Multiplier for cosine annealing')
    parser.add_argument('--dataset_length', type=int, default=1000, help='Length of the dataset')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss')
    parser.add_argument('--lambda_0', type=float, default=0.1, help='Lambda parameter for contrastive loss')
    parser.add_argument('--loss', type=str, default='clip', choices=['clip', 'ntxent', 'supcon', 'binary_supcon', 'remove_fn', 'binary_remove_fn', 'kpositive'], help='Contrastive loss function')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for kpositive loss')
    parser.add_argument('--train_similarity_matrix', type=str, default=None, help='Path to training similarity matrix')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for classification')
    parser.add_argument('--classifier_freq', type=int, default=1, help='Frequency of classifier evaluation')
    parser.add_argument('--log_images', action='store_true', help='Whether to log images')
    parser.add_argument('--enable_progress_bar', action='store_true', default=True, help='Enable progress bar for training')
    parser.add_argument('--gradient_clip_val', type=float, default=None, help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Number of batches to accumulate gradients')
    parser.add_argument('--precision', type=str, default='32', help='Precision for training')
    parser.add_argument('--detect_anomaly', action='store_true', help='Enable anomaly detection for debugging')
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='How often to log within steps')
    parser.add_argument('--accelerator', type=str, default='gpu', help='Accelerator to use')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices to use')
    parser.add_argument('--auto_lr_find', action='store_true', help='Auto find learning rate')
    parser.add_argument('--auto_scale_batch_size', action='store_true', help='Auto scale batch size')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic training')
    parser.add_argument('--fast_dev_run', action='store_true', help='Run a lightweight test run for debugging')
    
    # Data Parameters
    parser.add_argument('--data_base', type=str, default=os.path.dirname(os.path.abspath(__file__)), help='Base directory for data')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--live_loading', action='store_true', help='Whether to load images live')
    parser.add_argument('--delete_segmentation', action='store_true', help='Whether to delete segmentation')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--crop_scale_lower', type=float, default=0.08, help='Lower bound for crop scale')
    parser.add_argument('--target', type=str, default='binary', help='Target type')
    parser.add_argument('--task', type=str, default='classification', help='Task type')
    parser.add_argument('--eval_train_augment_rate', type=float, default=0.5, help='Augmentation rate for evaluation training')
    parser.add_argument('--eval_one_hot', action='store_true', help='Whether to use one-hot encoding for evaluation')
    parser.add_argument('--weighted_sampler', action='store_true', help='Whether to use weighted sampler')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights file')
    
    # Data Paths
    parser.add_argument('--data_train_tabular', type=str, default='processed-data/data_tabular.pt', help='Path to training tabular data')
    parser.add_argument('--data_train_imaging', type=str, default='processed-data/data_imaging.pt', help='Path to training imaging data')
    parser.add_argument('--data_val_tabular', type=str, default='processed-data/data_tabular_val.pt', help='Path to validation tabular data')
    parser.add_argument('--data_val_imaging', type=str, default='processed-data/data_imaging_val.pt', help='Path to validation imaging data')
    parser.add_argument('--data_test_tabular', type=str, default='processed-data/data_tabular_test.pt', help='Path to test tabular data')
    parser.add_argument('--data_test_imaging', type=str, default='processed-data/data_imaging_test.pt', help='Path to test imaging data')
    parser.add_argument('--labels_train', type=str, default='processed-data/labels_train.pt', help='Path to training labels')
    parser.add_argument('--labels_val', type=str, default='processed-data/labels_val.pt', help='Path to validation labels')
    parser.add_argument('--labels_test', type=str, default='processed-data/labels_test.pt', help='Path to test labels')
    parser.add_argument('--field_lengths_tabular', type=str, default='processed-data/field_lengths_tabular.pt', help='Path to tabular field lengths')
    parser.add_argument('--data_train_eval_imaging', type=str, default='processed-data/data_imaging.pt', help='Path to training evaluation imaging data')
    parser.add_argument('--data_train_eval_tabular', type=str, default='processed-data/data_tabular.pt', help='Path to training evaluation tabular data')
    parser.add_argument('--data_val_eval_imaging', type=str, default='processed-data/data_imaging_val.pt', help='Path to validation evaluation imaging data')
    parser.add_argument('--data_val_eval_tabular', type=str, default='processed-data/data_tabular_val.pt', help='Path to validation evaluation tabular data')
    parser.add_argument('--data_test_eval_imaging', type=str, default='processed-data/data_imaging_test.pt', help='Path to test evaluation imaging data')
    parser.add_argument('--data_test_eval_tabular', type=str, default='processed-data/data_tabular_test.pt', help='Path to test evaluation tabular data')
    parser.add_argument('--labels_train_eval_imaging', type=str, default='processed-data/labels_train.pt', help='Path to training evaluation imaging labels')
    parser.add_argument('--labels_train_eval_tabular', type=str, default='processed-data/labels_train.pt', help='Path to training evaluation tabular labels')
    parser.add_argument('--labels_val_eval_imaging', type=str, default='processed-data/labels_val.pt', help='Path to validation evaluation imaging labels')
    parser.add_argument('--labels_val_eval_tabular', type=str, default='processed-data/labels_val.pt', help='Path to validation evaluation tabular labels')
    parser.add_argument('--labels_test_eval_imaging', type=str, default='processed-data/labels_test.pt', help='Path to test evaluation imaging labels')
    parser.add_argument('--labels_test_eval_tabular', type=str, default='processed-data/labels_test.pt', help='Path to test evaluation tabular labels')
    
    # Data Augmentation Parameters
    parser.add_argument('--transform_train', nargs='+', default=['random_crop', 'random_horizontal_flip', 'random_vertical_flip', 'random_rotation', 'random_affine', 'normalize'], help='Training transformations')
    parser.add_argument('--transform_val', nargs='+', default=['center_crop', 'normalize'], help='Validation transformations')
    parser.add_argument('--transform_test', nargs='+', default=['center_crop', 'normalize'], help='Test transformations')
    parser.add_argument('--augmentation_rate', type=float, default=0.95, help='Augmentation rate')
    parser.add_argument('--one_hot', action='store_true', help='Whether to use one-hot encoding')
    parser.add_argument('--corruption_rate', type=float, default=0.3, help='Corruption rate for data augmentation')
    
    # Model Strategy
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    
    # Evaluation Parameters
    parser.add_argument('--test', action='store_true', help='Whether to run testing')
    parser.add_argument('--evaluate', action='store_true', help='Whether to run evaluation')
    parser.add_argument('--generate_embeddings', action='store_true', help='Whether to generate embeddings')
    
    return parser.parse_args()

def run(args):
    pl.seed_everything(args.seed)
    args = prepend_paths(args)
    time.sleep(random.randint(1,5))  # Prevents multiple runs getting the same version when launching many jobs at once

    if args.resume_training:
        checkpoint = args.checkpoint
        ckpt = torch.load(args.checkpoint)
        args = ckpt['hyper_parameters']
        args.checkpoint = checkpoint
        args.resume_training = True
        args = re_prepend_paths(args)
    
    if args.generate_embeddings:
        if not args.datatype:
            args.datatype = grab_arg_from_checkpoint(args, 'datatype')
        generate_embeddings(args)
        return args
    
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    logger = CSVLogger(base_dir, name='logs')
    
    if args.checkpoint and not args.resume_training:
        if not args.datatype:
            args.datatype = grab_arg_from_checkpoint(args, 'datatype')
            
    if args.pretrain:
        pretrain(args, logger)
        args.checkpoint = os.path.join(base_dir, 'runs', args.datatype, 'checkpoints', f'checkpoint_last_epoch_{args.max_epochs-1:02}.ckpt')
    
    if args.test:
        test(args, logger)
    elif args.evaluate:
        evaluate(args, logger)

if __name__ == "__main__":
    args = parse_args()
    run(args)

