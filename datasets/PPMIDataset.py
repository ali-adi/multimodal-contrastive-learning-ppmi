from typing import List, Tuple, Dict, Any
import random
import os
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import nibabel as nib
import torchio as tio
from torchvision.transforms import transforms

class PPMIDataset(Dataset):
    """
    Dataset class for PPMI data that loads processed tabular and imaging data.
    
    This dataset loads the processed data from the ppmi_prepare_dataset.py script
    and provides it in a format suitable for training.
    """
    def __init__(
        self,
        processed_data_dir: str,
        img_size: int = 224,
        live_loading: bool = True,
        train: bool = True,
        augmentation: transforms.Compose = None,
        augmentation_rate: float = 0.5,
        one_hot_tabular: bool = False,
        corruption_rate: float = 0.2
    ) -> None:
        """
        Initialize the PPMI dataset.
        
        Args:
            processed_data_dir: Directory containing the processed data files
            img_size: Size to resize images to
            live_loading: Whether to load images from disk or use pre-loaded tensors
            train: Whether this is for training or evaluation
            augmentation: Transforms to apply for augmentation
            augmentation_rate: Probability of applying augmentation to the second view
            one_hot_tabular: Whether to one-hot encode tabular data
            corruption_rate: Rate at which to corrupt tabular features for contrastive learning
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.img_size = img_size
        self.live_loading = live_loading
        self.train = train
        self.one_hot_tabular = one_hot_tabular
        self.corruption_rate = corruption_rate
        
        # Load data
        self.data_tabular = torch.load(self.processed_data_dir / 'data_tabular.pt')
        self.field_lengths_tabular = torch.load(self.processed_data_dir / 'field_lengths_tabular.pt')
        self.patient_ids = torch.load(self.processed_data_dir / 'patient_ids.pt', weights_only=False)
        self.image_paths = torch.load(self.processed_data_dir / 'data_imaging.pt', weights_only=False)
        
        print("Loaded data:")
        print(f"Number of patients in tabular data: {len(self.patient_ids)}")
        print(f"Number of image paths: {len(self.image_paths)}")
        
        # Create a mapping from patient ID to index
        self.patient_to_idx = {str(pid): i for i, pid in enumerate(self.patient_ids)}
        print(f"Patient IDs in tabular data: {list(self.patient_to_idx.keys())[:5]}...")
        
        # Create a list of valid indices (patients that have both tabular and imaging data)
        self.valid_indices = []
        for i, img_path in enumerate(self.image_paths[::2]):  # Step by 2 since each patient has 2 images
            try:
                # Extract patient ID from path
                path_parts = img_path.split('/')
                ppmi_index = path_parts.index('PPMI')
                patient_id = path_parts[ppmi_index + 2]  # Get the ID after the second 'PPMI' directory
                print(f"Processing image path: {img_path}")
                print(f"Extracted patient ID: {patient_id}")
                
                # Check if patient exists in mapping
                if patient_id in self.patient_to_idx:
                    self.valid_indices.append(self.patient_to_idx[patient_id])
            except Exception as e:
                print(f"Error processing path {img_path}: {str(e)}")
        
        print(f"Found {len(self.valid_indices)} patients with both tabular and imaging data")
        
        if len(self.valid_indices) == 0:
            print("WARNING: No valid patients found!")
            print("First few image paths:")
            for path in self.image_paths[:5]:
                print(f"  {path}")
            print("\nFirst few patient IDs:")
            for pid in list(self.patient_to_idx.keys())[:5]:
                print(f"  {pid}")
        
        # Set up transforms for NIfTI images
        self.transform = tio.Compose([
            tio.RescaleIntensity((0, 1)),  # Normalize intensity to [0, 1]
            tio.Resize((img_size, img_size, img_size))  # Resize to desired dimensions
        ])
        
        # Set up augmentation transforms
        if augmentation is None:
            self.augmentation = tio.Compose([
                tio.RandomAffine(
                    scales=(0.9, 1.1),
                    degrees=10,
                    translation=5,
                    p=0.75
                ),
                tio.RandomFlip(axes=(0,), p=0.5),
                tio.RandomNoise(p=0.25),
                tio.RandomBiasField(p=0.25),
                tio.RescaleIntensity((0, 1))
            ])
        else:
            self.augmentation = augmentation
            
        self.augmentation_rate = augmentation_rate
        
        # Generate marginal distributions for tabular data corruption
        if len(self.valid_indices) > 0:
            self.generate_marginal_distributions()
        
    def generate_marginal_distributions(self) -> None:
        """
        Generates empirical marginal distribution for tabular data corruption
        """
        # Convert tensor to list of lists for easier manipulation
        data_list = self.data_tabular[self.valid_indices].numpy().tolist()
        # Transpose to get features as rows
        self.marginal_distributions = list(map(list, zip(*data_list)))
        
    def corrupt(self, subject: torch.Tensor) -> torch.Tensor:
        """
        Creates a corrupted version of a subject's tabular data
        
        Args:
            subject: Original tabular data tensor
            
        Returns:
            Corrupted tabular data tensor
        """
        subject = subject.clone()
        
        # Determine which features to corrupt
        num_features = subject.shape[0]
        num_to_corrupt = int(num_features * self.corruption_rate)
        indices = random.sample(list(range(num_features)), num_to_corrupt)
        
        # Corrupt selected features
        for i in indices:
            # Sample a new value from the marginal distribution
            new_value = random.choice(self.marginal_distributions[i])
            subject[i] = new_value
            
        return subject
        
    def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
        """
        One-hot encodes a subject's tabular features
        
        Args:
            subject: Original tabular data tensor
            
        Returns:
            One-hot encoded tensor
        """
        out = []
        for i in range(len(subject)):
            if self.field_lengths_tabular[i] == 1:
                out.append(subject[i].unsqueeze(0))
            else:
                out.append(torch.nn.functional.one_hot(
                    torch.clamp(subject[i], min=0, max=self.field_lengths_tabular[i]-1).long(), 
                    num_classes=int(self.field_lengths_tabular[i])
                ))
        return torch.cat(out)
        
    def get_input_size(self) -> int:
        """
        Returns the input size for the tabular encoder
        
        Returns:
            Number of input features (after one-hot encoding if applicable)
        """
        if self.one_hot_tabular:
            return int(sum(self.field_lengths_tabular))
        else:
            return self.data_tabular.shape[1]
            
    def load_nifti_image(self, img_path: str) -> torch.Tensor:
        """
        Load a NIfTI image and preprocess it
        
        Args:
            img_path: Path to the NIfTI file
            
        Returns:
            Preprocessed image tensor
        """
        # Load NIfTI file
        img = nib.load(img_path).get_fdata()
        
        # Convert to tensor and add channel dimension
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
        # Apply basic preprocessing
        img = self.transform(img)
        
        return img
            
    def generate_imaging_views(self, index: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Generates two views of a subject's image
        
        Args:
            index: Index of the subject
            
        Returns:
            Tuple of (list of two image views, original image)
        """
        patient_id = self.patient_ids[self.valid_indices[index]]
        patient_images = [img for img in self.image_paths if patient_id in img]
        
        # Group images by modality
        t1_images = [img for img in patient_images if 'T1-anatomical' in img]
        fa_images = [img for img in patient_images if 'FA_map-MRI' in img]
        
        # Extract dates from filenames and sort by date
        def extract_date(img_path):
            # Extract date from path (format: YYYY-MM-DD_HH_MM_SS.0)
            date_str = img_path.split('/')[-3]
            return datetime.strptime(date_str, '%Y-%m-%d_%H_%M_%S.0')
        
        # Sort images by date and get the most recent one for each modality
        t1_images.sort(key=extract_date, reverse=True)
        fa_images.sort(key=extract_date, reverse=True)
        
        if not t1_images or not fa_images:
            raise ValueError(f"Missing required modality for patient {patient_id}. T1: {len(t1_images)}, FA: {len(fa_images)}")
        
        # Use the most recent scan for each modality
        t1_path = t1_images[0]
        fa_path = fa_images[0]
        
        # Load and preprocess images
        t1_img = self.load_nifti_image(t1_path)
        fa_img = self.load_nifti_image(fa_path)
        
        # Generate views
        t1_views = self.augmentation(tio.Subject(image=tio.ScalarImage(tensor=t1_img)))
        fa_views = self.augmentation(tio.Subject(image=tio.ScalarImage(tensor=fa_img)))
        
        # Combine views
        imaging_views = torch.cat([t1_views.image.data, fa_views.image.data], dim=0)
        
        # Return unaugmented image for visualization
        unaugmented_image = torch.cat([t1_img, fa_img], dim=0)
        
        return imaging_views, unaugmented_image
        
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a data sample
        
        Args:
            index: Index of the sample
            
        Returns:
            Dictionary containing the sample data
        """
        # Get valid index
        valid_idx = self.valid_indices[index]
        
        # Generate imaging views
        imaging_views, unaugmented_image = self.generate_imaging_views(index)
        
        # Generate tabular views
        tabular_data = self.data_tabular[valid_idx]
        tabular_views = [
            tabular_data,
            self.corrupt(tabular_data)
        ]
        
        # One-hot encode if needed
        if self.one_hot_tabular:
            tabular_views = [self.one_hot_encode(tv) for tv in tabular_views]
            
        # Get patient ID
        patient_id = self.patient_ids[valid_idx]
        
        return {
            'imaging_views': imaging_views,
            'tabular_views': tabular_views,
            'unaugmented_image': unaugmented_image,
            'patient_id': patient_id
        }
        
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset
        
        Returns:
            Number of samples
        """
        return len(self.valid_indices)
        
    def visualize_sample(self, idx: int, save_dir: str = "visualization") -> None:
        """
        Visualize a sample from the dataset
        
        Args:
            idx: Index of the sample to visualize
            save_dir: Directory to save visualizations
        """
        # Create save directory if it doesn't exist
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get sample
        sample = self[idx]
        
        # Get middle slices from each view
        def get_middle_slice(img):
            print(f"Input tensor shape: {img.shape}")
            # Handle 3D, 4D and 5D tensors
            if len(img.shape) == 5:  # [batch, channels, depth, height, width]
                return img[0, 0, img.shape[2]//2, :, :]
            elif len(img.shape) == 4:  # [channels, depth, height, width]
                return img[0, img.shape[1]//2, :, :]
            else:  # [depth, height, width]
                return img[img.shape[0]//2, :, :]
            
        # Get slices for both modalities
        print(f"Unaugmented image shape: {sample['unaugmented_image'].shape}")
        t1_orig = get_middle_slice(sample['unaugmented_image'][0:1])
        dti_orig = get_middle_slice(sample['unaugmented_image'][1:2])
        
        print(f"Imaging views shape: {[v.shape for v in sample['imaging_views']]}")
        t1_view1 = get_middle_slice(sample['imaging_views'][0])
        t1_view2 = get_middle_slice(sample['imaging_views'][1])
        
        # Create figure
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # Plot T1 slices
        axes[0,0].imshow(t1_orig, cmap='gray')
        axes[0,0].set_title('T1 Original')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(dti_orig, cmap='gray')
        axes[0,1].set_title('DTI Original')
        axes[0,1].axis('off')
        
        axes[1,0].imshow(t1_view1, cmap='gray')
        axes[1,0].set_title('T1 View 1')
        axes[1,0].axis('off')
        
        axes[1,1].imshow(t1_view2, cmap='gray')
        axes[1,1].set_title('T1 View 2')
        axes[1,1].axis('off')
        
        plt.suptitle(f'Patient {sample["patient_id"]}')
        plt.tight_layout()
        plt.savefig(save_dir / f"sample_{idx}.png")
        plt.close() 