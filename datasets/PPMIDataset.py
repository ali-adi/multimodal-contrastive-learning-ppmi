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
        augmentation = None,
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
            augmentation: Transforms to apply for augmentation (can be a list or a composed transform)
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
        self.augmentation_rate = augmentation_rate
        
        # Load data
        self.data_tabular = torch.load(self.processed_data_dir / 'data_tabular.pt', weights_only=False)
        self.field_lengths_tabular = torch.load(self.processed_data_dir / 'field_lengths_tabular.pt', weights_only=False)
        self.patient_ids = torch.load(self.processed_data_dir / 'patient_ids.pt', weights_only=False)
        self.image_paths = torch.load(self.processed_data_dir / 'data_imaging.pt', weights_only=False)
        
        print("Loaded data:")
        print(f"Number of patients in tabular data: {len(self.patient_ids)}")
        print(f"Number of image paths: {len(self.image_paths)}")
        
        # Create a mapping from patient ID to index
        self.patient_to_idx = {str(pid): i for i, pid in enumerate(self.patient_ids)}
        print(f"Patient IDs in tabular data: {list(self.patient_to_idx.keys())[:5]}...")
        
        # Create a mapping from patient ID to image paths
        self.patient_to_images = {}
        for img_path in self.image_paths:
            try:
                # Try different methods to extract patient ID
                path_parts = img_path.split('/')
                patient_id = None
                
                # Method 1: Look for PPMI directory and take ID after it
                if 'PPMI' in path_parts:
                    ppmi_index = path_parts.index('PPMI')
                    if ppmi_index + 2 < len(path_parts):
                        patient_id = path_parts[ppmi_index + 2]
                
                # Method 2: Extract from filename (often contains the ID)
                if patient_id is None and len(path_parts) > 0:
                    filename = path_parts[-1]
                    if 'PPMI_' in filename:
                        # Format: PPMI_3837_MR_T1-anatomical_...
                        id_part = filename.split('_')[1]
                        if id_part.isdigit():
                            patient_id = id_part
                
                if patient_id and patient_id in self.patient_to_idx:
                    if patient_id not in self.patient_to_images:
                        self.patient_to_images[patient_id] = []
                    self.patient_to_images[patient_id].append(img_path)
            except Exception as e:
                print(f"Error extracting patient ID from {img_path}: {str(e)}")
        
        # Create a list of valid indices (patients that have imaging data)
        self.valid_indices = []
        for patient_id, _ in self.patient_to_images.items():
            if patient_id in self.patient_to_idx:
                self.valid_indices.append(self.patient_to_idx[patient_id])
        
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
            # Default augmentations
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
        elif isinstance(augmentation, list):
            # Handle string-based augmentation names
            if all(isinstance(a, str) for a in augmentation):
                transform_list = []
                for aug_name in augmentation:
                    if aug_name == 'random_crop':
                        # Use Resize instead of RandomCrop since torchio doesn't have RandomCrop
                        transform_list.append(tio.Resize((img_size, img_size, img_size), p=0.5))
                    elif aug_name == 'random_horizontal_flip':
                        transform_list.append(tio.RandomFlip(axes=(1,), p=0.5))
                    elif aug_name == 'random_vertical_flip':
                        transform_list.append(tio.RandomFlip(axes=(2,), p=0.5))
                    elif aug_name == 'random_rotation':
                        # Use RandomAffine for rotation since TorchIO doesn't have RandomRotation
                        transform_list.append(tio.RandomAffine(
                            scales=(1.0, 1.0),  # No scaling
                            degrees=15,  # 15 degrees rotation
                            translation=0,  # No translation
                            p=0.5
                        ))
                    elif aug_name == 'random_affine':
                        transform_list.append(tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5, p=0.5))
                    elif aug_name == 'random_noise':
                        transform_list.append(tio.RandomNoise(p=0.5))
                    elif aug_name == 'random_bias_field':
                        transform_list.append(tio.RandomBiasField(p=0.5))
                    elif aug_name == 'normalize':
                        transform_list.append(tio.RescaleIntensity((0, 1)))
                    else:
                        print(f"Warning: Unknown augmentation '{aug_name}' will be skipped")
                self.augmentation = tio.Compose(transform_list)
            else:
                # Convert list of transform objects to Compose
                self.augmentation = tio.Compose(augmentation)
        else:
            # Use provided composed transform
            self.augmentation = augmentation
            
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
        try:
            # Load NIfTI file
            img = nib.load(img_path)
            # Convert to tensor
            img_data = torch.tensor(img.get_fdata(), dtype=torch.float32)
            # Add channel dimension if needed
            if len(img_data.shape) == 3:
                img_data = img_data.unsqueeze(0)
            # Apply transform
            transformed_img = self.transform(tio.Subject(image=tio.ScalarImage(tensor=img_data)))
            return transformed_img.image.data
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Create a fallback image (all zeros) with the right dimensions
            fallback_img = torch.zeros((1, self.img_size, self.img_size, self.img_size), dtype=torch.float32)
            return fallback_img
            
    def generate_imaging_views(self, index: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Generate augmented views of a subject's imaging data
        
        Args:
            index: Subject index
            
        Returns:
            Tuple of (list of augmented views, original image)
        """
        # Get patient ID for this index
        patient_id = str(self.patient_ids[self.valid_indices[index]])
        
        # Find the corresponding image path
        if patient_id not in self.patient_to_images or not self.patient_to_images[patient_id]:
            raise ValueError(f"No images found for patient {patient_id}")
        
        # Use the first image for this patient
        img_path = self.patient_to_images[patient_id][0]
        
        # Load and preprocess the image
        t1_img = self.load_nifti_image(img_path)
        
        # Create original view (unaugmented)
        original_subject = tio.Subject(image=tio.ScalarImage(tensor=t1_img))
        
        # Create augmented view if in training mode
        if self.train and random.random() < self.augmentation_rate:
            # Use augmentation as a callable (we've already converted lists to Compose in __init__)
            augmented_subject = self.augmentation(original_subject)
        else:
            # No augmentation
            augmented_subject = original_subject
        
        # Extract the tensors from the subjects
        original_tensor = original_subject.image.data
        augmented_tensor = augmented_subject.image.data
        
        return [augmented_tensor], original_tensor
        
    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset
        
        Args:
            index: Index of the sample
            
        Returns:
            Tuple of (imaging_views, tabular_views, patient_id, unaugmented_image)
        """
        # Get tabular data
        tabular_data = self.data_tabular[self.valid_indices[index]]
        
        # Apply one-hot encoding if needed
        if self.one_hot_tabular:
            tabular_data = self.one_hot_encode(tabular_data)
        
        # Create tabular views (original and corrupted)
        tabular_views = [tabular_data]
        if self.train:
            # Add corrupted view for training
            corrupted_data = self.corrupt(tabular_data)
            tabular_views.append(corrupted_data)
        else:
            # Add identical view for validation/testing
            tabular_views.append(tabular_data)
        
        # Get imaging views
        imaging_views, unaugmented_image = self.generate_imaging_views(index)
        
        # Get patient ID 
        patient_id = self.patient_ids[self.valid_indices[index]]
        
        return imaging_views, tabular_views, patient_id, unaugmented_image
        
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