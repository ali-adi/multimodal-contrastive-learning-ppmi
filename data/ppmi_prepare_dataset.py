import os
from pathlib import Path
import pandas as pd
import torch
import argparse

def prepare_ppmi_dataset(
    clinical_path,
    family_history_path,
    imaging_metadata_path,
    image_root_dir,
    output_dir
):
    """
    Prepare complete PPMI dataset with all necessary .pt files
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process tabular data
    print("Processing tabular data...")
    tabular_df = process_tabular_data(
        clinical_path,
        family_history_path,
        output_dir
    )
    
    # Process imaging data
    print("Processing imaging data...")
    image_paths, imaging_df = process_imaging_data(
        imaging_metadata_path,
        image_root_dir,
        output_dir
    )
    
    # Find common patients between tabular and imaging data
    tabular_df['PATNO'] = tabular_df['PATNO'].astype(str)
    imaging_df['PATNO'] = imaging_df['PATNO'].astype(str)
    
    tabular_patients = set(tabular_df['PATNO'])
    imaging_patients = set(imaging_df['PATNO'])
    
    common_patients = tabular_patients.intersection(imaging_patients)
    print(f"Number of patients with tabular data: {len(tabular_patients)}")
    print(f"Number of patients with imaging data: {len(imaging_patients)}")
    print(f"Number of patients with both: {len(common_patients)}")
    
    # Filter data to only include common patients
    tabular_df = tabular_df[tabular_df['PATNO'].isin(common_patients)]
    imaging_df = imaging_df[imaging_df['PATNO'].isin(common_patients)]
    
    # Save filtered data
    torch.save(tabular_df['PATNO'].values, output_dir / 'patient_ids.pt')
    
    # Count unique patients
    unique_patients = len(tabular_df['PATNO'].unique())
    
    print("Dataset preparation complete!")
    print(f"Number of patients with tabular data: {len(tabular_df)}")
    print(f"Number of unique patients: {unique_patients}")
    print(f"Number of image paths: {len(image_paths)}")
    print(f"Output directory: {output_dir}")

def process_tabular_data(clinical_path, family_history_path, output_dir):
    """
    Process tabular data from clinical and family history files
    """
    # Load clinical features
    print(f"Loading clinical features from {clinical_path}")
    df_features = pd.read_csv(clinical_path)
    print(f"Clinical features columns: {df_features.columns.tolist()}")
    print(f"Clinical features shape: {df_features.shape}")
    
    print(f"Loading family history from {family_history_path}")
    df_family = pd.read_csv(family_history_path)
    print(f"Family history columns: {df_family.columns.tolist()}")
    print(f"Family history shape: {df_family.shape}")
    
    # Filter for baseline or screening visits
    print("Filtering for BL or SC visits")
    df_features = df_features[df_features['EVENT_ID'].isin(['BL', 'SC'])]
    df_family = df_family[df_family['EVENT_ID'].isin(['BL', 'SC'])]
    print(f"After filtering, clinical features shape: {df_features.shape}")
    print(f"After filtering, family history shape: {df_family.shape}")
    
    # Sort by EVENT_ID to prioritize BL over SC and take first record for each patient
    print("Sorting and grouping by PATNO")
    df_features = df_features.sort_values('EVENT_ID').groupby('PATNO').first().reset_index()
    df_family = df_family.sort_values('EVENT_ID').groupby('PATNO').first().reset_index()
    
    # Select specific columns from features file
    print("Selecting specific columns")
    df_features = df_features[['PATNO', 'FEATBRADY', 'FEATPOSINS', 'FEATRIGID', 'FEATTREMOR']]
    
    # Select family history columns
    df_family = df_family[['PATNO', 'BIODADPD', 'BIOMOMPD', 'FULBROPD', 'FULSISPD',
                           'MAGFATHPD', 'MAGMOTHPD', 'PAGFATHPD', 'PAGMOTHPD']]
    
    # Merge features and family history with outer join
    print("Merging features and family history")
    df_combined = pd.merge(df_features, df_family, on='PATNO', how='outer')
    
    # Fill NaN with -1 (as in the notebook)
    df_combined = df_combined.fillna(-1)
    
    print(f"Number of patients after processing tabular data: {len(df_combined)}")
    
    # Save processed data
    numeric_columns = df_combined.columns.difference(['PATNO'])
    torch.save(torch.tensor(df_combined[numeric_columns].values), output_dir / 'data_tabular.pt')
    torch.save(torch.tensor([2] * len(numeric_columns)), output_dir / 'field_lengths_tabular.pt')  # Binary features
    torch.save(df_combined['PATNO'].values, output_dir / 'patient_ids.pt')
    
    return df_combined

def process_imaging_data(imaging_metadata_path, image_root_dir, output_dir):
    """Process imaging data and create .pt files"""
    # Read imaging metadata
    df_image = pd.read_csv(imaging_metadata_path)
    
    # Standardize 'Subject' and 'Description'
    df_image['Subject'] = df_image['Subject'].astype(str).str.strip()
    df_image['Description'] = df_image['Description'].astype(str).str.strip()
    
    # Convert "Acq Date" to datetime
    df_image['Acq Date'] = pd.to_datetime(df_image['Acq Date'], format='%m/%d/%Y', errors='coerce')
    
    # Filter for T1-anatomical MRI entries
    df_t1 = df_image[(df_image['Modality'] == 'MRI') & (df_image['Description'] == 'T1-anatomical')]
    df_t1_latest = df_t1.sort_values('Acq Date').groupby('Subject', as_index=False).tail(1)
    df_t1_latest = df_t1_latest.rename(columns={'Acq Date': 'T1_Acq_Date',
                                                'Image Data ID': 'T1_Image_Data_ID'})
    
    # Filter for DTI FA map-MRI entries
    df_dti = df_image[(df_image['Modality'] == 'DTI') & (df_image['Description'] == 'FA map-MRI')]
    df_dti_latest = df_dti.sort_values('Acq Date').groupby('Subject', as_index=False).tail(1)
    df_dti_latest = df_dti_latest.rename(columns={'Acq Date': 'DTI_Acq_Date',
                                                  'Image Data ID': 'DTI_Image_Data_ID'})
    
    # Merge the two imaging datasets
    df_imaging = pd.merge(df_t1_latest, df_dti_latest, on='Subject', how='inner')
    
    # Prepare imaging table
    df_imaging_table = df_imaging[['Subject', 'T1_Acq_Date', 'T1_Image_Data_ID',
                                   'DTI_Acq_Date', 'DTI_Image_Data_ID']].copy()
    df_imaging_table = df_imaging_table.rename(columns={'Subject': 'PATNO'})
    
    # Create paths for images
    image_paths = []
    for _, row in df_imaging_table.iterrows():
        patient_id = row['PATNO']
        t1_id = row['T1_Image_Data_ID']
        dti_id = row['DTI_Image_Data_ID']
        
        # Find T1 image
        t1_path = find_image_path(patient_id, t1_id, 'T1-anatomical', image_root_dir)
        if t1_path:
            image_paths.append(t1_path)
        
        # Find DTI image
        dti_path = find_image_path(patient_id, dti_id, 'FA_map-MRI', image_root_dir)
        if dti_path:
            image_paths.append(dti_path)
    
    # Save image paths
    torch.save(image_paths, output_dir / 'data_imaging.pt')
    
    return image_paths, df_imaging_table

def find_image_path(patient_id, image_id, image_type, image_root_dir):
    """Find the path to an image file given the patient ID and image ID."""
    # Convert patient_id to string and ensure it's padded correctly
    patient_id = str(patient_id)
    
    # The patient directory
    patient_dir = os.path.join(image_root_dir, 'PPMI', patient_id)
    if not os.path.exists(patient_dir):
        print(f"Patient directory not found: {patient_dir}")
        return None
        
    # The image type directory (T1-anatomical or FA_map-MRI)
    image_type_dir = os.path.join(patient_dir, image_type)
    if not os.path.exists(image_type_dir):
        print(f"Image type directory not found: {image_type_dir}")
        return None
        
    # List all date directories
    date_dirs = [d for d in os.listdir(image_type_dir) 
                if os.path.isdir(os.path.join(image_type_dir, d)) and not d.startswith('.')]
    
    # Search through each date directory
    for date_dir in date_dirs:
        date_path = os.path.join(image_type_dir, date_dir)
        
        # List all image ID directories
        img_dirs = [d for d in os.listdir(date_path) 
                   if os.path.isdir(os.path.join(date_path, d)) and not d.startswith('.')]
        
        # Search through each image ID directory
        for img_dir in img_dirs:
            img_path = os.path.join(date_path, img_dir)
            
            # List all .nii files
            nii_files = [f for f in os.listdir(img_path) 
                        if f.endswith('.nii') and not f.startswith('.')]
            
            # If we find any .nii files, return the path to the first one
            if nii_files:
                return os.path.join(img_path, nii_files[0])
    
    print(f"No .nii file found for patient {patient_id}, image type {image_type}")
    return None

# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare PPMI dataset')
    parser.add_argument('--clinical_path', type=str, required=True, help='Path to clinical features CSV')
    parser.add_argument('--family_history_path', type=str, required=True, help='Path to family history CSV')
    parser.add_argument('--imaging_metadata_path', type=str, required=True, help='Path to imaging metadata CSV')
    parser.add_argument('--image_root_dir', type=str, required=True, help='Root directory for images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed data')
    
    args = parser.parse_args()
    
    prepare_ppmi_dataset(
        clinical_path=args.clinical_path,
        family_history_path=args.family_history_path,
        imaging_metadata_path=args.imaging_metadata_path,
        image_root_dir=args.image_root_dir,
        output_dir=args.output_dir
    )
