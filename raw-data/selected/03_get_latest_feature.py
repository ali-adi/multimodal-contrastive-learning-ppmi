import pandas as pd
import os

def get_latest_feature_value(feature_name: str, patno: str) -> float:
    """
    Get the latest value for a specified feature and PATNO
    
    Args:
        feature_name (str): Feature name, e.g., 'AGE_AT_VISIT'
        patno (str): Patient number
        
    Returns:
        float: The latest feature value
    """
    # Read feature mapping file
    feature_map = pd.read_csv('Clinical/data/00_all_features.csv')
    
    # Get source file for the feature
    source_file = feature_map[feature_map['Feature'] == feature_name]['Source_Files'].iloc[0]
    
    # Build complete file path
    file_path = os.path.join('Clinical/data', 'selected', source_file)
    
    # Read data file, ensure PATNO column is read as string type
    df = pd.read_csv(file_path, dtype={'PATNO': str})
    
    # Ensure input patno is string type
    patno = str(patno)
    
    # Filter data for specified PATNO
    patient_data = df[df['PATNO'] == patno]
    
    if patient_data.empty:
        raise ValueError(f"No data found for PATNO {patno}")
    
    # If EVENT_ID column exists, sort and get the latest value
    if 'EVENT_ID' in patient_data.columns:
        # Sort by EVENT_ID to ensure we get the latest record
        patient_data = patient_data.sort_values('EVENT_ID')
        # Get feature value from the last row
        latest_value = patient_data.iloc[-1][feature_name]
    else:
        # If no EVENT_ID, get the first value
        latest_value = patient_data.iloc[0][feature_name]
    
    return latest_value

# Usage example
if __name__ == "__main__":
    try:
        feature = "AGE_AT_VISIT"
        patno = "3001"
        value = get_latest_feature_value(feature, patno)
        print(f"Patient {patno}'s latest {feature}: {value}")
    except Exception as e:
        print(f"Error: {str(e)}") 