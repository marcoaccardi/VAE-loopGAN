import json
import numpy as np
import os

def check_and_normalize_scaling_with_report(npy_folder, output_json):
    """
    Check the scaling of .npy files, normalize if needed, and generate a JSON report.
    
    Args:
    - npy_folder (str): Path to the folder containing .npy files.
    - output_json (str): Path to save the JSON report.
    
    Returns:
    - None: Saves the JSON report to the specified file.
    """
    report = []

    for file_name in os.listdir(npy_folder):
        if file_name.endswith('.npy'):
            file_path = os.path.join(npy_folder, file_name)
            mel_spec = np.load(file_path)

            min_val, max_val = np.min(mel_spec), np.max(mel_spec)
            normalized = False

            # Apply normalization if needed
            if min_val < 0 or max_val > 1:
                mel_spec = (mel_spec - min_val) / (max_val - min_val + 1e-6)
                np.save(file_path, mel_spec)
                normalized = True

            # Append to report
            report.append({
                "file_name": file_name,
                "min_value": float(min_val),
                "max_value": float(max_val),
                "normalized": normalized
            })
    
    # Save the report to a JSON file
    with open(output_json, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Scaling report saved to {output_json}")

# Path to folder and output JSON
npy_folder = r"D:\LV-NTF+LoopGAN\data\FSL10K-mel_spec"
output_json = r"D:\LV-NTF+LoopGAN\data\scaling_report.json"
check_and_normalize_scaling_with_report(npy_folder, output_json)

def analyze_dataset_balance_with_report(npy_folder, output_json):
    """
    Analyze the mean and standard deviation for .npy files and save results to a JSON report.
    
    Args:
    - npy_folder (str): Path to the folder containing .npy files.
    - output_json (str): Path to save the JSON report.
    
    Returns:
    - None: Saves the JSON report.
    """
    report = []

    for file_name in os.listdir(npy_folder):
        if file_name.endswith('.npy'):
            file_path = os.path.join(npy_folder, file_name)
            mel_spec = np.load(file_path)

            # Get mean and std
            mean_val = np.mean(mel_spec)
            std_val = np.std(mel_spec)

            # Append to report
            report.append({
                "file_name": file_name,
                "mean_value": float(mean_val),
                "std_value": float(std_val)
            })
    
    # Save the report to a JSON file
    with open(output_json, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Balance report saved to {output_json}")

# Path to folder and output JSON
output_json = r"D:\LV-NTF+LoopGAN\data\balance_report.json"
analyze_dataset_balance_with_report(npy_folder, output_json)

import json
import numpy as np
import os

def analyze_dataset_balance_with_report(npy_folder, output_json):
    """
    Analyze the mean and standard deviation for .npy files and save results to a JSON report.
    
    Args:
    - npy_folder (str): Path to the folder containing .npy files.
    - output_json (str): Path to save the JSON report.
    
    Returns:
    - mean_vals, std_vals (list): Lists of mean and standard deviation values.
    """
    report = []
    mean_vals = []  # List to store mean values
    std_vals = []  # List to store std deviation values

    for file_name in os.listdir(npy_folder):
        if file_name.endswith('.npy'):
            file_path = os.path.join(npy_folder, file_name)
            mel_spec = np.load(file_path)

            # Get mean and std
            mean_val = np.mean(mel_spec)
            std_val = np.std(mel_spec)

            # Store values for later outlier detection
            mean_vals.append(mean_val)
            std_vals.append(std_val)

            # Append to report
            report.append({
                "file_name": file_name,
                "mean_value": float(mean_val),
                "std_value": float(std_val)
            })
    
    # Save the report to a JSON file
    with open(output_json, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Balance report saved to {output_json}")

    return mean_vals, std_vals

# Path to folder and output JSON

output_json = r"D:\LV-NTF+LoopGAN\data\balance_report.json"
mean_vals, std_vals = analyze_dataset_balance_with_report(npy_folder, output_json)



def detect_outliers_with_report(mean_vals, std_vals, threshold=3.0, output_json="outliers_report.json"):
    """
    Detects outliers based on z-score thresholds for mean and standard deviation, and logs them to a JSON report.
    
    Args:
    - mean_vals (list): List of mean values across the dataset.
    - std_vals (list): List of standard deviation values across the dataset.
    - threshold (float): Z-score threshold for detecting outliers.
    - output_json (str): Path to save the JSON report.
    
    Returns:
    - None: Saves the outlier information to a JSON report.
    """
    mean_mean, mean_std = np.mean(mean_vals), np.std(mean_vals)
    std_mean, std_std = np.mean(std_vals), np.std(std_vals)

    outliers = []
    
    for i, (mean, std) in enumerate(zip(mean_vals, std_vals)):
        mean_z = (mean - mean_mean) / mean_std
        std_z = (std - std_mean) / std_std
        if abs(mean_z) > threshold or abs(std_z) > threshold:
            outliers.append({
                "index": i,
                "mean_z_score": float(mean_z),
                "std_z_score": float(std_z)
            })

    # Save the report to a JSON file
    with open(output_json, 'w') as f:
        json.dump(outliers, f, indent=4)
    print(f"Outliers report saved to {output_json}")

# Detect outliers and save to JSON
output_json = r"D:\LV-NTF+LoopGAN\data\outliers_report.json"
detect_outliers_with_report(mean_vals, std_vals, threshold=3.0, output_json=output_json)


# Detect outliers and save to JSON
detect_outliers_with_report(mean_vals, std_vals, threshold=3.0)


def preprocess_and_standardize_with_report(npy_folder, output_json):
    """
    Standardize the preprocessing of mel-spectrograms and log results in a JSON report.
    
    Args:
    - npy_folder (str): Path to the folder containing .npy files.
    - output_json (str): Path to save the JSON report.
    
    Returns:
    - None: Saves the standardization process to a JSON report.
    """
    report = []

    for file_name in os.listdir(npy_folder):
        if file_name.endswith('.npy'):
            file_path = os.path.join(npy_folder, file_name)
            mel_spec = np.load(file_path)

            # Normalize to [0, 1]
            mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
            np.save(file_path, mel_spec)

            report.append({
                "file_name": file_name,
                "status": "standardized"
            })

    # Save the report to a JSON file
    with open(output_json, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Standardization report saved to {output_json}")

# Path to folder and output JSON
output_json = "D:\LV-NTF+LoopGAN\data\standardization_report.json"
preprocess_and_standardize_with_report(npy_folder, output_json)
