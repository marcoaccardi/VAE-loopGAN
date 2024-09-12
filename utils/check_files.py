import os
import numpy as np
import librosa
import json

def analyze_npy_file(file_path):
    """
    Analyze a .npy file and return its shape and basic statistics.
    """
    data = np.load(file_path)
    analysis = {
        "file_path": file_path,
        "shape": data.shape,
        "min_value": float(np.min(data)),  # Convert to native Python types
        "max_value": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data))
    }
    return analysis

def analyze_wav_file(file_path):
    """
    Analyze a .wav file and return its properties like duration and sample rate.
    """
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    amplitude_range = (float(np.min(y)), float(np.max(y)))  # Ensure conversion to float
    analysis = {
        "file_path": file_path,
        "sample_rate": sr,
        "duration": duration,
        "amplitude_range": amplitude_range,
        "mean_amplitude": float(np.mean(y)),  # Convert NumPy types to Python float
        "std_amplitude": float(np.std(y))
    }
    return analysis

def analyze_dataset(npy_folder, wav_folder, output_json):
    """
    Analyze all .npy and .wav files in the given folders and generate a JSON report.
    """
    dataset_info = {
        "npy_files": [],
        "wav_files": []
    }

    # Analyze all .npy files
    for npy_file in os.listdir(npy_folder):
        if npy_file.endswith('.npy'):
            npy_file_path = os.path.join(npy_folder, npy_file)
            npy_analysis = analyze_npy_file(npy_file_path)
            dataset_info["npy_files"].append(npy_analysis)

    # Analyze all .wav files
    for wav_file in os.listdir(wav_folder):
        if wav_file.endswith('.wav'):
            wav_file_path = os.path.join(wav_folder, wav_file)
            wav_analysis = analyze_wav_file(wav_file_path)
            dataset_info["wav_files"].append(wav_analysis)

    # Save analysis to JSON file
    with open(output_json, 'w') as f:
        json.dump(dataset_info, f, indent=4)

    print(f"Dataset analysis saved to {output_json}")
# Example usage
if __name__ == "__main__":
    npy_folder = "D:\LV-NTF+LoopGAN\data\FSL10K-mel_spec"
    wav_folder = "D:\LV-NTF+LoopGAN\data\FSL10K-trimmed"
    output_json = "D:\LV-NTF+LoopGAN\data\dataset_analysis.json"

    analyze_dataset(npy_folder, wav_folder, output_json)
