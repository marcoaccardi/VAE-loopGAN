conda install pytorch torchvision torchaudio pytorch-cuda=12.3 -c pytorch -c nvidia

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, metadata, feature_extraction_fn, required_input_dim=128):
        self.audio_paths = audio_paths
        self.metadata = metadata or {}
        self.feature_extraction_fn = feature_extraction_fn
        self.required_input_dim = required_input_dim

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_file = self.audio_paths[idx]
        
        if not os.path.exists(audio_file):
            print(f"File not found: {audio_file}")
            return None

        try:
            # Extract features from the audio file (returns a dictionary)
            feature_dict = self.feature_extraction_fn(audio_file)
            if feature_dict is None:
                raise ValueError("Feature extraction returned None")
        except Exception as e:
            print(f"Error loading or extracting features from file: {audio_file}, Error: {e}")
            return None

        # Combine features into a single numerical array
        # You can select specific features you want, here are examples
        scalar_features = np.array([
            feature_dict.get('bpm', 0), 
            feature_dict.get('zero_crossing_rate', 0), 
            feature_dict.get('spectral_centroid', 0),
            feature_dict.get('spectral_bandwidth', 0),
            feature_dict.get('rms_energy', 0),
            feature_dict.get('energy', 0)
        ], dtype=np.float32)

        # Flatten array features like MFCCs and chroma
        mfcc_features = feature_dict.get('mfccs', np.zeros(13))  # MFCCs (13 elements)
        chroma_features = feature_dict.get('chroma', np.zeros(12))  # Chroma (12 elements)

        # Combine scalar and array features into one vector
        numeric_features = np.concatenate((scalar_features, mfcc_features, chroma_features)).astype(np.float32)

        # Pad features to the required input dimension (128)
        if numeric_features.size < self.required_input_dim:
            padded_features = np.pad(numeric_features, (0, self.required_input_dim - numeric_features.size), 'constant')
        else:
            padded_features = numeric_features[:self.required_input_dim]

        # Convert to PyTorch tensor
        return torch.tensor(padded_features, dtype=torch.float32)