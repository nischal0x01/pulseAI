"""
Memory-efficient data generator for training on large datasets.
Loads and preprocesses data on-the-fly to avoid memory overflow.
"""

import numpy as np
import os
import pickle
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import h5py
from scipy.io import loadmat


def _load_mat_file(file_path):
    """
    Load a MATLAB file, handling both v7.3 (HDF5) and older formats.
    
    Args:
        file_path: Path to .mat file
        
    Returns:
        Dictionary with 'signals', 'sbp', 'dbp', 'demographics' keys
    """
    try:
        # Try scipy.io.loadmat for older MAT files
        data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        
        if 'Subj_Wins' in data:
            subset = data['Subj_Wins']
            ppg = subset.PPG_Raw
            ecg = subset.ECG_Raw
            signals = np.array([np.vstack([p.ravel(), e.ravel()]) for p, e in zip(ppg, ecg)])
            sbp = subset.SegSBP
            dbp = subset.SegDBP
            
            age = subset.Age
            gender = subset.Gender
            height = subset.Height
            weight = subset.Weight
            demographics = np.column_stack([age, gender, height, weight])
            
            return {
                'signals': signals,
                'sbp': sbp,
                'dbp': dbp,
                'demographics': demographics
            }
        else:
            raise KeyError("Could not find 'Subj_Wins' structure")
            
    except Exception as e:
        msg = str(e).lower()
        if 'hdf' in msg or '7.3' in msg or 'h5py' in msg:
            # MAT v7.3 file - use h5py
            return _load_hdf5_mat_file(file_path)
        else:
            raise


def _load_hdf5_mat_file(file_path):
    """Load MAT v7.3 file using h5py."""
    signals = []
    sbps = []
    dbps = []
    ages = []
    genders = []
    heights = []
    weights = []
    
    with h5py.File(file_path, 'r') as f:
        if 'Subj_Wins' not in f:
            raise KeyError(f"'Subj_Wins' not found in {file_path}")
        
        sw = f['Subj_Wins']
        ppg_refs = sw.get('PPG_Raw')
        
        # Try common ECG keys
        ecg_key = None
        for key in ['ECG_Raw', 'ECG_Raw_II', 'II', 'ECG', 'ecg']:
            if key in sw:
                ecg_key = key
                break
        
        if ppg_refs is None or ecg_key is None:
            raise KeyError(f"Could not find PPG_Raw or ECG signal in {file_path}")
        
        ecg_refs = sw.get(ecg_key)
        n = ppg_refs.shape[1] if ppg_refs.ndim > 1 else ppg_refs.shape[0]
        
        def _get_field_val(name, idx):
            d = sw.get(name)
            if d is None:
                return np.nan
            ref = d[0, idx] if d.ndim == 2 and d.shape[0] == 1 else d[idx]
            try:
                val = f[ref][()]
            except:
                val = ref
            arr = np.asarray(val)
            if arr.size == 1:
                try:
                    return float(arr.item())
                except:
                    return np.nan
            return np.nan
        
        for i in range(int(n)):
            # Dereference PPG
            ppg_ref = ppg_refs[0, i] if ppg_refs.ndim == 2 else ppg_refs[i]
            try:
                ppg_sig = f[ppg_ref][()] if isinstance(ppg_ref, h5py.Reference) else ppg_ref
            except:
                ppg_sig = ppg_ref
            
            # Dereference ECG
            ecg_ref = ecg_refs[0, i] if ecg_refs.ndim == 2 else ecg_refs[i]
            try:
                ecg_sig = f[ecg_ref][()] if isinstance(ecg_ref, h5py.Reference) else ecg_ref
            except:
                ecg_sig = ecg_ref
            
            sig = np.vstack([np.asarray(ppg_sig).ravel(), np.asarray(ecg_sig).ravel()])
            signals.append(sig)
            
            sbps.append(_get_field_val('SegSBP', i))
            dbps.append(_get_field_val('SegDBP', i))
            ages.append(_get_field_val('Age', i))
            genders.append(_get_field_val('Gender', i))
            heights.append(_get_field_val('Height', i))
            weights.append(_get_field_val('Weight', i))
    
    signals = np.stack(signals) if all(s.shape == signals[0].shape for s in signals) else np.array(signals, dtype=object)
    demographics = np.column_stack([ages, genders, heights, weights])
    
    return {
        'signals': signals,
        'sbp': np.array(sbps, dtype=float),
        'dbp': np.array(dbps, dtype=float),
        'demographics': demographics
    }


class LazyBPDataGenerator(keras.utils.Sequence):
    """
    Memory-efficient data generator for BP prediction.
    
    Loads and preprocesses patient data in batches to avoid memory overflow.
    Caches individual patient preprocessed data to disk for faster loading.
    """
    
    def __init__(self, patient_files, data_dir, cache_dir, batch_size=16, 
                 shuffle=True, is_training=True, target_length=875):
        """
        Initialize data generator.
        
        Args:
            patient_files: List of patient .mat file names
            data_dir: Directory containing raw .mat files
            cache_dir: Directory for caching preprocessed patient data
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data after each epoch
            is_training: If True, applies data augmentation
            target_length: Target sequence length for standardization
        """
        self.patient_files = patient_files
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_training = is_training
        self.target_length = target_length
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load patient indices (maps to actual samples)
        self.patient_indices = []
        self.total_samples = 0
        
        print(f"   🔍 Indexing {len(patient_files)} patients...")
        for patient_file in patient_files:
            cache_path = self._get_cache_path(patient_file)
            if os.path.exists(cache_path):
                # Load from cache to get sample count
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                n_samples = cached_data['X'].shape[0]
            else:
                # Quick load to count samples (will cache later)
                data = _load_mat_file(os.path.join(data_dir, patient_file))
                n_samples = data['signals'].shape[0]
            
            # Store mapping: (patient_file, sample_index_in_patient)
            for i in range(n_samples):
                self.patient_indices.append((patient_file, i))
            
            self.total_samples += n_samples
        
        print(f"   ✅ Indexed {self.total_samples} samples from {len(patient_files)} patients")
        
        # Initialize indices for shuffling
        self.indices = np.arange(self.total_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Cache for loaded patients (LRU-style, keep last 5 patients in memory)
        self.patient_cache = {}
        self.cache_size = 5
    
    def _get_cache_path(self, patient_file):
        """Get cache file path for a patient."""
        cache_name = patient_file.replace('.mat', '_preprocessed.pkl')
        return os.path.join(self.cache_dir, cache_name)
    
    def _load_patient_data(self, patient_file):
        """Load and preprocess a single patient's data."""
        cache_path = self._get_cache_path(patient_file)
        
        # Check cache first
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Load and preprocess
        from preprocessing import preprocess_signals
        from feature_engineering import (
            extract_physiological_features,
            standardize_feature_length,
            create_4_channel_input
        )
        
        # Load raw data
        mat_data = _load_mat_file(os.path.join(self.data_dir, patient_file))
        signals = mat_data['signals']  # (n_samples, 2, timesteps) - [PPG, ECG]
        sbp_labels = mat_data['sbp'].flatten()
        dbp_labels = mat_data['dbp'].flatten()
        
        # Preprocess signals
        processed_signals = preprocess_signals(signals)
        
        # Extract features
        ppg_raw = signals[:, 0, :]
        ecg_raw = signals[:, 1, :]
        pat_seqs, hr_seqs, quality_mask = extract_physiological_features(ecg_raw, ppg_raw)
        
        # Filter low quality
        if not quality_mask.all():
            signals = signals[quality_mask]
            processed_signals = processed_signals[quality_mask]
            pat_seqs = pat_seqs[quality_mask]
            hr_seqs = hr_seqs[quality_mask]
            sbp_labels = sbp_labels[quality_mask]
            dbp_labels = dbp_labels[quality_mask]
        
        # Standardize lengths
        pat_seqs = standardize_feature_length(pat_seqs, self.target_length)
        hr_seqs = standardize_feature_length(hr_seqs, self.target_length)
        
        # Create 4-channel input
        X = create_4_channel_input(processed_signals, pat_seqs, hr_seqs)
        
        # Cache preprocessed data
        cached_data = {
            'X': X,
            'sbp': sbp_labels,
            'dbp': dbp_labels
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f)
        
        return cached_data
    
    def _get_patient_data(self, patient_file):
        """Get patient data from cache or load it."""
        # Check in-memory cache
        if patient_file in self.patient_cache:
            return self.patient_cache[patient_file]
        
        # Load from disk
        data = self._load_patient_data(patient_file)
        
        # Add to in-memory cache (LRU)
        self.patient_cache[patient_file] = data
        
        # Limit cache size
        if len(self.patient_cache) > self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.patient_cache))
            del self.patient_cache[oldest_key]
        
        return data
    
    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(self.total_samples / self.batch_size))
    
    def __getitem__(self, batch_idx):
        """Generate one batch of data."""
        # Get indices for this batch
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, self.total_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Collect batch data
        X_batch = []
        y_sbp_batch = []
        y_dbp_batch = []
        
        for idx in batch_indices:
            patient_file, sample_idx = self.patient_indices[idx]
            
            # Load patient data
            patient_data = self._get_patient_data(patient_file)
            
            # Get specific sample
            X_batch.append(patient_data['X'][sample_idx])
            y_sbp_batch.append(patient_data['sbp'][sample_idx])
            y_dbp_batch.append(patient_data['dbp'][sample_idx])
        
        # Convert to numpy arrays
        X_batch = np.array(X_batch)
        y_sbp_batch = np.array(y_sbp_batch)
        y_dbp_batch = np.array(y_dbp_batch)
        
        # Return in format expected by model
        return X_batch, {'sbp_output': y_sbp_batch, 'dbp_output': y_dbp_batch}
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def get_all_data(self):
        """
        Load all data into memory (use only for small val/test sets).
        Returns: X, y_sbp, y_dbp
        """
        X_all = []
        y_sbp_all = []
        y_dbp_all = []
        
        for patient_file, sample_idx in self.patient_indices:
            patient_data = self._get_patient_data(patient_file)
            X_all.append(patient_data['X'][sample_idx])
            y_sbp_all.append(patient_data['sbp'][sample_idx])
            y_dbp_all.append(patient_data['dbp'][sample_idx])
        
        return (np.array(X_all), 
                np.array(y_sbp_all), 
                np.array(y_dbp_all))


def split_patients_for_generators(processed_data_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split patient files into train/val/test sets.
    
    Args:
        processed_data_dir: Directory with .mat files
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
    
    Returns:
        train_files, val_files, test_files
    """
    import random
    
    # Get all patient files
    patient_files = [f for f in os.listdir(processed_data_dir) if f.endswith('.mat')]
    patient_files.sort()  # Sort for reproducibility
    
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(patient_files)
    
    # Split
    n_patients = len(patient_files)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    train_files = patient_files[:n_train]
    val_files = patient_files[n_train:n_train + n_val]
    test_files = patient_files[n_train + n_val:]
    
    print(f"📊 Patient split:")
    print(f"   Train: {len(train_files)} patients")
    print(f"   Val: {len(val_files)} patients")
    print(f"   Test: {len(test_files)} patients")
    
    return train_files, val_files, test_files
