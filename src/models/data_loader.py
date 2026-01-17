"""
Data loading utilities for blood pressure prediction.
Handles loading and aggregating patient data from MAT files.
"""

import numpy as np
import h5py
import os
from pathlib import Path
from scipy.io import loadmat


def _read_subj_wins(path, field='Subj_Wins', sample_limit=None):
    """Read per-window data from a MAT v7.3 'Subj_Wins' group without dereferencing the whole file.

    Extracts numeric Age, Gender, Height, and Weight per-window when available. Missing values
    are returned as np.nan (not 0) to avoid masking missingness.
    """
    signals = []
    sbps = []
    dbps = []
    ages = []
    genders = []
    heights = []
    weights = []
    with h5py.File(path, 'r') as f2:
        if field not in f2:
            raise KeyError(f'{field} not found in {path}')
        sw = f2[field]

        # Get signal references
        ppg_refs = sw.get('PPG_Raw')

        # Try common keys for ECG signal
        ecg_key_found = None
        for key in ['ECG_Raw', 'ECG_Raw_II', 'II', 'ECG', 'ecg']:
            if key in sw:
                ecg_key_found = key
                break

        if ppg_refs is None or ecg_key_found is None:
            raise KeyError(f'Could not find PPG_Raw or a valid ECG signal key in {list(sw.keys())}')

        ecg_refs = sw.get(ecg_key_found)

        n = ppg_refs.shape[1] if getattr(ppg_refs, 'ndim', 0) > 1 else ppg_refs.shape[0]
        if sample_limit is not None:
            n = min(int(n), int(sample_limit))

        for i in range(int(n)):
            # Dereference PPG signal
            ppg_ref = ppg_refs[0, i] if getattr(ppg_refs, 'ndim', 0) == 2 and ppg_refs.shape[0] == 1 else ppg_refs[i]
            try:
                ppg_sig = f2[ppg_ref][()] if isinstance(ppg_ref, h5py.Reference) else ppg_ref
            except Exception:
                ppg_sig = ppg_ref

            # Dereference ECG signal
            ecg_ref = ecg_refs[0, i] if getattr(ecg_refs, 'ndim', 0) == 2 and ecg_refs.shape[0] == 1 else ecg_refs[i]
            try:
                ecg_sig = f2[ecg_ref][()] if isinstance(ecg_ref, h5py.Reference) else ecg_ref
            except Exception:
                ecg_sig = ecg_ref

            # Combine signals into a (2, L) array
            sig = np.vstack([np.asarray(ppg_sig).ravel(), np.asarray(ecg_sig).ravel()])
            signals.append(sig)

            def _get_field_val(name):
                d = sw.get(name)
                if d is None:
                    return np.nan
                ref2 = d[0, i] if getattr(d, 'ndim', 0) == 2 and d.shape[0] == 1 else d[i]
                try:
                    val = f2[ref2][()]
                except Exception:
                    val = ref2
                arr = np.asarray(val)
                if arr.size == 1:
                    try:
                        return float(arr.item())
                    except Exception:
                        return np.nan
                return np.nan

            sbps.append(_get_field_val('SegSBP'))
            dbps.append(_get_field_val('SegDBP'))
            ages.append(_get_field_val('Age'))
            genders.append(_get_field_val('Gender'))
            heights.append(_get_field_val('Height'))
            weights.append(_get_field_val('Weight'))

    sigs = np.array(signals, dtype=object) if any(s.shape != signals[0].shape for s in signals) else np.stack(signals)

    sbp_arr = np.array(sbps, dtype=float)
    dbp_arr = np.array(dbps, dtype=float)
    age_arr = np.array(ages, dtype=float)

    def _gdecode(v):
        try:
            if v is None:
                return np.nan
            if isinstance(v, (bytes, bytearray)):
                s = v.decode(errors='ignore').strip().upper()
            elif isinstance(v, str):
                s = v.strip().upper()
            else:
                iv = float(v)
                if iv in (1.0, 0.0):
                    return iv
                if int(iv) in (77, 70):
                    s = chr(int(iv))
                else:
                    return np.nan
            return 1.0 if s == 'M' else 0.0 if s == 'F' else np.nan
        except Exception:
            return np.nan

    gender_arr = np.array([_gdecode(g) for g in genders], dtype=float)
    height_arr = np.array(heights, dtype=float)
    weight_arr = np.array(weights, dtype=float)

    demographics = np.column_stack([age_arr, gender_arr, height_arr, weight_arr])
    return sigs, sbp_arr, demographics


def load_aggregate_data(processed_dir='../data/processed'):
    """Load and aggregate data from all patient .mat files."""
    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        print("❌ Processed data directory not found.")
        return None, None, None, None

    mat_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.mat')])
    if not mat_files:
        print("❌ No .mat files found in the processed data directory.")
        return None, None, None, None

    all_signals, all_labels, all_demographics, all_patient_ids = [], [], [], []

    print(f"🔄 Loading data from {len(mat_files)} patient files...")
    for file_name in mat_files:
        patient_id = file_name.split('.')[0]
        file_path = str(processed_dir / file_name)

        signals, sbp, demographics = None, None, None

        try:
            # First, try to load with scipy.io.loadmat for older MAT files
            data = loadmat(file_path, squeeze_me=True, struct_as_record=False)

            if 'Subj_Wins' in data:
                subset = data['Subj_Wins']
                ppg = subset.PPG_Raw
                ecg = subset.ECG_Raw
                # Ensure signals are correctly stacked
                signals = np.array([np.vstack([p.ravel(), e.ravel()]) for p, e in zip(ppg, ecg)])
                sbp = subset.SegSBP

                age = subset.Age
                gender = subset.Gender
                height = subset.Height
                weight = subset.Weight
                demographics = np.column_stack([age, gender, height, weight])
            else:
                print(f"   ❌ Could not find 'Subj_Wins' structure in {file_name} using scipy. Skipping file.")
                continue

        except Exception as e:
            # If scipy fails, check if it's an HDF5 file and use the h5py reader
            msg = str(e).lower()
            if 'hdf' in msg or '7.3' in msg or 'h5py' in msg:
                try:
                    signals, sbp, demographics = _read_subj_wins(file_path)
                except Exception as h5_e:
                    print(f"   ❌ Failed to load {file_name} with HDF5 reader. Error: {h5_e}")
                    continue
            else:
                print(f"   ❌ Failed to load {file_name} with scipy.io.loadmat. Error: {e}")
                continue

        if signals is not None and len(signals) > 0:
            all_signals.append(signals)
            all_labels.append(sbp)
            all_demographics.append(demographics)
            all_patient_ids.extend([patient_id] * len(signals))

    if not all_signals:
        print("❌ No data could be loaded from any files.")
        return None, None, None, None

    # Concatenate all data
    signals_agg = np.vstack(all_signals)
    labels_agg = np.concatenate(all_labels)
    demographics_agg = np.vstack(all_demographics)
    patient_ids_agg = np.array(all_patient_ids)

    # Filter out rows with NaN labels
    valid_indices = ~np.isnan(labels_agg)
    signals_agg = signals_agg[valid_indices]
    labels_agg = labels_agg[valid_indices]
    demographics_agg = demographics_agg[valid_indices]
    patient_ids_agg = patient_ids_agg[valid_indices]

    print(f"✅ Aggregated data loaded successfully!")
    print(f"   - Total signals: {signals_agg.shape}")
    print(f"   - Total labels: {labels_agg.shape}")
    print(f"   - Total demographics: {demographics_agg.shape}")
    print(f"   - Total patient IDs: {patient_ids_agg.shape}")
    print(f"   - Unique patients: {len(np.unique(patient_ids_agg))}")

    return signals_agg, labels_agg, demographics_agg, patient_ids_agg
