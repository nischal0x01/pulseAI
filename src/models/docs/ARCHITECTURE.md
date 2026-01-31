# Module Dependency Graph

```
                                 model.py
                                    |
                                    v
                                 train.py
                                    |
        +---------------------------+---------------------------+
        |                           |                           |
        v                           v                           v
   data_loader.py           preprocessing.py          feature_engineering.py
        |                           |                           |
        |                           v                           |
        |                        config.py <-------------------+
        |                           ^                           
        |                           |                           
        +----------> model_builder.py <-----------+
        |                           ^             |
        v                           |             v
    utils.py                        +--------> train.py
        ^                                         |
        |                                         |
        +-----------------------------------------+

```

## Module Relationships

### Core Pipeline Flow
1. **model.py** → Entry point
2. **train.py** → Main orchestrator
3. **config.py** → Configuration (imported by all modules)
4. **data_loader.py** → Load MAT files
5. **preprocessing.py** → Filter & normalize signals
6. **feature_engineering.py** → Extract PAT & HR features
7. **utils.py** → Data validation & normalization
8. **model_builder.py** → Create neural network
9. **train.py** → Train & evaluate models

### Import Dependencies

**train.py imports:**
- config (EPOCHS, BATCH_SIZE, VERBOSE, PROCESSED_DATA_DIR)
- data_loader (load_aggregate_data)
- preprocessing (preprocess_signals, create_subject_wise_splits)
- feature_engineering (extract_physiological_features, standardize_feature_length, 
                       create_baseline_features, create_4_channel_input)
- model_builder (create_phys_informed_model)
- utils (_ensure_finite, _ensure_finite_1d, normalize_data)

**preprocessing.py imports:**
- config (TARGET_LENGTH, SAMPLING_RATE, filter params, split ratios)

**feature_engineering.py imports:**
- config (SAMPLING_RATE, TARGET_LENGTH, HR bounds, PAT bounds)

**model_builder.py imports:**
- config (LEARNING_RATE, model architecture params)

**data_loader.py:**
- No internal imports (standalone)

**utils.py:**
- No internal imports (standalone)

**model.py imports:**
- train (main function only)

## Data Flow

```
MAT Files (data/processed/)
         |
         v
  load_aggregate_data() [data_loader.py]
         |
         v
  signals, labels, demographics, patient_ids
         |
         +---> preprocess_signals() [preprocessing.py]
         |              |
         |              v
         |     processed_signals (filtered & normalized)
         |
         +---> create_subject_wise_splits() [preprocessing.py]
         |              |
         |              v
         |     train_mask, val_mask, test_mask
         |
         v
  extract_physiological_features() [feature_engineering.py]
         |
         v
  PAT sequences, HR sequences
         |
         v
  create_baseline_features() [feature_engineering.py]
         |
         +---> Linear Regression Model
         |              |
         |              v
         |     Baseline predictions & evaluation
         |
         v
  create_4_channel_input() [feature_engineering.py]
         |
         v
  4-channel input (PPG + ECG + PAT + HR)
         |
         v
  normalize_data() [utils.py]
         |
         v
  create_phys_informed_model() [model_builder.py]
         |
         v
  Trained CNN+GRU Model
         |
         v
  Final predictions & evaluation
```

## Configuration Flow

All modules that need configuration import from config.py:

```
config.py
    |
    +---> preprocessing.py (signal processing params)
    |
    +---> feature_engineering.py (feature extraction params)
    |
    +---> model_builder.py (model architecture params)
    |
    +---> train.py (training params & paths)
```

This ensures a single source of truth for all configuration values.
