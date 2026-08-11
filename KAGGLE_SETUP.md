# 🚀 Training Cuffless Blood Pressure on Kaggle

This project is fully configured to run and train on Kaggle using the [MIMIC-III PulseDB Dataset](https://www.kaggle.com/datasets/yesellee/mimiciii-pulsedb) (a standardized database of ECG, PPG, and BP waveforms).

The repository has been updated to support Kaggle environment paths and includes **`uv`** package manager integration for fast, isolated setups.

---

## 🛠️ Automated Setup via Jupyter Notebook

We have provided a ready-made Jupyter Notebook located at:
📁 **`notebooks/train_on_kaggle.ipynb`**

### Steps to Run:
1. **Create Notebook on Kaggle**: Go to Kaggle, click **"New Notebook"**.
2. **Import or Copy the Notebook**:
   - You can copy-paste the contents of [`notebooks/train_on_kaggle.ipynb`](notebooks/train_on_kaggle.ipynb) into your Kaggle cells.
   - Alternatively, upload the `.ipynb` file directly.
3. **Mount the PulseDB Dataset**:
   - In the right-hand panel, click **"+ Add Data"**.
   - Search for `yesellee/mimiciii-pulsedb`.
   - Click the **"+"** button to add the **"MIMICIII-PulseDB"** database (around 5,300 patient files).
4. **Enable Internet Access**:
   - In the right-hand panel, make sure **"Internet on"** is enabled under the settings. (This is required to pull the Git repository and download packages).
5. **Select GPU**:
   - In the settings, set your **Accelerator** to **GPU T4 x2** or **GPU P100** for fast CUDA-accelerated training.

---

## 📂 Multi-Environment Path Design

The paths in `src/models/config.py` auto-detect the environment:
- **Local Machine**: Reads standard paths (`data/raw`, `data/processed`, `checkpoints/`).
- **Kaggle**: 
  - Reads the input waveform databases from `/kaggle/input/mimiciii-pulsedb`.
  - Outputs cached preprocessed files, trained model checkpoints (`best_model.keras`), and logs into the writable `/kaggle/working/checkpoints/` directory.

---

## 🏃‍♀️ Running Training

Once in the notebook, you can run training using the synced `uv` environment.

### 1. Initialize `uv` and Sync Dependencies:
```bash
# Install uv package manager
pip install uv

# Sync environment
uv venv --python 3.11
uv sync
```

### 2. Choose Your Training Mode:

#### **Option A: Attention Model (In-Memory)**
Loads the matching dataset subsets into RAM. Best for local development and fast prototypes.
```bash
uv run src/models/train_attention.py
```

#### **Option B: Lazy Loading Model (Memory-Efficient)**
Utilizes batch generators to load patient waveform files dynamically from Kaggle's disk. Highly recommended for processing the full scale of 5,361 patient records.
```bash
uv run src/models/train_lazy.py
```

---

## 💾 Saving Your Results

Any checkpoints, evaluation plots, and files will save under `/kaggle/working/checkpoints/`. Once training finishes, you can download them directly from the Kaggle dashboard, or commit the notebook structure to save your run output.
