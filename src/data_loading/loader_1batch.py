import os
import subprocess
from pathlib import Path

# -------------------------------
# Setup directory
# -------------------------------
# Use $SCRATCH environment variable for HPC systems
base_data_dir = os.environ.get('SCRATCH', os.path.join(os.path.dirname(__file__), '../..'))
base_dir = Path(base_data_dir) / 'data' / 'raw'
base_dir.mkdir(parents=True, exist_ok=True)
os.chdir(base_dir)

# -------------------------------
# ONE module per dataset
# -------------------------------
mimic_url = "https://rutgers.box.com/shared/static/wmzndowgfa5xi3tvtqahxkld3ngdyjds.016"
vital_url = "https://rutgers.box.com/shared/static/fu0m9tx33jkxywq32shh0g8dg3not15u.010"

# -------------------------------
# Download function
# -------------------------------
def download_and_unzip(url, dataset_name):
    part_name = url.split("/")[-1]             # keeps .016 / .010
    zip_name = f"{dataset_name}.zip"

    # Download
    if not Path(part_name).exists():
        print(f"Downloading {part_name} ...")
        subprocess.run(
            f"curl -L -o {part_name} -C - {url}",
            shell=True,
            check=True
        )
    else:
        print(f"{part_name} already exists, skipping download.")

    # Rename to .zip
    if not Path(zip_name).exists():
        print(f"Renaming {part_name} → {zip_name}")
        os.rename(part_name, zip_name)

    # Unzip
    print(f"Extracting {zip_name} ...")
    subprocess.run(f"unzip -o {zip_name}", shell=True, check=True)

    print(f"{dataset_name} module downloaded and extracted successfully!\n")

# -------------------------------
# Run for both datasets
# -------------------------------
download_and_unzip(mimic_url, "PulseDB_MIMIC")
download_and_unzip(vital_url, "PulseDB_Vital")

print("All selected PulseDB modules processed successfully!")
