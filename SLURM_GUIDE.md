# Slurm Job Submission Guide

This guide explains how to use the Slurm job scripts for training the cuffless blood pressure prediction models on an HPC cluster.

## Available Job Scripts

### 1. `train_job.slurm`
Standard model training script for the physiology-informed CNN model.

**Resources:**
- 1 GPU
- 8 CPU cores
- 32GB RAM
- 12 hour time limit

### 2. `train_attention_job.slurm`
Training script for the more complex attention-based model.

**Resources:**
- 1 GPU
- 8 CPU cores
- 64GB RAM
- 24 hour time limit

## Prerequisites

Before submitting jobs, make sure:

1. **Data is prepared**: 
   - Raw data should be in `$SCRATCH/data/raw/` on HPC systems (or `data/raw/` locally)
   - Processed data should be in `$SCRATCH/data/processed/` on HPC systems (or `data/processed/` locally)
   - The code automatically uses `$SCRATCH` environment variable for HPC storage
2. **Email notifications**: Update the `--mail-user` field in the scripts with your email
3. **Modules are correct**: Verify module names match your cluster's available modules
4. **Logs directory exists**: The scripts will create it, but you can pre-create it:
   ```bash
   mkdir -p logs
   ```

## Using HPC Scratch Storage

The codebase is configured to automatically use `$SCRATCH` environment variable for data storage on HPC systems:

- **Local development**: Data stored in `data/raw/` and `data/processed/`
- **HPC with $SCRATCH**: Data stored in `$SCRATCH/data/raw/` and `$SCRATCH/data/processed/`

### Setting up data on HPC:

```bash
# Your $SCRATCH is typically set automatically, but you can verify:
echo $SCRATCH

# Copy data to scratch (if needed)
cp -r data/raw/* $SCRATCH/data/raw/
cp -r data/processed/* $SCRATCH/data/processed/

# Or download data directly to scratch
cd src/data_loading
python loader.py  # Will use $SCRATCH automatically
```

## Submitting Jobs

### Submit standard model training:
```bash
sbatch train_job.slurm
```

### Submit attention model training:
```bash
sbatch train_attention_job.slurm
```

## Monitoring Jobs

### Check job status:
```bash
squeue -u $USER
```

### View live output:
```bash
tail -f logs/slurm-<JOB_ID>.out
```

### Check job details:
```bash
scontrol show job <JOB_ID>
```

### Cancel a job:
```bash
scancel <JOB_ID>
```

## Customizing Resource Allocation

Adjust these SBATCH directives based on your needs:

### For larger datasets or longer training:
```bash
#SBATCH --time=48:00:00      # Increase time limit
#SBATCH --mem=128G           # Increase memory
```

### For multi-GPU training:
```bash
#SBATCH --gres=gpu:2         # Request 2 GPUs
```

### For specific GPU types:
```bash
#SBATCH --gres=gpu:v100:1    # Request specific GPU model
```

### For priority queues:
```bash
#SBATCH --qos=high           # Use high priority queue
```

## Module Configuration

The scripts assume these modules are available:
- `cuda/11.8`
- `cudnn/8.6`
- `python/3.10`

To check available modules on your cluster:
```bash
module avail cuda
module avail python
```

Update the module load commands in the scripts to match your cluster.

## Output Files

### Standard output/error logs:
- `logs/slurm-<JOB_ID>.out` - Standard output and training logs
- `logs/slurm-<JOB_ID>.err` - Error messages (if any)

### Model checkpoints:
- `checkpoints/best_model.h5` - Best model from standard training
- `checkpoints/physiology_informed_model.h5` - Final physiology-informed model

### Training artifacts:
- Plots and visualizations saved by the training scripts

## Troubleshooting

### Job fails immediately:
- Check that modules exist: `module avail`
- Verify Python path: `which python`
- Check partition availability: `sinfo`

### Out of memory errors:
- Increase `--mem` allocation
- Reduce batch size in `src/models/config.py`

### GPU not detected:
- Verify GPU partition name with `sinfo`
- Check CUDA module is loaded correctly
- Ensure `CUDA_VISIBLE_DEVICES` is set properly

### Import errors:
- Ensure requirements.txt is complete
- Check virtual environment is activated
- Verify working directory is correct

## Interactive Debugging

For debugging, request an interactive GPU session:
```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=2:00:00 --pty bash
```

Then manually run the setup commands:
```bash
module load cuda/11.8 cudnn/8.6 python/3.10
source venv/bin/activate
python src/models/train.py
```

## Best Practices

1. **Test locally first**: Run a quick test on a small subset of data before submitting large jobs
2. **Monitor resources**: Use `sacct` to check resource usage after jobs complete
3. **Use checkpoints**: The training scripts save checkpoints - useful for resuming interrupted jobs
4. **Batch similar jobs**: Use job arrays for hyperparameter tuning
5. **Clean up**: Remove old logs and checkpoints periodically

## Job Arrays for Hyperparameter Tuning

To run multiple training jobs with different parameters, create a job array:

```bash
#SBATCH --array=1-10
```

Then modify the training script to use different configs based on `$SLURM_ARRAY_TASK_ID`.

## Example: Checking Completed Job Performance

```bash
# View resource usage summary
sacct -j <JOB_ID> --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,MaxRSS,MaxVMSize

# View GPU usage (if available)
sacct -j <JOB_ID> --format=JobID,TRESUsageInTot,TRESUsageOutMax
```

## Support

For cluster-specific help:
- Check your HPC documentation
- Contact your cluster support team
- Review Slurm documentation: https://slurm.schedmd.com/
