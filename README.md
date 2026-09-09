# BigEarthNet Multimodal Retrieval

## Week 3 Work Completed

Implemented and validated the Two-Tower multimodal retrieval foundation:

- Added separate ResNet-50 encoders for Sentinel-1 SAR and Sentinel-2 multispectral images.
- Added 512-dimensional projection heads for both encoders.
- Implemented symmetric in-batch InfoNCE contrastive loss.
- Added paired dataset, dataloader, configuration, and preprocessing modules.
- Added a PyTorch Lightning training entry point with validation, checkpointing, CSV logging, and gradient clipping.
- Added deterministic 19-class BigEarthNet label encoding.
- Verified paired batch collation and tensor interfaces.
- Tested FAISS `IndexFlatL2` retrieval.
- Evaluated baseline retrieval on 4,500 validation patches.

## Generated and Verified Results

- Baseline Sentinel-2 embeddings: `(4500, 2048)`
- Baseline Sentinel-1 embeddings: `(4500, 2048)`
- Baseline retrieval mAP@10: `0.966140` (`96.614%`)
- FAISS retrieval test: passed
- Two-Tower random-data sanity test: passed
- Dataset label encoder test: passed
- Source compilation check: passed

## Main Files

- `src/models/two_tower.py` - Two-Tower model and InfoNCE loss
- `src/dataset/dataset.py` - Dataset and label encoding
- `src/dataset/dataloader.py` - Paired train and validation loaders
- `src/config/settings.py` - Shared paths and settings
- `src/preprocessing/sar_preprocessing.py` - SAR preprocessing
- `src/preprocessing/preprocess_s2.py` - Sentinel-2 preprocessing
- `scripts/train_two_tower.py` - Sanity test and real-data training entry point
- `outputs/metadata/train_split.csv` - Training split metadata
- `outputs/metadata/val_split.csv` - Validation split metadata
- `outputs/metadata/test_split.csv` - Test split metadata

## Validation Commands

Run the model sanity test:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\train_two_tower.py --samples 8 --batch-size 4 --epochs 1
```

Run real-data training after the complete paired datasets are available:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\train_two_tower.py --real-data --epochs 20 --batch-size 64
```

## Week 4 Requirement

Real contrastive training and trained embeddings require the complete paired SAR and Sentinel-2 datasets. The current local checkout does not contain the full `data/s1_patches/` and `data/s2_patches/` directories, so `best_model.ckpt` and trained validation/test embeddings have not yet been generated.

The Week 3 source-code integration was pushed to GitHub in commit `43eb162`.
