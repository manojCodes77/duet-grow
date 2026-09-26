# Kaggle setup

## Kaggle notebook

Add the dataset to the notebook instead of downloading it repeatedly. Dataset files are normally available under `/kaggle/input/...`.

## Local Kaggle CLI

```bash
pip install kaggle
kaggle datasets download <owner>/<dataset> -p data/imagenet --unzip
```

The current Kaggle CLI supports `kaggle datasets download <DATASET>`, `--path`, and `--unzip`.

For ImageNet, use a dataset/mirror for which your access and the dataset's licence/competition rules permit the experiment. A Kaggle ImageNet object-localization dataset is extremely large, so full ImageNet should be a later experiment, not the first debugging target.
