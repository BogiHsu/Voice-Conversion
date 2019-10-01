# Voice-Conversion-PyTorch

## Colab Demo
Colab demo is provided [here](https://drive.google.com/open?id=1lXijwiNkn5dvzYIDuqPZbyzJyUatQQtY).

## Preprocess
```bash
python preprocess/make_dataset_vctk.py vctk.h5
python preprocess/make_single_samples.py vctk.h5 index.json
```

## Training
```bash
python main.py
```

