# Voice-Conversion
This work is organized from [here](https://github.com/jjery2243542/voice_conversion).

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

