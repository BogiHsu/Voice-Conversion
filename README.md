# 2019-AI-Summer-School-Voice-Conversion

## Preprocess
```bash
python preprocess/make_dataset_vctk.py vctk.h5
python preprocess/make_single_samples.py vctk.h5 index.json
```

## Training
```bash
python main.py
```

