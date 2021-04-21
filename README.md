# WMT20 Biomedical Translation scripts

The main scripts are present in the `src` directory. In the `dataset` directory few lines of real dataset can be found: this is to have a material for running the scripts. Trained models will be placed in the `logs` directory, with few nested directories, each naming model's parameters.

### Usage examples

**preprocess.py**
```
python preprocess.py ./dataset/train --ratio 3
```

**train.sh**
```
./train.sh -i ./dataset/FSQ/clean
-s en \
-t ru \
--log-dir ./logs \
-vi 1 \
-lr 0.00001 \
-ls 0 \
-pt 500
```

**score.sh**
```
./score.sh -i ./dataset/valid.en-ru \
-s en \
-t ru \
beam 32 \
-m ./logs/en2ru/dataset/FSQ/clean/H/d0.0-lr0.00001-ls0-w4000-bs128-mt1536-up1-cn0.0-vi1
```

**backtranslate.sh**
```
./backtranslate.sh -i ./dataset \
--prefix train \
--source-lang en \
--target-lang ru \
-m ./logs/en2ru/dataset/FSQ/clean/H/d0.0-lr0.00001-ls0-w4000-bs128-mt1536-up1-cn0.0-vi1
```
