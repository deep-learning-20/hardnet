#!/bin/bash

RUNPATH="$( cd "$(dirname "$0")" ; pwd -P )/.."
DATASETS="$RUNPATH/data/sets"
DATALOGS="$RUNPATH/data/sets_megadepth_RGB_STN_64"

mkdir -p "$DATASETS"
mkdir -p "$DATALOGS"


( # Download and prepare data
    cd "$DATASETS"
    if [ ! -d "wxbs-descriptors-benchmark/data/W1BS" ]; then
        git clone https://github.com/ducha-aiki/wxbs-descriptors-benchmark.git
        chmod +x wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
        ./wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
        mv W1BS wxbs-descriptors-benchmark/data/
        rm -f W1BS*.tar.gz
    fi
)

( # Run the code
    cd "$RUNPATH"
    python ./code/HardNet_MegaDepth_validloss_STN_64.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --imageSize=64 --batch-size=256 --epochs=1000 --test-batch-size=256 --log-dir=data/logsmegadepth_RGB_STN_64 --model-dir=data/models_megadepth_RGB_STN_64/ --experiment-name=megadepth_RGB_STN_64/ $@ | tee -a "$DATALOGS/log_HardNet_MegaDepth_Lib.log"
)




