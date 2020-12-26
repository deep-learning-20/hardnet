#!/bin/bash

RUNPATH="$( cd "$(dirname "$0")" ; pwd -P )/.."
DATASETS="$RUNPATH/data/sets"
DATALOGS="$RUNPATH/data/sets_megadepth_RGB_64res"

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
    python ./code/HardNet_MegaDepth_validloss.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --log-dir=data/logsmegadepth_RGB_64res --model-dir=data/models_megadepth_RGB_64res/ --experiment-name=megadepth_RGB_64res/ $@ | tee -a "$DATALOGS/log_HardNet_MegaDepth_Lib.log"
)




