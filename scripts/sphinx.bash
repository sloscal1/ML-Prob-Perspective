#!/usr/bin/env bash

. ~/miniconda3/etc/profile.d/conda.sh
conda activate ./ml
cd docs
make html
conda deactivate