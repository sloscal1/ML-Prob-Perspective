#!/usr/bin/env bash

. ~/miniconda3/etc/profile.d/conda.sh
conda activate ./ml
cd docs
# New files in the module will not get processed without adding them to the module rst file first
sphinx-apidoc -f -o source/ ../chapter
make html
conda deactivate