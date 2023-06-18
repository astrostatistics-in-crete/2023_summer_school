#!/usr/bin/env bash -l
git pull origin main
conda activate astrostat23
pip install .
pip install --upgrade nbconvert
jupyter notebook
