#!/bin/bash
# -------
# Runs every assignment component in order and create all output
# directories exist beforehand (if not, please use mkdir), so individual scripts don't fail on a
# missing folder.
#
# Usage:  bash job.sh

echo "Running GMM"
python code/q1/gmm.py
python code/q1/bmm.py
python code/q2/tta.py
python code/q3/authorship_attribution.py

# similarly add commands for running other files

echo "=== All components finished. Check the outputs/ directory. ==="
