#!/usr/bin/env bash

# Initialise the variables.
LAMBDA=0.5
ROOTRESULTSDIR=$(python -m utils.get_results_directory)
RESULTSDIR="$ROOTRESULTSDIR/$LAMBDA"
FEATUREMODELPATH="$RESULTSDIR/feature_model/model.pt"

echo "Lambda: $LAMBDA"
echo "Results Directory (Root)  : $ROOTRESULTSDIR"
echo "Results Directory (Lambda): $RESULTSDIR"
echo "Feature Model Path: $FEATUREMODELPATH"

python -m features_train with "results_directory=$RESULTSDIR" "lambda_value=$LAMBDA"