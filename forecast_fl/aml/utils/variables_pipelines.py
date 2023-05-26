#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

# PIPELINE_DESCRIPTION
MODEL_TRAINING_PIPELINE_DESCRIPTION = "Pipeline to train models at meta_ub level"
MODEL_TRAINING_PIPELINE_BASENAME = "Train models pipeline"
MODEL_TRAINING_PIPELINE_ENDPOINT_NAME = "FL360_TRAIN_META_UB_ENDPOINT"

UB_MODEL_TRAINING_PIPELINE_DESCRIPTION = "Pipeline to train models at ub level"
UB_MODEL_TRAINING_PIPELINE_BASENAME = "Train UB level pipeline"
UB_MODEL_TRAINING_PIPELINE_ENDPOINT_NAME = "FL360_TRAIN_UB_ENDPOINT"

MODEL_PREDICTION_PIPELINE_DESCRIPTION = (
    "Pipeline to predict from meta_ub & ub models depending on date min & max setted up"
)
MODEL_PREDICTION_PIPELINE_BASENAME = "Prediction UB level pipeline"
MODEL_PREDICTION_PIPELINE_ENDPOINT_NAME = "FL360_PREDICTION_ENDPOINT"

# STORAGE
DATA_INPUTS = "./outil-predictions/data_inputs"
DATA_INPUTS_FL360 = "./FL360"
PIPELINE_OUTPUTS = "./outil-predictions/pipeline_outputs"

# Azure ML
AML_CPU_CLUSTER_NAME = "cpu-compute-upgraded"
AML_ALLOW_REUSE = False
AML_MAX_DURATION = 9200  # In seconds
