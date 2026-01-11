#!/bin/bash
###
 # @Author: ViolinSolo
 # @Date: 2025-11-06 11:10:58
 # @LastEditTime: 2025-11-27 19:18:22
 # @LastEditors: ViolinSolo
 # @Description: 
 # @FilePath: /ProjectBrainBaseline/scripts/trains/batch_training.sh
### 

train_configs=(
    "abide,configs/braingnn_classification.yaml"
    "abide,configs/brainnetcnn_classification.yaml"
    "adhd,configs/braingnn_classification.yaml"
    "adhd,configs/brainnetcnn_classification.yaml"
    "adni,configs/braingnn_classification_adni_ad.yaml"
    "adni,configs/brainnetcnn_classification_adni_ad.yaml"
    "adni,configs/braingnn_classification_adni_mci.yaml"
    "adni,configs/brainnetcnn_classification_adni_mci.yaml"
    "ppmi,configs/braingnn_classification_ppmi.yaml"
    "ppmi,configs/brainnetcnn_classification_ppmi.yaml"
    "abide,configs/brainnetcnn_regression.yaml"
    "abide,configs/braingnn_regression.yaml"
    "abide,configs/braingnn_classification_age_cls.yaml"
    "abide,configs/brainnetcnn_classification_age_cls.yaml"
)

# Array of upstream models and their corresponding number of ROIs
upstream_models=(
    # "jepa,450"
    # "lm,424"
    "mass,100"
)

# bs, lr, wd
hparams=(
    "64,0.01,0.05"
    "64,0.005,0.05"
    "64,0.001,0.05"
    "64,0.0005,0.05"
    "64,0.0001,0.05"
    "64,0.01,0.005"
    "64,0.005,0.005"
    "64,0.001,0.005"
    "64,0.0005,0.005"
    "64,0.0001,0.005"
    "64,0.01,0.0005"
    "64,0.005,0.0005"
    "64,0.001,0.0005"
    "64,0.0005,0.0005"
    "64,0.0001,0.0005"
)

# Loop through all combinations and run training
for config in "${train_configs[@]}"; do
    IFS=',' read -r dataset config_path <<< "$config"
    for model_pair in "${upstream_models[@]}"; do
        IFS=',' read -r model n_rois <<< "$model_pair"
        for param_set in "${hparams[@]}"; do
            IFS=',' read -r bs lr wd <<< "$param_set"
            echo "Training with config: $config_path, dataset: $dataset, upstream_model: $model, n_rois: $n_rois, batch_size: $bs, learning_rate: $lr, weight_decay: $wd"
            python train.py --config "$config_path" \
                --downstream_dataset "$dataset" \
                --upstream_model "$model" \
                --num_rois "$n_rois" \
                --batch_size "$bs" \
                --lr "$lr" \
                --wd "$wd"
        done
    done
done

