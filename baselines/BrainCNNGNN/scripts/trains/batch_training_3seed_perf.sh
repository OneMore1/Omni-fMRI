#!/bin/bash
###
 # @Author: ViolinSolo
 # @Date: 2025-11-06 11:10:58
 # @LastEditTime: 2025-11-27 19:21:40
 # @LastEditors: ViolinSolo
 # @Description: 
 # @FilePath: /ProjectBrainBaseline/scripts/trains/batch_training_3seed_perf.sh
### 

train_configs_best_perf=(
    "abide,configs/braingnn_classification.yaml,64,0.01,0.05"
    "abide,configs/brainnetcnn_classification.yaml,64,0.0001,0.0005"
    "abide,configs/brainnetcnn_regression.yaml,64,0.0005,0.05"
    "abide,configs/braingnn_regression.yaml,64,0.01,0.05"
    "abide,configs/brainnetcnn_classification_age_cls.yaml,64,0.005,0.005"
    "abide,configs/braingnn_classification_age_cls.yaml,64,0.01,0.005"

    "adhd,configs/braingnn_classification.yaml,64,0.01,0.005"
    "adhd,configs/brainnetcnn_classification.yaml,64,0.0001,0.005"

    "adni,configs/braingnn_classification_adni_ad.yaml,64,0.005,0.0005"
    "adni,configs/brainnetcnn_classification_adni_ad.yaml,64,0.0005,0.05"

    "adni,configs/braingnn_classification_adni_mci.yaml,64,0.001,0.005"
    "adni,configs/brainnetcnn_classification_adni_mci.yaml,64,0.0005,0.005"

    "ppmi,configs/braingnn_classification_ppmi.yaml,64,0.01,0.5"
    "ppmi,configs/brainnetcnn_classification_ppmi.yaml,64,0.0001,0.0005"
)

# Array of upstream models and their corresponding number of ROIs
upstream_models=(
    # "jepa,450"
    # "lm,424"
    "mass,100"
)


seeds=(42 0 999)

# Loop through all combinations and run training
for config in "${train_configs_best_perf[@]}"; do
    IFS=',' read -r dataset config_path bs lr wd <<< "$config"
    for model_pair in "${upstream_models[@]}"; do
        IFS=',' read -r model n_rois <<< "$model_pair"
        for seed in "${seeds[@]}"; do
            echo "Training with config: $config_path, dataset: $dataset, upstream_model: $model, n_rois: $n_rois, batch_size: $bs, learning_rate: $lr, weight_decay: $wd, seed: $seed"
            python train.py --config "$config_path" \
                --downstream_dataset "$dataset" \
                --upstream_model "$model" \
                --num_rois "$n_rois" \
                --batch_size "$bs" \
                --lr "$lr" \
                --wd "$wd" \
                --seed "$seed"
        done
    done
done    
