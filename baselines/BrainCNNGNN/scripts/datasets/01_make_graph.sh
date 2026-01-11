#!/bin/bash
###
 # @Author: ViolinSolo
 # @Date: 2025-11-05 13:15:39
 # @LastEditTime: 2025-11-11 19:06:09
 # @LastEditors: ViolinSolo
 # @Description: 
 # @FilePath: /ProjectBrainBaseline/scripts/datasets/01_make_graph.sh
### 

target_datasets=(
    # ADHD
    # "data/cvpr_dataset/baseline_val_ADHD_npy/mass data/cvpr_dataset/fmri_ROI/adhd/mass/adhd_need,data/adhd_graph/mass/,100"
    # "data/cvpr_dataset/baseline_val_ADHD_npy/jepa data/cvpr_dataset/fmri_ROI/adhd/jepa/combined_jepa_450,data/adhd_graph/jepa/,450"
    # "data/cvpr_dataset/baseline_val_ADHD_npy/lm data/cvpr_dataset/fmri_ROI/adhd/lm/adhd_need,data/adhd_graph/lm/,424"
    # ADHD
    "data/cvpr_dataset/adhd_ROIxtime_trans/mass,data/adhd_graph/mass/,100"
    "data/cvpr_dataset/adhd_ROIxtime_trans/jepa,data/adhd_graph/jepa/,450"
    "data/cvpr_dataset/adhd_ROIxtime_trans/lm,data/adhd_graph/lm/,424"
    # abide
    "data/cvpr_dataset/abide_ROIxtime_trans/mass,data/abide_graph/mass/,100"
    "data/cvpr_dataset/abide_ROIxtime_trans/jepa,data/abide_graph/jepa/,450"
    "data/cvpr_dataset/abide_ROIxtime_trans/lm,data/abide_graph/lm/,424"
    # ppmi
    "data/cvpr_dataset/ppmi_tr072/mass,data/ppmi_graph/mass/,100"
    "data/cvpr_dataset/ppmi_tr072/jepa/combined_jepa_450,data/ppmi_graph/jepa/,450"
    "data/cvpr_dataset/ppmi_tr072/lm,data/ppmi_graph/lm/,424"
    # adni
    "data/cvpr_dataset/adni_tr072/mass/ad,data/adni_graph/mass/ad/,100"
    "data/cvpr_dataset/adni_tr072/mass/cn,data/adni_graph/mass/cn/,100"
    "data/cvpr_dataset/adni_tr072/mass/mci,data/adni_graph/mass/mci/,100"
    "data/cvpr_dataset/adni_tr072/jepa/ad/combined_jepa_450,data/adni_graph/jepa/ad/,450"
    "data/cvpr_dataset/adni_tr072/jepa/cn/combined_jepa_450,data/adni_graph/jepa/cn/,450"
    "data/cvpr_dataset/adni_tr072/jepa/mci/combined_jepa_450,data/adni_graph/jepa/mci/,450"
    "data/cvpr_dataset/adni_tr072/lm/ad,data/adni_graph/lm/ad/,424"
    "data/cvpr_dataset/adni_tr072/lm/cn,data/adni_graph/lm/cn/,424"
    "data/cvpr_dataset/adni_tr072/lm/mci,data/adni_graph/lm/mci/,424"
)


for dataset_pair in "${target_datasets[@]}"; do
    IFS=',' read -r input_path output_path n_rois <<< "$dataset_pair"
    echo "Processing dataset from $input_path to $output_path"
    # if input_path contains multiple directories separated by space, use --input_dirs
    if [[ "$input_path" == *" "* ]]; then
        IFS=' ' read -r -a input_dirs_array <<< "$input_path"
        python scripts/datasets/make_graph.py --input_dirs "${input_dirs_array[@]}" --output_dir "$output_path" --edge_method correlation --top_percent 10 --n_rois "$n_rois"
        python scripts/datasets/make_graph.py --input_dirs "${input_dirs_array[@]}" --output_dir "$output_path" --edge_method partial --top_percent 10 --n_rois "$n_rois"
        continue
    fi
    python scripts/datasets/make_graph.py --input_dir "$input_path" --output_dir "$output_path" --edge_method correlation --top_percent 10 --n_rois "$n_rois"
    python scripts/datasets/make_graph.py --input_dir "$input_path" --output_dir "$output_path" --edge_method partial --top_percent 10 --n_rois "$n_rois"
done