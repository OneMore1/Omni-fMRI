python scripts/datasets/match_splits_with_npys.py \
    --npy_dir data/abide_graph/jepa \
    --train_split data/cvpr_dataset/split/abide_train.txt \
    --val_split data/cvpr_dataset/split/abide_val.txt \
    --test_split data/cvpr_dataset/split/abide_test.txt \
    --label data/abide_graph/abide_v3_dx_group_age_norm.csv \
    --saved_label_cols SUB_ID DX_GROUP AGE_NORM age_group_4 \
    --saved_unique_prefix 00_jepa_

python scripts/datasets/match_splits_with_npys.py \
    --npy_dir data/abide_graph/lm \
    --train_split data/cvpr_dataset/split/abide_train.txt \
    --val_split data/cvpr_dataset/split/abide_val.txt \
    --test_split data/cvpr_dataset/split/abide_test.txt \
    --label data/abide_graph/abide_v3_dx_group_age_norm.csv \
    --saved_label_cols SUB_ID DX_GROUP AGE_NORM age_group_4 \
    --saved_unique_prefix 00_lm_

python scripts/datasets/match_splits_with_npys.py \
    --npy_dir data/abide_graph/mass \
    --train_split data/cvpr_dataset/split/abide_train.txt \
    --val_split data/cvpr_dataset/split/abide_val.txt \
    --test_split data/cvpr_dataset/split/abide_test.txt \
    --label data/abide_graph/abide_v3_dx_group_age_norm.csv \
    --saved_label_cols SUB_ID DX_GROUP AGE_NORM age_group_4 \
    --saved_unique_prefix 00_mass_


python scripts/datasets/match_splits_with_npys.py \
    --dataset adhd \
    --npy_dir data/adhd_graph/jepa \
    --train_split data/cvpr_dataset/split/adhd_train.txt \
    --val_split data/cvpr_dataset/split/adhd_val.txt \
    --test_split data/cvpr_dataset/split/adhd_test.txt \
    --label data/adhd_graph/adhd_v2_dx_group.csv \
    --saved_label_cols SUB_ID DX_GROUP \
    --saved_unique_prefix 00_jepa_

python scripts/datasets/match_splits_with_npys.py \
    --dataset adhd \
    --npy_dir data/adhd_graph/lm \
    --train_split data/cvpr_dataset/split/adhd_train.txt \
    --val_split data/cvpr_dataset/split/adhd_val.txt \
    --test_split data/cvpr_dataset/split/adhd_test.txt \
    --label data/adhd_graph/adhd_v2_dx_group.csv \
    --saved_label_cols SUB_ID DX_GROUP \
    --saved_unique_prefix 00_lm_

python scripts/datasets/match_splits_with_npys.py \
    --dataset adhd \
    --npy_dir data/adhd_graph/mass \
    --train_split data/cvpr_dataset/split/adhd_train.txt \
    --val_split data/cvpr_dataset/split/adhd_val.txt \
    --test_split data/cvpr_dataset/split/adhd_test.txt \
    --label data/adhd_graph/adhd_v2_dx_group.csv \
    --saved_label_cols SUB_ID DX_GROUP \
    --saved_unique_prefix 00_mass_



python scripts/datasets/match_splits_with_npys.py \
    --npy_dir data/abide_adj/jepa \
    --train_split data/cvpr_dataset/split/abide_train.txt \
    --val_split data/cvpr_dataset/split/abide_val.txt \
    --test_split data/cvpr_dataset/split/abide_test.txt \
    --label data/abide_adj/abide_v3_dx_group_age_norm.csv \
    --saved_label_cols SUB_ID DX_GROUP AGE_NORM age_group_4 \
    --saved_unique_prefix 00_jepa_

python scripts/datasets/match_splits_with_npys.py \
    --npy_dir data/abide_adj/lm \
    --train_split data/cvpr_dataset/split/abide_train.txt \
    --val_split data/cvpr_dataset/split/abide_val.txt \
    --test_split data/cvpr_dataset/split/abide_test.txt \
    --label data/abide_adj/abide_v3_dx_group_age_norm.csv \
    --saved_label_cols SUB_ID DX_GROUP AGE_NORM age_group_4 \
    --saved_unique_prefix 00_lm_
    
python scripts/datasets/match_splits_with_npys.py \
    --npy_dir data/abide_adj/mass \
    --train_split data/cvpr_dataset/split/abide_train.txt \
    --val_split data/cvpr_dataset/split/abide_val.txt \
    --test_split data/cvpr_dataset/split/abide_test.txt \
    --label data/abide_adj/abide_v3_dx_group_age_norm.csv \
    --saved_label_cols SUB_ID DX_GROUP AGE_NORM age_group_4 \
    --saved_unique_prefix 00_mass_


python scripts/datasets/match_splits_with_npys.py \
    --dataset adhd \
    --npy_dir data/adhd_adj/jepa \
    --train_split data/cvpr_dataset/split/adhd_train.txt \
    --val_split data/cvpr_dataset/split/adhd_val.txt \
    --test_split data/cvpr_dataset/split/adhd_test.txt \
    --label data/adhd_adj/adhd_v2_dx_group.csv \
    --saved_label_cols SUB_ID DX_GROUP \
    --saved_unique_prefix 00_jepa_

python scripts/datasets/match_splits_with_npys.py \
    --dataset adhd \
    --npy_dir data/adhd_adj/lm \
    --train_split data/cvpr_dataset/split/adhd_train.txt \
    --val_split data/cvpr_dataset/split/adhd_val.txt \
    --test_split data/cvpr_dataset/split/adhd_test.txt \
    --label data/adhd_adj/adhd_v2_dx_group.csv \
    --saved_label_cols SUB_ID DX_GROUP \
    --saved_unique_prefix 00_lm_

python scripts/datasets/match_splits_with_npys.py \
    --dataset adhd \
    --npy_dir data/adhd_adj/mass \
    --train_split data/cvpr_dataset/split/adhd_train.txt \
    --val_split data/cvpr_dataset/split/adhd_val.txt \
    --test_split data/cvpr_dataset/split/adhd_test.txt \
    --label data/adhd_adj/adhd_v2_dx_group.csv \
    --saved_label_cols SUB_ID DX_GROUP \
    --saved_unique_prefix 00_mass_


# ----------- original fMRI npys, not FC, used for harmony fMRI --------------
python scripts/datasets/match_splits_with_npys.py \
    --npy_dir data/cvpr_dataset/abide_ROIxtime_trans/jepa \
    --train_split data/cvpr_dataset/split/abide_train.txt \
    --val_split data/cvpr_dataset/split/abide_val.txt \
    --test_split data/cvpr_dataset/split/abide_test.txt \
    --label data/abide_adj/abide_v3_dx_group_age_norm.csv \
    --saved_label_cols SUB_ID DX_GROUP AGE_NORM age_group_4 \
    --saved_unique_prefix 00_jepa_

python scripts/datasets/match_splits_with_npys.py \
    --npy_dir data/cvpr_dataset/abide_ROIxtime_trans/lm \
    --train_split data/cvpr_dataset/split/abide_train.txt \
    --val_split data/cvpr_dataset/split/abide_val.txt \
    --test_split data/cvpr_dataset/split/abide_test.txt \
    --label data/abide_adj/abide_v3_dx_group_age_norm.csv \
    --saved_label_cols SUB_ID DX_GROUP AGE_NORM age_group_4 \
    --saved_unique_prefix 00_lm_
    
python scripts/datasets/match_splits_with_npys.py \
    --npy_dir data/cvpr_dataset/abide_ROIxtime_trans/mass \
    --train_split data/cvpr_dataset/split/abide_train.txt \
    --val_split data/cvpr_dataset/split/abide_val.txt \
    --test_split data/cvpr_dataset/split/abide_test.txt \
    --label data/abide_adj/abide_v3_dx_group_age_norm.csv \
    --saved_label_cols SUB_ID DX_GROUP AGE_NORM age_group_4 \
    --saved_unique_prefix 00_mass_


python scripts/datasets/match_splits_with_npys.py \
    --dataset adhd \
    --npy_dir data/cvpr_dataset/adhd_ROIxtime_trans/jepa \
    --train_split data/cvpr_dataset/split/adhd_train.txt \
    --val_split data/cvpr_dataset/split/adhd_val.txt \
    --test_split data/cvpr_dataset/split/adhd_test.txt \
    --label data/adhd_adj/adhd_v2_dx_group.csv \
    --saved_label_cols SUB_ID DX_GROUP \
    --saved_unique_prefix 00_jepa_

python scripts/datasets/match_splits_with_npys.py \
    --dataset adhd \
    --npy_dir data/cvpr_dataset/adhd_ROIxtime_trans/lm \
    --train_split data/cvpr_dataset/split/adhd_train.txt \
    --val_split data/cvpr_dataset/split/adhd_val.txt \
    --test_split data/cvpr_dataset/split/adhd_test.txt \
    --label data/adhd_adj/adhd_v2_dx_group.csv \
    --saved_label_cols SUB_ID DX_GROUP \
    --saved_unique_prefix 00_lm_
    
python scripts/datasets/match_splits_with_npys.py \
    --dataset adhd \
    --npy_dir data/cvpr_dataset/adhd_ROIxtime_trans/mass \
    --train_split data/cvpr_dataset/split/adhd_train.txt \
    --val_split data/cvpr_dataset/split/adhd_val.txt \
    --test_split data/cvpr_dataset/split/adhd_test.txt \
    --label data/adhd_adj/adhd_v2_dx_group.csv \
    --saved_label_cols SUB_ID DX_GROUP \
    --saved_unique_prefix 00_mass_



python scripts/datasets/match_splits_with_npys.py \
    --dataset adni \
    --npy_dir data/adni_adj/jepa/ad data/adni_adj/jepa/cn \
    --train_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_train.txt \
    --val_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_val.txt \
    --test_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_test.txt \
    --label data/adni_adj/adni_v2_AD_vs_CN_labels.csv \
    --saved_label_cols SUB_ID type AD_vs_CN_label \
    --saved_unique_prefix 00_jepa_AD_

python scripts/datasets/match_splits_with_npys.py \
    --dataset adni \
    --npy_dir data/adni_adj/lm/ad data/adni_adj/lm/cn \
    --train_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_train.txt \
    --val_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_val.txt \
    --test_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_test.txt \
    --label data/adni_adj/adni_v2_AD_vs_CN_labels.csv \
    --saved_label_cols SUB_ID type AD_vs_CN_label \
    --saved_unique_prefix 00_lm_AD_
    
python scripts/datasets/match_splits_with_npys.py \
    --dataset adni \
    --npy_dir data/adni_adj/mass/ad data/adni_adj/mass/cn \
    --train_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_train.txt \
    --val_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_val.txt \
    --test_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_test.txt \
    --label data/adni_adj/adni_v2_AD_vs_CN_labels.csv \
    --saved_label_cols SUB_ID type AD_vs_CN_label \
    --saved_unique_prefix 00_mass_AD_


python scripts/datasets/match_splits_with_npys.py \
    --dataset adni \
    --npy_dir data/adni_adj/jepa/mci data/adni_adj/jepa/cn \
    --train_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_train.txt \
    --val_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_val.txt \
    --test_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_test.txt \
    --label data/adni_adj/adni_v2_MCI_vs_CN_labels.csv \
    --saved_label_cols SUB_ID type MCI_vs_CN_label \
    --saved_unique_prefix 00_jepa_MCI_

python scripts/datasets/match_splits_with_npys.py \
    --dataset adni \
    --npy_dir data/adni_adj/lm/mci data/adni_adj/lm/cn \
    --train_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_train.txt \
    --val_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_val.txt \
    --test_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_test.txt \
    --label data/adni_adj/adni_v2_MCI_vs_CN_labels.csv \
    --saved_label_cols SUB_ID type MCI_vs_CN_label \
    --saved_unique_prefix 00_lm_MCI_
    
python scripts/datasets/match_splits_with_npys.py \
    --dataset adni \
    --npy_dir data/adni_adj/mass/mci data/adni_adj/mass/cn \
    --train_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_train.txt \
    --val_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_val.txt \
    --test_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_test.txt \
    --label data/adni_adj/adni_v2_MCI_vs_CN_labels.csv \
    --saved_label_cols SUB_ID type MCI_vs_CN_label \
    --saved_unique_prefix 00_mass_MCI_



python scripts/datasets/match_splits_with_npys.py \
    --dataset adni \
    --npy_dir data/adni_graph/jepa/ad data/adni_graph/jepa/cn \
    --train_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_train.txt \
    --val_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_val.txt \
    --test_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_test.txt \
    --label data/adni_adj/adni_v2_AD_vs_CN_labels.csv \
    --saved_label_cols SUB_ID type AD_vs_CN_label \
    --saved_unique_prefix 00_jepa_AD_

python scripts/datasets/match_splits_with_npys.py \
    --dataset adni \
    --npy_dir data/adni_graph/lm/ad data/adni_graph/lm/cn \
    --train_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_train.txt \
    --val_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_val.txt \
    --test_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_test.txt \
    --label data/adni_adj/adni_v2_AD_vs_CN_labels.csv \
    --saved_label_cols SUB_ID type AD_vs_CN_label \
    --saved_unique_prefix 00_lm_AD_
    
python scripts/datasets/match_splits_with_npys.py \
    --dataset adni \
    --npy_dir data/adni_graph/mass/ad data/adni_graph/mass/cn \
    --train_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_train.txt \
    --val_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_val.txt \
    --test_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_test.txt \
    --label data/adni_adj/adni_v2_AD_vs_CN_labels.csv \
    --saved_label_cols SUB_ID type AD_vs_CN_label \
    --saved_unique_prefix 00_mass_AD_


python scripts/datasets/match_splits_with_npys.py \
    --dataset adni \
    --npy_dir data/adni_graph/jepa/mci data/adni_graph/jepa/cn \
    --train_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_train.txt \
    --val_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_val.txt \
    --test_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_test.txt \
    --label data/adni_adj/adni_v2_MCI_vs_CN_labels.csv \
    --saved_label_cols SUB_ID type MCI_vs_CN_label \
    --saved_unique_prefix 00_jepa_MCI_

python scripts/datasets/match_splits_with_npys.py \
    --dataset adni \
    --npy_dir data/adni_graph/lm/mci data/adni_graph/lm/cn \
    --train_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_train.txt \
    --val_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_val.txt \
    --test_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_test.txt \
    --label data/adni_adj/adni_v2_MCI_vs_CN_labels.csv \
    --saved_label_cols SUB_ID type MCI_vs_CN_label \
    --saved_unique_prefix 00_lm_MCI_
    
python scripts/datasets/match_splits_with_npys.py \
    --dataset adni \
    --npy_dir data/adni_graph/mass/mci data/adni_graph/mass/cn \
    --train_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_train.txt \
    --val_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_val.txt \
    --test_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_test.txt \
    --label data/adni_adj/adni_v2_MCI_vs_CN_labels.csv \
    --saved_label_cols SUB_ID type MCI_vs_CN_label \
    --saved_unique_prefix 00_mass_MCI_


python scripts/datasets/match_splits_with_npys.py \
    --dataset ppmi \
    --npy_dir data/cvpr_dataset/ppmi_tr072/jepa/combined_jepa_450 \
    --train_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_train.txt \
    --val_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_val.txt \
    --test_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_test.txt \
    --label data/ppmi_adj/ppmi_v2_fulllist.csv \
    --saved_label_cols SUB_ID DX DX_GROUP \
    --saved_unique_prefix 00_jepa_

python scripts/datasets/match_splits_with_npys.py \
    --dataset ppmi \
    --npy_dir data/cvpr_dataset/ppmi_tr072/lm \
    --train_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_train.txt \
    --val_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_val.txt \
    --test_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_test.txt \
    --label data/ppmi_adj/ppmi_v2_fulllist.csv \
    --saved_label_cols SUB_ID DX DX_GROUP \
    --saved_unique_prefix 00_lm_
    
python scripts/datasets/match_splits_with_npys.py \
    --dataset ppmi \
    --npy_dir data/cvpr_dataset/ppmi_tr072/mass \
    --train_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_train.txt \
    --val_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_val.txt \
    --test_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_test.txt \
    --label data/ppmi_adj/ppmi_v2_fulllist.csv \
    --saved_label_cols SUB_ID DX DX_GROUP \
    --saved_unique_prefix 00_mass_


python scripts/datasets/match_splits_with_npys.py \
    --dataset ppmi \
    --npy_dir data/ppmi_adj/jepa/combined_jepa_450 \
    --train_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_train.txt \
    --val_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_val.txt \
    --test_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_test.txt \
    --label data/ppmi_adj/ppmi_v2_fulllist.csv \
    --saved_label_cols SUB_ID DX DX_GROUP \
    --saved_unique_prefix 00_jepa_

python scripts/datasets/match_splits_with_npys.py \
    --dataset ppmi \
    --npy_dir data/ppmi_adj/lm \
    --train_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_train.txt \
    --val_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_val.txt \
    --test_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_test.txt \
    --label data/ppmi_adj/ppmi_v2_fulllist.csv \
    --saved_label_cols SUB_ID DX DX_GROUP \
    --saved_unique_prefix 00_lm_
    
python scripts/datasets/match_splits_with_npys.py \
    --dataset ppmi \
    --npy_dir data/ppmi_adj/mass \
    --train_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_train.txt \
    --val_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_val.txt \
    --test_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_test.txt \
    --label data/ppmi_adj/ppmi_v2_fulllist.csv \
    --saved_label_cols SUB_ID DX DX_GROUP \
    --saved_unique_prefix 00_mass_


python scripts/datasets/match_splits_with_npys.py \
    --dataset ppmi \
    --npy_dir data/ppmi_graph/jepa \
    --train_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_train.txt \
    --val_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_val.txt \
    --test_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_test.txt \
    --label data/ppmi_adj/ppmi_v2_fulllist.csv \
    --saved_label_cols SUB_ID DX DX_GROUP \
    --saved_unique_prefix 00_jepa_

python scripts/datasets/match_splits_with_npys.py \
    --dataset ppmi \
    --npy_dir data/ppmi_graph/lm \
    --train_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_train.txt \
    --val_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_val.txt \
    --test_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_test.txt \
    --label data/ppmi_adj/ppmi_v2_fulllist.csv \
    --saved_label_cols SUB_ID DX DX_GROUP \
    --saved_unique_prefix 00_lm_
    
python scripts/datasets/match_splits_with_npys.py \
    --dataset ppmi \
    --npy_dir data/ppmi_graph/mass \
    --train_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_train.txt \
    --val_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_val.txt \
    --test_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_test.txt \
    --label data/ppmi_adj/ppmi_v2_fulllist.csv \
    --saved_label_cols SUB_ID DX DX_GROUP \
    --saved_unique_prefix 00_mass_



# ------------- ADNI original fMRI npys, not FC, used for harmony fMRI --------------
python scripts/datasets/match_splits_with_npys.py \
    --dataset adni \
    --npy_dir data/cvpr_dataset/adni_tr072/jepa/mci/combined_jepa_450 data/cvpr_dataset/adni_tr072/jepa/cn/combined_jepa_450 \
    --train_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_train.txt \
    --val_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_val.txt \
    --test_split data/cvpr_dataset/splits_used/adni_split/adni_mci_native_test.txt \
    --label data/adni_adj/adni_v2_MCI_vs_CN_labels.csv \
    --saved_label_cols SUB_ID type MCI_vs_CN_label \
    --saved_unique_prefix 00_jepa_MCI_

python scripts/datasets/match_splits_with_npys.py \
    --dataset adni \
    --npy_dir data/cvpr_dataset/adni_tr072/jepa/ad/combined_jepa_450 data/cvpr_dataset/adni_tr072/jepa/cn/combined_jepa_450 \
    --train_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_train.txt \
    --val_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_val.txt \
    --test_split data/cvpr_dataset/splits_used/adni_split/adni_ad_native_test.txt \
    --label data/adni_adj/adni_v2_AD_vs_CN_labels.csv \
    --saved_label_cols SUB_ID type AD_vs_CN_label \
    --saved_unique_prefix 00_jepa_AD_

# -------------- PPMI original fMRI npys, not FC, used for harmony fMRI --------------
python scripts/datasets/match_splits_with_npys.py \
    --dataset ppmi \
    --npy_dir data/cvpr_dataset/ppmi_tr072/jepa/combined_jepa_450 \
    --train_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_train.txt \
    --val_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_val.txt \
    --test_split data/cvpr_dataset/splits_used/ppmi_split/ppmi_test.txt \
    --label data/ppmi_adj/ppmi_v2_fulllist.csv \
    --saved_label_cols SUB_ID DX DX_GROUP \
    --saved_unique_prefix 00_jepa_
