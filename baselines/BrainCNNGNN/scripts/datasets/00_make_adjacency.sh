#!/bin/bash
###
python scripts/datasets/make_adjacency.py --src data/cvpr_dataset/abide_ROIxtime_trans/mass --dest data/abide_adj/mass --transpose --time_length 150 --workers 12 --save_format npy --overwrite
python scripts/datasets/make_adjacency.py --src data/cvpr_dataset/abide_ROIxtime_trans/jepa --dest data/abide_adj/jepa --transpose --time_length 160 --workers 12 --save_format npy --overwrite
python scripts/datasets/make_adjacency.py --src data/cvpr_dataset/abide_ROIxtime_trans/lm   --dest data/abide_adj/lm   --transpose --time_length 200 --workers 12 --save_format npy --overwrite
python scripts/datasets/make_adjacency.py --src data/cvpr_dataset/adhd_ROIxtime_trans/mass  --dest data/adhd_adj/mass  --transpose --time_length 150 --workers 12 --save_format npy --overwrite
python scripts/datasets/make_adjacency.py --src data/cvpr_dataset/adhd_ROIxtime_trans/jepa  --dest data/adhd_adj/jepa  --transpose --time_length 160 --workers 12 --save_format npy --overwrite
python scripts/datasets/make_adjacency.py --src data/cvpr_dataset/adhd_ROIxtime_trans/lm    --dest data/adhd_adj/lm    --transpose --time_length 200 --workers 12 --save_format npy --overwrite
python scripts/datasets/make_adjacency.py --src data/cvpr_dataset/ppmi_tr072/mass           --dest data/ppmi_adj/mass  --transpose --time_length 150 --workers 12 --save_format npy --overwrite
python scripts/datasets/make_adjacency.py --src data/cvpr_dataset/ppmi_tr072/jepa           --dest data/ppmi_adj/jepa  --transpose --time_length 160 --workers 12 --save_format npy --overwrite
python scripts/datasets/make_adjacency.py --src data/cvpr_dataset/ppmi_tr072/lm             --dest data/ppmi_adj/lm    --transpose --time_length 200 --workers 12 --save_format npy --overwrite

adni_sub_dirs=('ad' 'cn' 'mci')
for sub_dir in "${adni_sub_dirs[@]}"; do
    python scripts/datasets/make_adjacency.py --src data/cvpr_dataset/adni_tr072/mass/"$sub_dir"                    --dest data/adni_adj/mass/"$sub_dir" --transpose --time_length 150 --workers 12 --save_format npy --overwrite
    python scripts/datasets/make_adjacency.py --src data/cvpr_dataset/adni_tr072/jepa/"$sub_dir"/combined_jepa_450  --dest data/adni_adj/jepa/"$sub_dir" --transpose --time_length 160 --workers 12 --save_format npy --overwrite
    python scripts/datasets/make_adjacency.py --src data/cvpr_dataset/adni_tr072/lm/"$sub_dir"                      --dest data/adni_adj/lm/"$sub_dir"   --transpose --time_length 200 --workers 12 --save_format npy --overwrite
done
