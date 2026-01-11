
'''
load all split texts and match it with the csv file, check the unique sids in each split
'''


import os
import argparse
import pandas as pd
from tqdm import tqdm



def load_split_file(split_file):
    """Load a split text file and return a list of identifiers."""
    with open(split_file, 'r') as f:
        lines = f.read().splitlines()
    return lines



def extract_sid_from_filename(filename, file_type='split'):
    """Extract subject ID from the .npy filename."""
    base = os.path.basename(filename)
    # print(f"Debug: Extracting SID from filename: {filename} with file_type: {file_type}")
    if file_type == 'abide':
        # Example filename: Caltech_0051461_func_preproc_0.npy, first find the "func_preproc", then get the part before it 0051461
        func_preproc_index = base.find("_func_preproc")
        if func_preproc_index != -1:
            sid = base[:func_preproc_index].split('_')[-1]
            return sid, int(sid)  # use int to omit leading zeros
    elif file_type == 'abide_split':
        # base name example:
        # Yale_0050628_control_func_preproc_frames_000000-000199.nii.gz  => extract 0050628
        func_preproc_index = base.find("_func_preproc")
        if func_preproc_index != -1:
            sid = base[:func_preproc_index].split('_')[-2]
            return sid, int(sid)
    elif file_type == 'adhd':
        # Example filename: data/adhd_adj/mass/0010001_run-1.npy -> extract 0010001
        run_index = base.find("_run-")
        if run_index != -1:
            sid = base[:run_index].split('_')[-1]
            return sid, int(sid)
    elif file_type == 'adhd_split':
        # base name example:
        # ADHD_NeuroIMAGE_7446626_patient_zscore.nii.gz  => extract 7446626
        # ADHD_Peking_1_9890726_control_zscore.nii.gz => extract 9890726
        sid_idx = -3
        sid = base.split('_')[sid_idx]
        return sid, int(sid)
    return None

dataset = 'adhd'  # change to 'abide' for ABIDE dataset
# dataset = 'abide'  # change to 'abide' for ABIDE dataset
train_split = f'data/cvpr_dataset/split/{dataset}_train.txt'
val_split = f'data/cvpr_dataset/split/{dataset}_val.txt'
test_split = f'data/cvpr_dataset/split/{dataset}_test.txt'
label_csv = f'data/{dataset}_adj/{dataset}_v2_dx_group.csv'

npy_dir = f'data/{dataset}_adj/mass'
all_npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
all_npy_sidints = set([extract_sid_from_filename(f, file_type=f'{dataset}')[1] for f in all_npy_files])
# all_npy_sidints = set(int(sid) for sid in all_npy_sids)


splits = {
    'train': load_split_file(train_split),
    'val': load_split_file(val_split),
    'test': load_split_file(test_split),
}

labels = pd.read_csv(label_csv)

all_matched_sids = set()

for split_name, split_ids in splits.items():
    split_sidint = [extract_sid_from_filename(f, file_type=f'{dataset}_split')[1] for f in split_ids]
    # _sidint = [int(sid) for sid in _sids]


    csv_sids = set(labels['SUB_ID'].tolist())

    matched_sids = set(split_sidint).intersection(csv_sids)
    # not_matched_sids = set(split_sidint).difference(csv_sids)
    # in_csv_not_in_split = csv_sids.difference(set(split_sidint))
    in_split_not_in_csv = set(split_sidint).difference(csv_sids)

    print("--------------------------------------------------")
    print(f"Split: {split_name}")
    print(f"file path: {train_split if split_name=='train' else val_split if split_name=='val' else test_split}")
    print('---------------')
    print(f"Total unique SIDs in split: {len(set(split_sidint))}")
    print(f"Both in split and csv: {len(matched_sids)}")
    print(f"In split not in CSV: {len(in_split_not_in_csv)}")

    all_matched_sids.update(matched_sids)

print("==================================================")
print(f"Total unique matched SIDs across all splits: {len(all_matched_sids)}")
print(f"Total unique in csv: {len(csv_sids)}")
print(f"Total unique NPY SIDs: {len(all_npy_sidints)}")
print("---------------------------------------------------")
print(f"In CSV not in matched SIDs: {len(csv_sids.difference(all_matched_sids))}")
print(f"In matched SIDs not in CSV: {len(all_matched_sids.difference(csv_sids))}")
print(f"Both in CSV and matched SIDs: {len(csv_sids.intersection(all_matched_sids))}")
print("---------------------------------------------------")
print(f"In NPY SIDs not in matched SIDs: {len(all_npy_sidints.difference(all_matched_sids))}")
print(f"In matched SIDs not in NPY SIDs: {len(all_matched_sids.difference(all_npy_sidints))}")
print(f"Both in NPY and matched SIDs: {len(all_npy_sidints.intersection(all_matched_sids))}")
print("---------------------------------------------------")


# # unmatched_npy_sids = all_npy_sidints.difference(all_matched_sids)
# unmatched_npy_sids = all_matched_sids - all_npy_sidints # include in all_matched_sids but not in npy files
# print(f"Total unmatched NPY SIDs: {len(unmatched_npy_sids)}")
# print(f"In split not in NPY SIDs: {len(unmatched_npy_sids)}")
# print(f"In NPY not in split SIDs: {len(all_npy_sidints - all_matched_sids)}")
# print(f"In both NPY and split SIDs: {len(all_npy_sidints.intersection(all_matched_sids))}")
# print(f"Total unique NPY SIDs: {len(all_npy_sidints.union(all_matched_sids))}")
# if len(unmatched_npy_sids) > 0:
#     with open('unmatched_npy_sids.txt', 'w') as f:
#         for sid in unmatched_npy_sids:
#             f.write(f"{sid}\n")