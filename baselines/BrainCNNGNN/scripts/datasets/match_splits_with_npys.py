'''
Author: ViolinSolo
Date: 2025-10-31 18:08:57
LastEditTime: 2025-11-13 10:46:45
LastEditors: ViolinSolo
Description: 

    load three dataset slits text files and match with corresponding npy files

FilePath: /ProjectBrainBaseline/scripts/datasets/match_splits_with_npys.py
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


def extract_sid_from_filename(filename, dataset='abide'):
    """Extract subject ID from the .npy filename.
    return (sid_str, sid_int) or None if not found
    """
    base = os.path.basename(filename)
    if dataset == 'abide':  # disk filepath format
        # Example filename: Caltech_0051461_func_preproc_0.npy, first find the "func_preproc", then get the part before it 0051461
        func_preproc_index = base.find("_func_preproc")
        if func_preproc_index != -1:
            sid = base[:func_preproc_index].split('_')[-1]
            return sid, int(sid)  # use int to omit leading zeros
    elif dataset == 'abide_split':  # split file format
        # base name example:
        # Yale_0050628_control_func_preproc_frames_000000-000199.nii.gz  => extract 0050628
        func_preproc_index = base.find("_func_preproc")
        if func_preproc_index != -1:
            sid = base[:func_preproc_index].split('_')[-2]
            return sid, int(sid)
    elif dataset == 'adhd':
        # Example filename: data/adhd_adj/mass/0010001_run-1.npy -> extract 0010001
        run_index = base.find("_run-")
        if run_index != -1:
            sid = base[:run_index].split('_')[-1]
            return sid, int(sid)
    elif dataset == 'adhd_split':
        # base name example:
        # ADHD_NeuroIMAGE_7446626_patient_zscore.nii.gz  => extract 7446626
        # ADHD_Peking_1_9890726_control_zscore.nii.gz => extract 9890726
        sid_idx = -3
        sid = base.split('_')[sid_idx]
        sid = f'{int(sid):07d}'  # ensure leading zeros for adhd SIDs
        return sid, int(sid)
    elif dataset == 'adni':
        # Example filename: sub-003S6264_ses-01_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold_timeseries.npy
        sid = base.split('_')[0].split('-')[1]
        return sid, None  # ADNI SIDs are alphanumeric, return None for int
    elif dataset == 'adni_split':
        # Example filename: ADNI_sub-168S6131_ses-01_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz_cn_zscore.nii.gz
        # extract 168S6131
        sid = base.split('_')[1].split('-')[1]
        return sid, None  # ADNI SIDs are alphanumeric, return None for int
    elif dataset == 'ppmi':  # in the os, npy filenames
        # Example filename: sub-100878_ses-01_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold_timeseries.npy
        sid = base.split('_')[0].split('-')[1]
        return sid, int(sid)  # PPMI SIDs are alphanumeric, return None for int
    elif dataset == 'ppmi_split':  # in the split text files
        # Example filename: ADNI_sub-106126_ses-01_task-rest_bold_mc.nii.gz_native_zscore.nii.gz
        sid = base.split('_')[1].split('-')[1]
        return sid, int(sid)  # PPMI SIDs are alphanumeric, return None for int
    return None


def main():
    parser = argparse.ArgumentParser(description="Match dataset splits with corresponding npy files.")
    parser.add_argument("--train_split", required=True, help="Path to the train split file.")
    parser.add_argument("--val_split", required=True, help="Path to the validation split file.")
    parser.add_argument("--test_split", required=True, help="Path to the test split file.")
    parser.add_argument("--label", required=True, help="Path to the labels CSV file.")
    parser.add_argument("--saved_label_cols", default=[], nargs='+', help="On witch cols will be saved the matched npy files of the labels.")
    parser.add_argument("--npy_dir", required=True, default=[], nargs='+', help="Directory(s) containing the .npy files.")
    parser.add_argument("--saved_unique_prefix", default="", help="Prefix to append to saved matched split files.")
    parser.add_argument("--dataset", default="abide", choices=['abide', 'adhd', 'adni', 'ppmi'], help="Dataset name for SID extraction logic.")
    args = parser.parse_args()

    train_ids = load_split_file(args.train_split)
    val_ids = load_split_file(args.val_split)
    test_ids = load_split_file(args.test_split)

    all_npy_files = []
    for _npy_dir in args.npy_dir:
        all_npy_files.extend([os.path.join(_npy_dir, f) for f in os.listdir(_npy_dir) if f.endswith('.npy')])

    print("--------------------------------------------------")
    print(f"Loaded {len(train_ids)} train IDs, {len(val_ids)} val IDs, {len(test_ids)} test IDs.")
    print(f"Found {len(all_npy_files)} npy files in directories {args.npy_dir}.")

    # resample the args.npy_dir, to find the common root directory
    common_root = os.path.commonpath(args.npy_dir)
    print(f"Common root directory for npy files: {common_root}")

    labels = pd.read_csv(args.label, dtype={'SUB_ID': str})

    # print('label head:')
    # print(labels.head())

    matched_npys = {'train': [], 'val': [], 'test': []}
    not_matched = []

    for _npy_file in tqdm(all_npy_files, desc="Matching npy files with splits", unit="files"):
        sid_str, sid_int = extract_sid_from_filename(_npy_file, dataset=args.dataset)

        # print(f"Processing NPY file: {_npy_file}, extracted SID: {sid_str} -> {sid_int}")
        # print(f"Available SIDs in labels CSV: {labels['SUB_ID'].tolist()[:10]} ...")  # print first 10 for brevity

        assert sid_str in labels['SUB_ID'].values, f"SID {sid_str} not found in labels CSV."

        results = []
        _found = []
        for train_id in train_ids:
            _train_sid, _train_sid_int = extract_sid_from_filename(train_id, dataset=f'{args.dataset}_split')
            # if sid_int == _train_sid_int:
            if sid_str == _train_sid:
                results.append('train')
                _found.append((train_id, _npy_file))
            # elif str(sid_int) in train_id:
                # print(f"Debug: SID int {sid_int} found in train_id {train_id} for NPY file {_npy_file}")
        for val_id in val_ids:
            _val_sid, _val_sid_int = extract_sid_from_filename(val_id, dataset=f'{args.dataset}_split')
            # if sid_int == _val_sid_int:
            if sid_str == _val_sid:
                results.append('val')
                _found.append((val_id, _npy_file))
            # elif str(sid_int) in val_id:
                # print(f"Debug: SID int {sid_int} found in val_id {val_id} for NPY file {_npy_file}")
                # print(f"Debug: SID int {sid_int} found in val_id {val_id} for NPY file {_npy_file}")
        for test_id in test_ids:
            _test_sid, _test_sid_int = extract_sid_from_filename(test_id, dataset=f'{args.dataset}_split')
            # if sid_int == _test_sid_int:
            if sid_str == _test_sid:
                results.append('test')
                _found.append((test_id, _npy_file))
            # elif str(sid_int) in test_id:
                # print(f"Debug: SID int {sid_int} found in test_id {test_id} for NPY file {_npy_file}")
                # print(f"Debug: SID int {sid_int} found in test_id {test_id} for NPY file {_npy_file}")
        if len(results) > 1:
            print("--------------------------------------------------")
            print(f"Debug Info: Multiple matches found for SID {sid_str} in NPY file {_npy_file}: {_found}")
            print(f"Found: {', '.join([f'{entry[0]} (NPY: {entry[1]})' for entry in _found])}")
            raise ValueError(f"Error: SID {sid_str} found in multiple splits: {results}")
        elif len(results) == 0:
            if args.dataset == 'abide':
                raise ValueError(f"Error: SID {sid_str} not found in any split.")
            elif args.dataset == 'adhd':
                not_matched.append(_npy_file)
                continue  # skip this npy file
                # raise ValueError(f"Error: SID {sid_str} not found in any split.")
            elif args.dataset == 'adni':
                raise ValueError(f"Error: SID {sid_str} not found in any split.")
            elif args.dataset == 'ppmi':
                # raise ValueError(f"Error: SID {sid_str} not found in any split.")
                not_matched.append(_npy_file)
                continue  # skip this npy file
        
        if len(args.saved_label_cols) > 0:
            label_row = labels[labels['SUB_ID'] == sid_str]
            # if label_row.empty:
            #     raise ValueError(f"SID {sid} not found in labels CSV.")
            to_save = {
                'npy_file': _npy_file,
                'split': results[0]
            }
            for col in args.saved_label_cols:
                if col not in label_row.columns:
                    raise ValueError(f"Column {col} not found in labels CSV.")
                label_value = label_row[col].values[0]
                to_save[col] = label_value
                # print(f"SID: {sid}, NPY: {_npy_file}, {col}: {label_value}")

            matched_npys[results[0]].append(to_save)
        else:
            matched_npys[results[0]].append({
                'npy_file': _npy_file,
                'split': results[0]
            })

    for split in ['train', 'val', 'test']:
        df = pd.DataFrame(matched_npys[split])
        df['npy_file'] = df['npy_file'].apply(lambda x: os.path.basename(x))
        # order by SUB_ID and npy_file
        df = df.sort_values(by=['SUB_ID', 'npy_file'])
        save_path = os.path.join(common_root, f"{args.saved_unique_prefix}{split}_matched_labels.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved matched {split} split to {save_path} with {len(df)} entries.")
        print(f"Unique SIDs in {split}: {df['SUB_ID'].nunique()}")

        # also save the npy absolute paths into a text file
        # npy_paths = [os.path.abspath(os.path.join(common_root, entry['npy_file'])) for entry in matched_npys[split]]
        npy_paths = [os.path.abspath(entry['npy_file']) for entry in matched_npys[split]]
        split_txt_path = os.path.join(common_root, f"{args.saved_unique_prefix}{split}_npy_paths.txt")
        with open(split_txt_path, 'w') as f:
            for npy_path in npy_paths:
                f.write(f"{npy_path}\n")
        print(f"Saved {split} npy paths to {split_txt_path}.")

    if not_matched:
        print("--------------------------------------------------")
        print(f"Found {len(not_matched)} unmatched NPY files:")
        print("--------------------------------------------------")
        # for unmatched in not_matched:
        #     print(f" - {unmatched}")

        with open(os.path.join(common_root, 'not_matched_npy_files.txt'), 'w') as f:
            for unmatched in not_matched:
                f.write(f"{unmatched}\n")

if __name__ == "__main__":
    main()