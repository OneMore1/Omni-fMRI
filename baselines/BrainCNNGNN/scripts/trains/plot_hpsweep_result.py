'''
Author: ViolinSolo
Date: 2025-11-20 14:36:58
LastEditTime: 2025-11-20 14:36:59
LastEditors: ViolinSolo
Description: This should be run after using the collect_results.py script to gather results from hyperparameter sweeps.
Generates plots to visualize the performance across different hyperparameter settings.
FilePath: /ProjectBrainBaseline/scripts/trains/plot_hpsweep_result.py
'''

import pandas as pd


df = pd.read_csv('/home/u2280887/GitHub/ProjectBrainBaseline/combined_results.csv')
df.head()

# plot the results per dataset, for each model type, draw lines, each line represent a fixed wd, and each x is the lr, y is accuracy, only consider the mass pretrained model


df_mass = df[df['subdir'].str.contains('mass')]

unique_datasets = df_mass['downstream_dataset'].unique()
unique_models = df_mass['model'].unique()
unique_wds = df_mass['wd'].unique()
unique_lrs = df_mass['lr'].unique()

n_models = len(unique_models)
n_datasets = len(unique_datasets)


import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8, 12))
for i, dataset in enumerate(unique_datasets):
    for j, model in enumerate(unique_models):
        plt.subplot(n_datasets, n_models, i * n_models + j + 1)
        for wd in unique_wds:
            subset = df_mass[(df_mass['downstream_dataset'] == dataset) &
                             (df_mass['model'] == model) &
                             (df_mass['wd'] == wd)]
            lrs = subset['lr']
            accs = subset['accuracy']
            plt.plot(lrs, accs, marker='o', label=f'wd={wd}')
            # highlight best lr for each wd with a red square and corresponding lr
            best_idx = np.argmax(accs)
            plt.plot(lrs.iloc[best_idx], accs.iloc[best_idx], marker='s', markersize=10, color='red')
            plt.axvline(x=lrs.iloc[best_idx], color='red', linestyle='--')

        # find the best lr wd combination within this group and print int on the title
        best_subset = df_mass[(df_mass['downstream_dataset'] == dataset) &
                               (df_mass['model'] == model)]
        best_idx = best_subset['accuracy'].idxmax()
        best_lr = best_subset.loc[best_idx, 'lr']
        best_wd = best_subset.loc[best_idx, 'wd']
        best_acc = best_subset.loc[best_idx, 'accuracy']
        plt.title(f"{dataset} - {model} \n Best: lr={best_lr}, wd={best_wd}, acc={best_acc:.4f}")
        plt.xlabel('Learning Rate')
        plt.ylabel('Accuracy')
        plt.xscale('log')
        plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('/home/u2280887/GitHub/ProjectBrainBaseline/hpsweep_results_mass_pretrained.png', dpi=300)