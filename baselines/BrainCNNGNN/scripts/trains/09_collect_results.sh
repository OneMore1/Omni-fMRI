#!/usr/bin/env bash
###
 # @Author: ViolinSolo
 # @Date: 2025-11-07 15:06:07
 # @LastEditTime: 2025-11-27 17:47:09
 # @LastEditors: ViolinSolo
 # @Description: 
 # @FilePath: /ProjectBrainBaseline/scripts/trains/09_collect_results.sh
### 

python scripts/trains/collect_results.py checkpoints/checkpoints -o combined_results.csv --non-recursive
python scripts/trains/collect_results.py checkpoints/00_grid_search_bs_64 -o grid_search_bs_64_results.csv --non-recursive
python scripts/trains/collect_results.py checkpoints/01_reported_with_bestHP -o reported_with_bestHP_results.csv --non-recursive