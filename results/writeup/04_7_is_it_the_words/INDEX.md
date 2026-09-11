# 4.7 Is it the words? (F1-F4)

Plan bullets:
- a' layer 18 val 0.950, shuffled ~0.50, layer-0 0.767; bare-neutral 0.983 untrained; implied recall 0.850 / BA 0.912; human 0.917; bag-of-words 0.850 / 0.575 / 0.350; length-only 0.583; ask-the-model implied 0.150. Verify all against phase3b_headline.csv and phase3b_baselines.csv.
- F4 plots probe-weight cosines in standardised space (0.53-0.77 distressed vs frustrated); the 0.85 in the text is the raw-space mean-difference cosine (phase3b_geometry_layer18.csv). Verify the 0.24-0.50 distressed-vs-positive range against phase3b_probe_by_layer.csv.
- v1 confounded renders and tables are archived in results/archive/phase3_v1/ (not copied here).

Files:
- `F1_probe_acc_by_layer_task_a.png` <- `results/phase3/F1_probe_acc_by_layer_task_a.png`: v2
- `F2_probe_acc_by_layer_task_b.png` <- `results/phase3/F2_probe_acc_by_layer_task_b.png`: v2
- `F3_pdist_boxplots_bestlayer.png` <- `results/phase3/F3_pdist_boxplots_bestlayer.png`: v2
- `F4_probe_direction_cosines.png` <- `results/phase3/F4_probe_direction_cosines.png`: v2
- `phase3b_headline.csv` <- `results/phase3/phase3b_headline.csv`: headline numbers
- `phase3b_baselines.csv` <- `results/phase3/phase3b_baselines.csv`: bag-of-words, length-only, ask-the-model
- `phase3b_probe_by_layer.csv` <- `results/phase3/phase3b_probe_by_layer.csv`: per-layer accuracies and cosines (F1, F2, F4 source)
- `phase3b_geometry_layer18.csv` <- `results/phase3/phase3b_geometry_layer18.csv`: raw-space cosines
- `phase3b_best_layers.json` <- `results/phase3/phase3b_best_layers.json`
