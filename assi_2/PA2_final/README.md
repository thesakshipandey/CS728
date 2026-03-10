# PA2 final

This folder is the cleaned hand-in copy.

- `code/` has the working PyTorch code.
- `baselines/` keeps the original `*_org.py` files for comparison.
- `results/checkpoints/` has the final checkpoints for `A1` to `A5` and `B1` to `B2`.
- `results/plots/` has the seaborn figures and the main write-up.
- `results/torder/` has the extra-credit sweep logs, summary, and the tuned checkpoint.
- `docs/` has the assignment PDF.

Relative to the baseline files, `code/model.py` and `code/train.py` mainly add the required RNN and GRU implementations, gradient-through-time diagnostics, gate diagnostics, clipping support, and checkpoint fields used by the report. A few small cleanup changes are also present so the consolidated folder runs cleanly.

Best extra-credit setup found:

- task: temporal order
- model: vanilla RNN
- init: `smart_tanh`
- `nhid=100`
- `cutoff=1.0`
- `alpha=2.0`
- `lr=0.01`
- `min_length=max_length=100`
- GP search recovered the same setting
- best kept checkpoint: `results/checkpoints/EC_best_torder_clip1000_alpha2000_lr00100_nh100_fix100_final_state.npz`
- best observed validation error for that kept checkpoint: `4.54%`

Useful entry points:

- train a required run: `python code/train.py --task mem --model rnn --alpha 0 --name ../results/checkpoints/A1_mem_rnn_tanh_noclip`
- rebuild plots: `python code/plotter.py --device cpu`
- rerun the required set: `python code/run_required.py --device cpu --force`
- rerun the temporal-order GP search: `python code/tune_torder.py --device cpu --force`
