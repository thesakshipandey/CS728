# Extra-credit temporal order note

The GP search used a surrogate loss of `best_valid_error + 40 * max(best_train_nll - 1.0, 0) + 10 * max(final_train_nll - best_train_nll, 0)`.
The best short-run surrogate was `ECgp_short_02_clip1000_alpha2000_lr00100_nh100_fix100` with cutoff=1.0000, alpha=2.0000, lr=0.01000, nhid=100, and length 100-100.

The GP-selected full rerun was:

- Run name: `ECgp_full_00_clip1000_alpha2000_lr00100_nh100_fix100`
- Key settings: `--min_length 100 --max_length 100 --nhid 100 --cutoff 1.0000 --alpha 2.0000 --lr 0.01000`
- Best observed validation error: `15.49%`
- Best train NLL: `0.4372`
- Final rho(W_hh): `1.1503`

The stronger kept checkpoint for the same setting is:

- Checkpoint: `EC_best_torder_clip1000_alpha2000_lr00100_nh100_fix100_final_state.npz`
- Source run: `EC_torder_smart_clip10_alpha20_lr001_nh100_fix100_final_state.npz`
- Best observed validation error: `4.54%`
- Best train NLL: `0.3575`
