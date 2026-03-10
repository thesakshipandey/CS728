# Temporal-order GP search

The GP search recovered the same hyperparameter region as the earlier successful fixed-length run. The stronger kept checkpoint for that setting is `EC_best_torder_clip1000_alpha2000_lr00100_nh100_fix100_final_state.npz`, which reached `4.54%` best validation error.

Best short-run surrogate: `ECgp_short_02_clip1000_alpha2000_lr00100_nh100_fix100` with surrogate loss 39.334, best valid error 39.150%, and best train nll 0.7251.

Best full rerun: `ECgp_full_00_clip1000_alpha2000_lr00100_nh100_fix100` with best valid error 15.490%, best train nll 0.4372, and final rho 1.150.

- `ECgp_full_00_clip1000_alpha2000_lr00100_nh100_fix100` [full]: cutoff=1.0000, alpha=2.0000, lr=0.01000, nhid=100, len=100-100, surrogate=19.182, best valid=15.490%, best train nll=0.4372
- `ECgp_short_02_clip1000_alpha2000_lr00100_nh100_fix100` [short]: cutoff=1.0000, alpha=2.0000, lr=0.01000, nhid=100, len=100-100, surrogate=39.334, best valid=39.150%, best train nll=0.7251
- `ECgp_short_03_clip0500_alpha1000_lr00100_nh100_fix100` [short]: cutoff=0.5000, alpha=1.0000, lr=0.01000, nhid=100, len=100-100, surrogate=41.838, best valid=41.200%, best train nll=0.8357
- `ECgp_short_04_clip0206_alpha1863_lr00117_nh125_fix100` [short]: cutoff=0.2061, alpha=1.8632, lr=0.01165, nhid=125, len=100-100, surrogate=89.005, best valid=73.980%, best train nll=1.3725
- `ECgp_short_01_clip1000_alpha2000_lr00100_nh50_rand50_200` [short]: cutoff=1.0000, alpha=2.0000, lr=0.01000, nhid=50, len=50-200, surrogate=89.297, best valid=74.320%, best train nll=1.3698
- `ECgp_short_00_clip0050_alpha2000_lr00100_nh50_rand50_200` [short]: cutoff=0.0500, alpha=2.0000, lr=0.01000, nhid=50, len=50-200, surrogate=89.589, best valid=74.330%, best train nll=1.3804
