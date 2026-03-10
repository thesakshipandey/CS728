# PA2 dynamics summary

## A1 (A1_mem_rnn_tanh_noclip)
- Final valid error: 100.000% | best: 99.990% | final grad norm: 0.04526
- Gradient histogram: mostly vanishing | hidden saturation: heavily saturated | saturated fraction (<0.05): 96.4%
- rho(W_hh): start 0.841, end 9.282, max 9.314, corr(rho, median log10 g_t) nan

## A2 (A2_mem_rnn_tanh_clip005)
- Final valid error: 100.000% | best: 100.000% | final grad norm: 33.14
- Gradient histogram: mostly vanishing | hidden saturation: mostly unsaturated | saturated fraction (<0.05): 0.0%
- rho(W_hh): start 0.709, end 1.139, max 1.168, corr(rho, median log10 g_t) nan

## A3 (A3_mem_rnn_tanh_clip001)
- Final valid error: 100.000% | best: 100.000% | final grad norm: 0.1221
- Gradient histogram: mostly vanishing | hidden saturation: mostly unsaturated | saturated fraction (<0.05): 1.9%
- rho(W_hh): start 0.709, end 1.090, max 1.166, corr(rho, median log10 g_t) nan

## A4 (A4_mem_gru_noclip)
- Final valid error: 100.000% | best: 100.000% | final grad norm: 0.06923
- Gradient histogram: mostly vanishing | hidden saturation: mostly unsaturated | saturated fraction (<0.05): 0.0%
- rho(W_hh): start 0.076, end 0.094, max 0.194, corr(rho, median log10 g_t) nan
- Gate saturation histograms: z mixed, r more mass near 0.5 (unsaturated)

## A5 (A5_mem_gru_clip005)
- Final valid error: 100.000% | best: 100.000% | final grad norm: 0.04677
- Gradient histogram: mostly vanishing | hidden saturation: mostly unsaturated | saturated fraction (<0.05): 0.0%
- rho(W_hh): start 0.076, end 0.183, max 0.183, corr(rho, median log10 g_t) nan
- Gate saturation histograms: z mixed, r more mass near 0.5 (unsaturated)

## B1 (B1_mul_rnn_tanh_noclip)
- Final valid error: 40.140% | best: 25.900% | final grad norm: 0.03919
- Gradient histogram: mostly vanishing | hidden saturation: mostly unsaturated | saturated fraction (<0.05): 0.0%
- rho(W_hh): start 0.722, end 0.711, max 0.722, corr(rho, median log10 g_t) nan

## B2 (B2_mul_gru_noclip)
- Final valid error: 38.700% | best: 24.250% | final grad norm: 0.06164
- Gradient histogram: mostly vanishing | hidden saturation: mostly unsaturated | saturated fraction (<0.05): 0.0%
- rho(W_hh): start 0.081, end 0.081, max 0.081, corr(rho, median log10 g_t) nan
- Gate saturation histograms: z mixed, r more mass near 0.5 (unsaturated)

## Clipping on memorization (A1 vs A2 vs A3)
- A1 no clipping: mostly vanishing; A2 cutoff 0.05: mostly vanishing; A3 cutoff 0.01: mostly vanishing.
- Saturation fraction drops from 96.4% in A1 to 0.0% in A2 and 1.9% in A3.

## RNN vs GRU on memorization
- A4 GRU without clipping shows mostly vanishing; A1 RNN without clipping shows mostly vanishing.
- GRU gates: A4 has z mixed and r more mass near 0.5 (unsaturated); A5 has z mixed and r more mass near 0.5 (unsaturated).

## Multiplication task
- B1 RNN: mostly vanishing with mostly unsaturated; B2 GRU: mostly vanishing with mostly unsaturated.
