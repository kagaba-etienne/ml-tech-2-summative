# DQN: hyperparameters and results

Final eval = final checkpoint, greedy with invalid-action masking on held-out seeds 10000–10019. Frames to 90% (training) = first frame where the 100-episode mean training return (sampled actions, so it includes exploration) reaches 90% of the heuristic; (greedy eval) = first 100k-frame evaluation at or above that level. Plateau = mean training return over the last 10% of frames. Best eval during training is selected on the same seeds (optimistic).

| Run | lr | gamma | buffer | batch | eps_frac | target | double | Note | Final eval (mean ± std) | Frames to 90% (training) | Frames to 90% (greedy eval) | Plateau return | Best eval during training |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 0.0005 | 0.99 | 100000 | 128 | 0.5 | soft | True | Baseline | 198.5 ± 52.4 | 463,808 | 100,000 | 188.3 | 203.2 |
| D2 | 0.0001 | 0.99 | 100000 | 128 | 0.5 | soft | True | Slow lr | 167.1 ± 57.8 | 852,832 | not reached | 166.5 | 172.2 |
| D3 | 0.001 | 0.99 | 100000 | 128 | 0.5 | soft | True | Fast lr | 189.3 ± 50.5 | 473,344 | 100,000 | 185.4 | 213.7 |
| D4 | 0.0005 | 0.95 | 100000 | 128 | 0.5 | soft | True | Short horizon | 189.9 ± 59.6 | 473,888 | 100,000 | 186.8 | 207.1 |
| D5 | 0.0005 | 0.99 | 10000 | 128 | 0.5 | soft | True | Small replay | 185.3 ± 56.3 | 476,416 | 100,000 | 184.0 | 206.9 |
| D6 | 0.0005 | 0.99 | 100000 | 32 | 0.5 | soft | True | Small batch | 212.3 ± 61.5 | 476,640 | 200,000 | 188.2 | 212.3 |
| D7 | 0.0005 | 0.99 | 100000 | 128 | 0.1 | soft | True | Little exploration | 191.5 ± 56.8 | 105,120 | 100,000 | 187.0 | 197.9 |
| D8 | 0.0005 | 0.99 | 100000 | 128 | 0.8 | soft | True | Long exploration | 195.1 ± 49.5 | 742,432 | 100,000 | 187.7 | 208.7 |
| D9 | 0.0005 | 0.99 | 100000 | 128 | 0.5 | hard | True | Hard target every 2,000 updates | 192.3 ± 53.9 | 470,688 | 100,000 | 183.1 | 215.8 |
| D10 | 0.0005 | 0.99 | 100000 | 128 | 0.5 | soft | False | Overestimation (no double Q) | 195.9 ± 53.8 | 475,200 | 100,000 | 187.1 | 214.6 |
