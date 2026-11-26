# A2C: hyperparameters and results

Final eval = final checkpoint, greedy with invalid-action masking on held-out seeds 10000–10019. Frames to 90% (training) = first frame where the 100-episode mean training return (sampled actions, so it includes exploration) reaches 90% of the heuristic; (greedy eval) = first 100k-frame evaluation at or above that level. Plateau = mean training return over the last 10% of frames. Best eval during training is selected on the same seeds (optimistic).

| Run | lr | steps | gamma | lmbda | entropy | value_coef | Note | Final eval (mean ± std) | Frames to 90% (training) | Frames to 90% (greedy eval) | Plateau return | Best eval during training |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 0.0007 | 16 | 0.99 | 1.0 | 0.01 | 0.5 | Baseline (n-step) | 189.0 ± 55.6 | 445,824 | 500,096 | 182.4 | 194.9 |
| A2 | 0.00025 | 16 | 0.99 | 1.0 | 0.01 | 0.5 | Slow lr | 187.3 ± 54.8 | 999,552 | 900,096 | 164.4 | 187.3 |
| A3 | 0.002 | 16 | 0.99 | 1.0 | 0.01 | 0.5 | Fast lr | 198.0 ± 61.8 | 288,256 | 200,064 | 190.1 | 198.0 |
| A4 | 0.0007 | 5 | 0.99 | 1.0 | 0.01 | 0.5 | Short rollout | 164.7 ± 53.7 | 579,320 | not reached | 171.1 | 173.6 |
| A5 | 0.0007 | 64 | 0.99 | 1.0 | 0.01 | 0.5 | Long rollout | 161.1 ± 60.2 | not reached | not reached | 137.1 | 162.8 |
| A6 | 0.0007 | 16 | 0.99 | 0.95 | 0.01 | 0.5 | GAE lambda 0.95 | 193.1 ± 50.3 | 402,048 | 300,032 | 184.4 | 193.1 |
| A7 | 0.0007 | 16 | 0.99 | 1.0 | 0.0 | 0.5 | No entropy bonus | 185.5 ± 57.5 | 500,352 | 500,096 | 182.6 | 192.3 |
| A8 | 0.0007 | 16 | 0.99 | 1.0 | 0.05 | 0.5 | Large entropy bonus | 191.0 ± 50.9 | 498,560 | 500,096 | 183.9 | 193.4 |
| A9 | 0.0007 | 16 | 0.99 | 1.0 | 0.01 | 1.0 | Large value coefficient | 191.6 ± 45.7 | 430,208 | 400,000 | 184.7 | 191.6 |
| A10 | 0.0007 | 16 | 0.95 | 1.0 | 0.01 | 0.5 | Short horizon | 195.9 ± 54.3 | 368,256 | 400,000 | 185.6 | 195.9 |
