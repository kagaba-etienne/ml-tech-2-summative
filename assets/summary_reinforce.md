# REINFORCE: hyperparameters and results

Final eval = final checkpoint, greedy with invalid-action masking on held-out seeds 10000–10019. Frames to 90% (training) = first frame where the 100-episode mean training return (sampled actions, so it includes exploration) reaches 90% of the heuristic; (greedy eval) = first 100k-frame evaluation at or above that level. Plateau = mean training return over the last 10% of frames. Best eval during training is selected on the same seeds (optimistic).

| Run | lr | gamma | episodes | baseline | entropy | adv_norm | Note | Final eval (mean ± std) | Frames to 90% (training) | Frames to 90% (greedy eval) | Plateau return | Best eval during training |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 0.001 | 0.99 | 8 | True | 0.0 | True | Baseline | 152.0 ± 59.2 | not reached | not reached | 128.1 | 159.7 |
| R2 | 0.0003 | 0.99 | 8 | True | 0.0 | True | Slow lr | 68.4 ± 53.9 | not reached | not reached | -17.9 | 68.4 |
| R3 | 0.003 | 0.99 | 8 | True | 0.0 | True | Fast lr | 167.3 ± 61.4 | 958,359 | 900,551 | 168.3 | 183.4 |
| R4 | 0.001 | 0.95 | 8 | True | 0.0 | True | Short horizon | 185.2 ± 53.5 | 947,621 | 800,573 | 168.2 | 185.2 |
| R5 | 0.001 | 0.99 | 2 | True | 0.0 | True | Few episodes per update | 175.5 ± 54.7 | not reached | not reached | 151.5 | 175.7 |
| R6 | 0.001 | 0.99 | 32 | True | 0.0 | True | Many episodes per update | 97.9 ± 49.3 | not reached | not reached | 32.8 | 97.9 |
| R7 | 0.001 | 0.99 | 8 | False | 0.0 | True | No baseline (pure MC) | 164.3 ± 60.4 | not reached | not reached | 145.4 | 164.3 |
| R8 | 0.001 | 0.99 | 8 | True | 0.01 | True | Small entropy bonus | 191.9 ± 64.1 | not reached | 800,613 | 135.7 | 191.9 |
| R9 | 0.001 | 0.99 | 8 | True | 0.05 | True | Large entropy bonus | 198.6 ± 67.3 | not reached | 700,141 | 79.1 | 198.6 |
| R10 | 0.001 | 0.99 | 8 | True | 0.0 | False | No advantage normalisation | 145.2 ± 53.7 | not reached | not reached | 127.9 | 165.7 |
