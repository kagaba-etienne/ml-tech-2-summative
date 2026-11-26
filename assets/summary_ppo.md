# PPO: hyperparameters and results

Final eval = final checkpoint, greedy with invalid-action masking on held-out seeds 10000–10019. Frames to 90% (training) = first frame where the 100-episode mean training return (sampled actions, so it includes exploration) reaches 90% of the heuristic; (greedy eval) = first 100k-frame evaluation at or above that level. Plateau = mean training return over the last 10% of frames. Best eval during training is selected on the same seeds (optimistic).

| Run | lr | fpb | minibatch | epochs | clip | entropy | lmbda | Note | Final eval (mean ± std) | Frames to 90% (training) | Frames to 90% (greedy eval) | Plateau return | Best eval during training |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | 0.0003 | 2048 | 256 | 4 | 0.2 | 0.01 | 0.95 | Baseline (verified) | 196.7 ± 54.7 | 378,880 | 301,056 | 182.0 | 196.7 |
| P2 | 0.0001 | 2048 | 256 | 4 | 0.2 | 0.01 | 0.95 | Slow lr | 174.9 ± 59.1 | 921,600 | 800,768 | 165.8 | 187.6 |
| P3 | 0.001 | 2048 | 256 | 4 | 0.2 | 0.01 | 0.95 | Fast lr | 194.5 ± 55.7 | 188,416 | 100,352 | 194.9 | 201.5 |
| P4 | 0.0003 | 512 | 128 | 4 | 0.2 | 0.01 | 0.95 | Short rollout | 205.2 ± 61.6 | 258,560 | 200,192 | 190.0 | 205.2 |
| P5 | 0.0003 | 8192 | 512 | 4 | 0.2 | 0.01 | 0.95 | Long rollout | 198.2 ± 49.2 | 532,480 | 507,904 | 186.8 | 198.2 |
| P6 | 0.0003 | 2048 | 256 | 10 | 0.2 | 0.01 | 0.95 | Many epochs | 193.9 ± 59.8 | 231,424 | 200,704 | 193.0 | 197.1 |
| P7 | 0.0003 | 2048 | 256 | 4 | 0.1 | 0.01 | 0.95 | Tight clip | 185.5 ± 51.3 | 421,888 | 401,408 | 180.3 | 189.7 |
| P8 | 0.0003 | 2048 | 256 | 4 | 0.3 | 0.01 | 0.95 | Loose clip | 195.6 ± 52.1 | 399,360 | 401,408 | 185.7 | 195.6 |
| P9 | 0.0003 | 2048 | 256 | 4 | 0.2 | 0.05 | 0.95 | Large entropy bonus | 188.0 ± 56.6 | 843,776 | 301,056 | 167.7 | 188.1 |
| P10 | 0.0003 | 2048 | 256 | 4 | 0.2 | 0.01 | 0.8 | Low GAE lambda | 189.6 ± 46.0 | 503,808 | 501,760 | 186.6 | 192.1 |
