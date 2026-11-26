# Best configuration of each algorithm

Best = highest final eval within the algorithm's 10-config grid. Same 1M-frame budget, training seed 0.

| Algorithm | Config | Change from baseline | Final eval (mean ± std) | Hard variant (p_order 0.02) | Frames to 90% (training) | Frames to 90% (greedy eval) | Plateau | Deliveries | Emergencies | Expired | Forced landings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DQN | D6 | Small batch | 212.3 ± 61.5 | 286.0 ± 38.3 | 476,640 | 200,000 | 188.2 | 13.40 | 4.25 | 0.20 | 0.00 |
| REINFORCE | R9 | Large entropy bonus | 198.6 ± 67.3 | 274.1 ± 49.7 | not reached | 700,141 | 79.1 | 12.50 | 4.10 | 1.20 | 0.00 |
| PPO | P4 | Short rollout | 205.2 ± 61.6 | 298.7 ± 51.0 | 258,560 | 200,192 | 190.0 | 13.35 | 4.55 | 0.50 | 0.00 |
| A2C | A3 | Fast lr | 198.0 ± 61.8 | 281.9 ± 64.3 | 288,256 | 200,064 | 190.1 | 13.25 | 4.20 | 0.50 | 0.10 |

Baselines on the same held-out seeds:

- heuristic: 195.6
- random: -103.3
- masked_random: -21.2
- hold: -72.5
