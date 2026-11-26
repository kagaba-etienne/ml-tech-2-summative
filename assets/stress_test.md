# Paired comparison with the greedy heuristic

Same 200 unseen seeds (20000–20199) for agent and heuristic; difference = agent − heuristic per seed; 95% CI = mean ± 1.96 standard errors. p_order 0.02 doubles the order rate and was never seen in training.

| Order rate | Algorithm | Config | Agent mean | Heuristic mean | Difference (95% CI) | Agent better on |
|---|---|---|---|---|---|---|
| 0.01 | DQN | D6 | 200.4 | 197.5 | +3.0 (-1.6 to +7.6) | 48% of seeds |
| 0.01 | REINFORCE | R9 | 192.6 | 197.5 | -4.9 (-10.0 to +0.3) | 41% of seeds |
| 0.01 | PPO | P4 | 190.0 | 197.5 | -7.4 (-12.5 to -2.4) | 37% of seeds |
| 0.01 | A2C | A3 | 191.0 | 197.5 | -6.4 (-11.6 to -1.2) | 39% of seeds |
| 0.02 | DQN | D6 | 285.0 | 225.1 | +59.9 (+51.4 to +68.5) | 81% of seeds |
| 0.02 | REINFORCE | R9 | 288.1 | 225.1 | +63.1 (+55.5 to +70.6) | 90% of seeds |
| 0.02 | PPO | P4 | 291.9 | 225.1 | +66.9 (+57.8 to +75.9) | 85% of seeds |
| 0.02 | A2C | A3 | 288.3 | 225.1 | +63.3 (+53.7 to +72.8) | 82% of seeds |
