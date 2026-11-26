# Dispatcher behaviour per episode (held-out seeds 10000–10019, masked greedy)

*When an emergency and a routine order were both open at a FLY_DROP decision, the share of drops that went to the emergency. Fly-on drops = a second drop made without returning to the hub. Counts are per episode; battery at launch is in km (max 160).

| Policy | Deliveries | Emergencies | Expired | Wasted units | Sorties from hub | 2-unit sorties | Fly-on drops | Battery swaps | Battery at launch | Flights through storm | Emergency chosen first* | HOLDs | Invalid actions | Masking changed choice |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Greedy heuristic | 12.5 | 4.0 | 0.50 | 0.4 | 12.9 | 0% | 0.0 | 12.3 | 160 | 2% | 100% | 35.1 | 0.0 | 0.0 |
| DQN D6 | 13.4 | 4.2 | 0.20 | 3.5 | 15.1 | 46% | 1.9 | 6.5 | 131 | 2% | 94% | 18.6 | 0.0 | 0.0 |
| REINFORCE R9 | 12.5 | 4.1 | 1.20 | 0.7 | 11.1 | 55% | 2.1 | 8.0 | 148 | 2% | 72% | 49.8 | 0.0 | 56.6 |
| PPO P4 | 13.3 | 4.5 | 0.50 | 4.5 | 15.1 | 46% | 2.7 | 9.7 | 142 | 2% | 84% | 0.0 | 0.0 | 0.1 |
| A2C A3 | 13.2 | 4.2 | 0.50 | 4.5 | 13.2 | 97% | 4.6 | 9.2 | 146 | 2% | 89% | 0.0 | 0.0 | 0.2 |
