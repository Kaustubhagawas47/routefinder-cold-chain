| Method | Distance | Thermal | Cost | Gap to PyVRP |
|---|---|---|---|---|
| PyVRP (reference) | 12.70 | 0.001 | 12.70 | — |
| Fine-tuned (EAL) | 13.37 | 0.060 | 13.43 | +5.7% |
| PyVRP, distance only | 12.66 | 2.451 | 15.11 | +18.8% |
| One store per trip | 21.77 | 0.022 | 21.79 | +71.5% |
| Nearest neighbour | 19.41 | 7.733 | 27.14 | +113.9% |
| Before fine-tuning | 23.65 | 15.680 | 39.33 | +210.1% |

Late deliveries (% of stores past safe time, first 200 instances)

| Method | Frozen | Chilled | Tropical fruit | Chilled veg. |
|---|---|---|---|---|
| Before fine-tuning | 21.4 | 4.9 | 0.7 | 2.5 |
| Fine-tuned (EAL) | 0.3 | 0.0 | 0.0 | 0.0 |
| PyVRP (reference) | 0.0 | 0.0 | 0.0 | 0.0 |

Trips by number of stores (50 km x 50 km grid; trips per instance, mean trip km in brackets; trip = depot -> stores -> depot)

| Method | 1-stop | 2-stop | 3-stop | All trips | Stores/trip | Mean trip km | Mean store-to-store km |
|---|---|---|---|---|---|---|---|
| Before fine-tuning | 0.62 (38.0) | 16.99 (50.2) | 5.14 (59.7) | 22.74 | 2.20 | 52.0 | 8.5 |
| Nearest neighbour | 5.53 (23.1) | 17.15 (38.9) | 3.39 (51.8) | 26.07 | 1.92 | 37.2 | 5.2 |
| Fine-tuned (EAL) | 3.10 (15.6) | 20.22 (27.3) | 2.15 (31.6) | 25.48 | 1.96 | 26.2 | 4.8 |
| PyVRP (reference) | 0.90 (7.7) | 17.46 (26.8) | 4.72 (34.3) | 23.09 | 2.17 | 27.6 | 6.4 |

PyVRP row: first 200 test instances (re-solved at 2 s); other rows: all test instances.
