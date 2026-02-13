# Baseline report

## Runs

- `forest` from `runs/forest/run_test.json`
- `forest+greedy` from `runs/forest/run_test_greedy.json`
- `forest` from `runs/forest/run_test_none.json`
- `graph_stat` from `runs/graph_stat/run_test_300.json`
- `graph_stat` from `runs/graph_stat/run_test.json`
- `relaxed_cube` from `runs/relaxed_cube/run_test.json`
- `random_feasible+greedy` from `runs/random_feasible/run_test_greedy.json`
- `random_feasible` from `runs/random_feasible/run_test_none.json`

## Metrics table

| Run | RMSE_xz | MAE_xz | BoundaryViolRate | CollisionPairRate |
|---|---:|---:|---:|---:|
| random_feasible+greedy | 4.969091 | 2.737146 | 0.000000 | 0.037595 |
| forest+greedy | 4.038779 | 1.359140 | 0.000000 | 0.054204 |
| graph_stat | 2.331413 | 2.002818 | 0.000000 | 0.078556 |
| relaxed_cube | 4.993223 | 2.833255 | 0.000000 | 0.093344 |
| graph_stat | 4.478688 | 2.291949 | 0.000000 | 0.099781 |
| random_feasible | 4.977421 | 2.749646 | 0.000000 | 0.146260 |
| forest | 3.912814 | 1.061572 | 0.000000 | 0.330501 |
| forest | 3.912814 | 1.061572 | 0.000000 | 0.330501 |

## Plot

- Scatter saved to `reports/baselines_all.png`
