# RCCO: Rounding-based Containerized Chain Orchestration

## Requirements
```
Python 3.7
Gurobi 9
matplotlib-3.1.1
networkx-2.4
numpy-1.16.5
```

## Run
Use `main.py` to launch experiments.

```
python main.py --test-type batch --inter-arrival 1
python main.py --test-type optimal --inter-arrival 1 --scale-bw False
python main.py --test-type optimal --inter-arrival 1 --scale-bw True
python main.py --test-type layer --inter-arrival 1
python main.py --test-type share --inter-arrival 1
python main.py --test-type learning --inter-arrival 1
python main.py --test-type backtrack --inter-arrival 1
python main.py --test-type scaling --inter-arrival 1
```