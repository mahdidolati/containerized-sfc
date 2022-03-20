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

## License
Copyright 2022 Mahdi Dolati.

The project's source code are released here under the [GNU Affero General Public License v3](https://www.gnu.org/licenses/agpl-3.0.html). In particular,
- You are entitled to redistribute the program or its modified version, however you must also make available the full source code and a copy of the license to the recipient. Any modified version or derivative work must also be licensed under the same licensing terms.
- You also must make available a copy of the modified program's source code available, under the same licensing terms, to all users interacting with the modified program remotely through a computer network.

(TL;DR: you should also open-source your derivative work's source code under AGPLv3.)
