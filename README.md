This repository contains the python code files that help implement logistic and linear regression in a distributed environment with Google Cloud. The compute nodes are Google Functions and the user PC acts as the master of the distributed computation task. Algorithms such as Time Straggling, ErasureHead[https://arxiv.org/abs/1901.09671] and Stochastic Gradient Coding[https://arxiv.org/abs/1905.05383] have beem implemented to tackle stragglers during distributed computation.