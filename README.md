# Distributed Machine Learning Project
This project was done under Dr. Rawad Bitar of Technical University of Munich, Germany. The project was done under the DAAD-WISE scholarship.

This repository contains the python code files that help implement logistic and linear regression in a distributed environment with Google Cloud. Datasets such as MNIST and CIFAR were used to run the logistic regression task. The compute nodes are Google Functions and the user PC acts as the master of the distributed computation task. Algorithms such as Time Straggling, ErasureHead[https://arxiv.org/abs/1901.09671] and Stochastic Gradient Coding[https://arxiv.org/abs/1905.05383] have beem implemented to tackle stragglers during distributed computation.
