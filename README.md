# Distributed Machine Learning Project
This project was done with the guidance of Dr. Rawad Bitar of Technical University of Munich, Germany. The project was partially done with the support of the DAAD-WISE scholarship ([Link](https://www2.daad.de/deutschland/stipendium/datenbank/en/21148-scholarship-database/?detail=50015295)).

This repository contains the python code files that help implement logistic and linear regression in a distributed environment with Google Cloud. Datasets such as MNIST and CIFAR were used to run the logistic regression task. The compute _nodes_ are Google Cloud Functions and the user PC acts as the _master_ of the distributed computation task. Algorithms such as Time Straggling, ErasureHead ([Arxiv Link](https://arxiv.org/abs/1901.09671)) and Stochastic Gradient Coding ([Arxiv Link](https://arxiv.org/abs/1905.05383)) have been implemented to study their ability to tackle stragglers during distributed computation.
