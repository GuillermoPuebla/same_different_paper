# Repository for: Can deep convolutional neural networks support relational reasoning in the same-different task?

1. We provide the data of the problem #1 of SVRT in the "data" folder.
2. The script "generate_datasets.py" generates all the aditional data to replicate simulations 1 to 6.   
3. The script "generate_sort_of_clevr.py" generates the Sort-of-CLEVR dataset.
4. The script "generate_TUBerlin.py" downloads a local copy of the TU-Berlin dataset and generates the version that we used to pretrain the Resnet-50 model on Simulation 1.
5. To run "simulation_1_ResNet.py" it is necesarry to run "generate_TUBerlin.py" and "pretrain_ResNet_TUBerlin.py" first.
6. The script "benchmark_RN_sort_of_clevr.py" runs the benchmark presented in the Appendix. For this it is necesarry to run "generate_sort_of_clevr.py" first.
7. The rest of the simulations can be run individually.
