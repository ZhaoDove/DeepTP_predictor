# DeepTP
DeepTP: A deep learning model for thermophilic protein prediction

# Dependency
Python 3.7

Tensorflow 2.4.0

Sklearn 0.22.2

NumPy 1.19.5

Pandas 1.1.5

# Document introduction
1. The dataset folder: 5 datasets used in the experiments (1) training dataset, biological features for training (2) independent balanced test set (3) independent unbalanced test set (4) validation set (5) homologous test set (similarity larger than 40%).

2. The example folder: the example contains all Biological features for training.

3. The feature_selection folder: RFECV model for feature_selection.

4. The prediction_model folder: deep learning model with CNN, BiLSTM and self-attention used for predict thermophilic protein.

5. The training_code folder: source code for model training.

6. The test_code folder: source code for independent test.

# How to use DeepTP
1. Pick the file example.csv from example folder, the example contains all Biological features for training, including AAC(Frequency of 20 amino acids), DPC(Frequency of 400 dipeptides), CTD(Composition, transition, and distribution), QSO(Distance matrix between 20 amino acids), PAAC(Pseudo-Amino Acid Composition) and APAAC(Amphiphilic Pseudo-Amino Acid Composition).

2. Run `python test.py`.
If you have other dataset file to test, please remember to modify the route and name of the data file with your current dataset (default, "example.csv") in test.py file before running.

3. The result including UniProt_id, predict_result and score will be saved in `../predict_result.csv`


If you have any problem, please contact us. (20204227057@stu.suda.edu.cn)