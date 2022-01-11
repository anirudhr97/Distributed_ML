"""
Code to compare the results obtained from 2 runs. This code is used for analysis and has
nothing to do with the functioning of algorithms or the distributed computing job.
"""
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_dataset(filename, if_ones):
    """
    Function to load the csv into the program.
    ###
    Arguments:
    filename      -- filename of the csv
    ###
    Return:
    Dict          -- 'data': dataset_1
        dataset_1: numpy array of the csv
    """
    dataset_df = pd.read_csv(filename, header=None)
    dataset_1 = dataset_df.to_numpy()
    if if_ones:
        data_ = np.ones((dataset_1.shape[0], dataset_1.shape[1] + 1))
        data_[:, 1:] = dataset_1
    else:
        data_ = dataset_1
    return {"data": data_}


########################################################################

EH_DIR = "task-log-data-mnist_train-test-mnist_test-nod-200-it-70-choice-FRC-c-600-alg-ErasHead-d-1-ivl-True-ident-6799"
SGC_DIR = "task-log-data-mnist_train-test-mnist_test-nod-200-it-70-choice-StocGD-c-600-alg-StocGD-d-2-ivl-True-ident-8282"
NO_OF_ITER = 70

dict_eh = {}
dict_sgc = {}
eh_times = []
sgc_times = []
eh_acc_train = load_dataset("./{}/acc_transition.csv".format(EH_DIR), 0)["data"]
sgc_acc_train = load_dataset("./{}/acc_transition.csv".format(SGC_DIR), 0)["data"]
eh_acc_test = load_dataset("./{}/test_acc_transition.csv".format(EH_DIR), 0)["data"]
sgc_acc_test = load_dataset("./{}/test_acc_transition.csv".format(SGC_DIR), 0)["data"]


with open("./{}/all_time_data_dict.csv".format(EH_DIR)) as csv_file:
    read = csv.reader(csv_file)
    for row in read:
        if len(row):
            dict_eh[row[0]] = row[1]
with open("./{}/all_time_data_dict.csv".format(SGC_DIR)) as csv_file:
    read = csv.reader(csv_file)
    for row in read:
        if len(row):
            dict_sgc[row[0]] = row[1]
for i in range(NO_OF_ITER + 1):
    if i == 0:
        eh_times.append(float(dict_eh["init_time"]))
        sgc_times.append(float(dict_sgc["init_time"]))
    else:
        eh_times.append(eh_times[-1] + float(dict_eh["http_iter_{}_wout".format(i)]))
        sgc_times.append(sgc_times[-1] + float(dict_sgc["http_iter_{}_wout".format(i)]))
plt.figure()
plt.plot(eh_times, eh_acc_train, "r", linestyle="dotted", label="ErasureHead train")
plt.plot(eh_times, eh_acc_test, "r", linestyle="solid", label="ErasureHead test")
plt.plot(sgc_times, sgc_acc_train, "b", linestyle="dotted", label="SGC train")
plt.plot(sgc_times, sgc_acc_test, "b", linestyle="solid", label="SGC test")
plt.title("Accuracy vs Time (ErasureHead & SGC - d=2)")
plt.xlabel(r"Time(s) $\rightarrow$")
plt.ylabel(r"Accuracy(%) $\rightarrow$")
plt.grid()
plt.legend()
plt.savefig("comparison_plot_redun=2.png")
plt.show()


################################################################################
###	ROUGH WORK
################################################################################
# print(dict_eh)
# with open('coors.csv', mode='r') as infile:
#     reader = csv.reader(infile)
#     with open('coors_new.csv', mode='w') as outfile:
#         writer = csv.writer(outfile)
#         mydict = {rows[0]:rows[1] for rows in reader}
