# import struct
# import math
# from types import LambdaType
# from numpy.core.fromnumeric import var

'''
Learning Rates that work well:
1) MNIST      :   0.025 with variable lrate with 0.013 exponent; factor=1
2) CIFAR-10   :   
3) CIFAR-100  :   
'''

import sys
from google.cloud import storage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time
import concurrent.futures
import requests
import os
import csv
import logging
from math import comb
from scipy.special import expit as sigmoid
from scipy.special import softmax as softmax_scipy

logging.basicConfig(filename='Messages.log', filemode='w',format='%(asctime)s.%(msecs)03d :  %(message)s', datefmt='%I:%M:%S', level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
time_start_dict = {}
acc_transition = []
test_acc_transition = []
no_of_unique_labels = 0

def plot_confusion_matrix(y_true, y_pred, iterno, folder_name, choice, if_labels=False):
  '''
  Takes in predicted and ground truth values and plots confusion matrix
  #####
  Arguments:
  y_true      - Ground truth labels
  y_pred      - Predicted labels
  if_labels   - default False
  ######
  Return:
  None
  '''
  #labels = np.unique(y_true)
  #num_labels = len(labels)
  
  cf_mat = confusion_matrix(y_true, y_pred)

  if choice == 'train':
    print('\nTrain --- Iter No:{0}   Accuracy: {1:.5f}%'.format(iterno+1, 100*np.trace(cf_mat)/np.sum(np.sum(cf_mat)) ))
    acc_transition.append(100*np.trace(cf_mat)/np.sum(np.sum(cf_mat)))
  elif choice == 'test':
    print('\nTest ---- Iter No:{0}   Accuracy: {1:.5f}%'.format(iterno+1, 100*np.trace(cf_mat)/np.sum(np.sum(cf_mat)) ))
    test_acc_transition.append(100*np.trace(cf_mat)/np.sum(np.sum(cf_mat)))

  class_names = []
  for i in range(no_of_unique_labels):
    class_names.append(str(i))

  if if_labels:
    df_cf = pd.DataFrame(data=cf_mat, index=class_names, columns=class_names)
  else:
    df_cf = pd.DataFrame(data=cf_mat)

  df_cf.index.name, df_cf.columns.name = 'True Label', 'Predicted Label'
  plt.figure(figsize=(5, 5))
  sns.heatmap(df_cf, annot=True, fmt='g')
  plt.savefig('./{}/{}_conf_mat_iter_{}.png'.format(folder_name, choice, iterno+1), bbox_inches='tight')
  #plt.show()
  plt.close()

  return cf_mat

class Misc:
  @staticmethod
  def load_dataset(filename, if_ones):
    '''
    Function to load the csv into the program.
    ###
    Arguments:
    filename      -- filename of the csv
    ###
    Return:
    Dict          -- 'data': dataset_1
        dataset_1: numpy array of the csv
    '''
    dataset_df = pd.read_csv(filename, header=None)
    dataset_1 = dataset_df.to_numpy()
    if if_ones:
      data_ = np.ones((dataset_1.shape[0],dataset_1.shape[1]+1)); data_[:,1:] = dataset_1
    else:
      data_ = dataset_1

    return {
        'data': data_
    }


def blobber(gcs_filename, inst_path, buckett, choice):
  '''
    Function to upload/download files to/from gcloud storage bucket.
    ###
    Arguments:
    gcs_filename  -- Filename of the file in the gcloud bucket
    inst_path     -- Path to the file in local machine
    buckett       -- Bucket in gcloud storage
    choice        -- 2 options:
                        -- 'upload': To upload file from 'inst_path' to 'gcs_filname' in 'buckett'
                        -- 'download': To download file from 'gcs_filname' in 'buckett' to 'inst_path'
    ###
    Return:
    -
  '''
  if choice == 'download':
    blob = buckett.blob(gcs_filename)
    blob.download_to_filename(inst_path)
  elif choice == 'upload':
    blob = buckett.blob(gcs_filename)
    blob.upload_from_filename(inst_path)

def load_url(url, timeout):
  '''
  Function to make head request to 'url' with timeout
  ###
  Arguments:
  url             -- URL
  timeout         -- Duration in seconds given to timeout argument of requests.head()
  ###
  Return:
  status_code     -- Status code of the request  
  '''

  time_start_dict[url.split('?')[0].split('-')[-1]] = time.time()
  ans = requests.head(url, timeout=timeout)
  return ans.status_code

def logit(message):
  logging.info('LOG:  %s'%message)

#################################################################################################################

def sigm(result):
  final_result = 1/(1+np.exp(-result))
  return final_result

def softm(vector):
  e = np.exp(vector)
  return e / np.sum(e, axis=1, keepdims=True)

def softmax(vector):
  return softmax_scipy(vector, axis=1)

def compute_alpha1(iteration_number):
  step = 1 / (iteration_number ** 0.8)
  return step

def compute_alpha(iteration_number):
  step = 1 / (iteration_number ** 0.15)
  return step

#################################################################################################################
#################################################################################################################
#################################################################################################################

def master(dataset_name, test_dset, no_of_nodes, no_of_iter, lrate, strag, choice, c, factor, connections, timeout, algo, red, if_ones, if_custom_init_wt, if_change_assgn, if_verbose, if_variable_lrate):

  # We store starting time to calculate execution time for master() at the end
  time_start = time.time()

  # Assigning the arguments obtained to variables that will be used throughout the function
  CONNECTIONS = int(connections)
  TIMEOUT = timeout
  NO_OF_NODES = int(no_of_nodes)
  DATASET_NAME = dataset_name
  TEST_DATASET = test_dset
  NO_OF_ITER = int(no_of_iter)
  LRATE = lrate
  CHOICE = choice
  STRAG = strag
  c = int(c)
  red = int(red)
  FACTOR = float(factor)
  IF_ONES = int(if_ones)
  if algo == 'percent_strag' or algo == 'ErasHead' or algo == 'StocGD':
    THRESHOLD = int(STRAG*NO_OF_NODES/100)
    VALID_COUNT = NO_OF_NODES - THRESHOLD

  # Creating a folder to store all the collected data in this run
  cwd = os.path.abspath(os.getcwd())
  # folder_name = 'task-{}-data-{}-test-{}-nod-{}-it-{}-lr-{}-str-{}-choice-{}-c-{}-fac-{}-alg-{}-d-{}-io-{}-iciw-{}-ica-{}-iv-{}-ivl-{}-ident-{}'.format('log', DATASET_NAME.strip('.zip'), TEST_DATASET.strip('.zip'), NO_OF_NODES, NO_OF_ITER, LRATE, STRAG, CHOICE, c, FACTOR, algo, red, if_ones, if_custom_init_wt, if_change_assgn, if_verbose, if_variable_lrate, np.random.random())
  folder_name = 'task-{}-data-{}-test-{}-nod-{}-it-{}-choice-{}-c-{}-alg-{}-d-{}-ivl-{}-ident-{}'.format('log', DATASET_NAME.strip('.zip'), TEST_DATASET.strip('.zip'), NO_OF_NODES, NO_OF_ITER, CHOICE, c, algo, red, if_variable_lrate, np.random.randint(0, high=10000))
  print('\nStoring files for this execution in the following directory: \n{}\n'.format(folder_name))
  path = os.path.join(cwd, folder_name)
  os.mkdir(path)

  storage_client = storage.Client()
  bucket1 = storage_client.bucket('data_tum_base')
  bucket2 = storage_client.bucket('data_tum_master-to-node')
  bucket3 = storage_client.bucket('data_tum_node-to-master')
  ####################################################################################################################

  # Printing and logging the relevant information before the run
  print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
  print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
  print('\tDataset Name ---- ', DATASET_NAME)
  print('\tTest Dataset ---- ', TEST_DATASET)
  print('\tNumber of Iters - ', NO_OF_ITER)
  print('\tNumber of Nodes - ', NO_OF_NODES)
  print('\tChoice ---------- ', CHOICE)
  print('\tLearning Rate --- ', LRATE)
  print('\tStrag ----------- ', STRAG)
  if algo == 'percent_strag' or algo == 'StocGD':
    print('\tThreshold ------- ', THRESHOLD)
    print('\tNet nodes used -- ', VALID_COUNT)
  elif algo == 'ErasHead':
    print('\tThreshold ------- ', THRESHOLD)
    print('\tMax nodes used -- ', VALID_COUNT)
  print('\tFactor ---------- ', FACTOR)
  print('\tConnections ----- ', CONNECTIONS)
  print('\tTimeout --------- ', TIMEOUT)
  print('\tAdd 1s column --- ', IF_ONES)
  print('\tAlgorithm ------- ', algo)
  print('\tRedundancy(d) --- ', red)
  print('\tCustom init wt -- ', if_custom_init_wt)
  print('\tChange Asgn Mat - ', if_change_assgn)
  print('\tVerbosity ------- ', if_verbose)
  print('\tVariable lrate -- ', if_variable_lrate)
  logit('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
  logit('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
  logit('\tDataset Name ---- %s' %DATASET_NAME)
  logit('\tTest Dataset ---- %s' %TEST_DATASET)
  logit('\tNumber of Iters - %s' %NO_OF_ITER)
  logit('\tNumber of Nodes - %s' %NO_OF_NODES)
  logit('\tChoice ---------- %s' %CHOICE)
  logit('\tLearning Rate --- %s' %LRATE)
  logit('\tStrag ----------- %s' %STRAG)
  if algo == 'percent_strag' or algo == 'StocGD':
    logit('\tThreshold ------- %s' %THRESHOLD)
    logit('\tNet nodes used -- %s' %VALID_COUNT)
  elif algo == 'ErasHead':
    logit('\tThreshold ------- %s' %THRESHOLD)
    logit('\tMax nodes used -- %s' %VALID_COUNT)
  logit('\tFactor ---------- %s' %FACTOR)
  logit('\tConnections ----- %s' %CONNECTIONS)
  logit('\tTimeout --------- %s' %TIMEOUT)
  logit('\tAdd 1s column --- %s' %IF_ONES)
  logit('\tAlgorithm ------- %s' %algo)
  logit('\tRedundancy(d) --- %s' %red)
  logit('\tCustom init wt -- %s' %if_custom_init_wt)
  logit('\tChange Asgn Mat - %s' %if_change_assgn)
  logit('\tVerbosity ------- %s' %if_verbose)
  logit('\tVariable lrate -- %s' %if_variable_lrate)
  ##############################################################################################################

  # Downloading the assignment matrix and dataset for cost calculation
  blobber(DATASET_NAME, './{}/dataset.zip'.format(folder_name), bucket1, 'download')
  data_ = Misc.load_dataset('./{}/dataset.zip'.format(folder_name), IF_ONES)['data']
  X = data_[:, :-1]
  Y = data_[:, -1]

  labels = np.unique(Y)
  no_of_unique_labels = labels.shape[0]
  if labels.shape[0] == 2:
    VARIETY = 'logistic'
  elif labels.shape[0] > 2:
    VARIETY = 'multi-logistic'

  # Downloading the test dataset
  blobber(TEST_DATASET, './{}/test_dataset.zip'.format(folder_name), bucket1, 'download')
  data_ = Misc.load_dataset('./{}/test_dataset.zip'.format(folder_name), IF_ONES)['data']
  Xtest = data_[:, :-1]
  Ytest = data_[:, -1]

  ##############################################################################################################

  # Making the initiator call and logging, printing relevant information
  if if_verbose:
    # Initiating by calling the initiator cloud function:
    URL_INIT = 'http://asia-south1-phonic-command-314914.cloudfunctions.net/function-init?dataset_name={}&no_of_iter={}&no_of_nodes={}&choice={}&if_ones={}&c={}&factor={}&variety={}&d={}'.format(DATASET_NAME, 0, NO_OF_NODES, CHOICE, IF_ONES, c, FACTOR, VARIETY, red)
    time1 = time.time()
    r = requests.post(URL_INIT)
    print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
    print('\tInitiator: ')
    print('\n---- URL         : ', r.url)
    print('---- Status Code : ', r.status_code, '\n---- Reason      : ', r.reason)
    print('---- Text        : ', r.text)
    logit('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    logit('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    logit('\tInitiator: ')
    logit('---- URL         : %s'%r.url)
    logit('---- Status Code : %s'%r.status_code)
    logit('---- Reason      : %s'%r.reason)
    logit('---- Text        : %s'%r.text)

    time2 = time.time()
    print('---- Time taken  :  {0:.5f}s'.format(time2-time1))
    logit('---- Time taken  :  %ss'%(time2-time1))
    sum_time = time2-time1
    http_time = time2-time1
  else:
    # Initiating by calling the initiator cloud function:
    URL_INIT = 'http://asia-south1-phonic-command-314914.cloudfunctions.net/function-init?dataset_name={}&no_of_iter={}&no_of_nodes={}&choice={}&if_ones={}&c={}&factor={}&variety={}&d={}'.format(DATASET_NAME, 0, NO_OF_NODES, CHOICE, IF_ONES, c, FACTOR, VARIETY, red)
    time1 = time.time()
    r = requests.post(URL_INIT)
    print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
    print('\tInitiator: ')
    print('\n---- Status Code : ', r.status_code, '\n---- Reason      : ', r.reason)
    logit('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    logit('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    logit('\tInitiator: ')
    logit('---- Status Code : %s'%r.status_code)
    logit('---- Reason      : %s'%r.reason)

    time2 = time.time()
    print('---- Time taken  :  {0:.5f}s'.format(time2-time1))
    logit('---- Time taken  :  %ss'%(time2-time1))
    sum_time = time2-time1
    http_time = time2-time1

  all_time_data_dict = {}
  all_time_data_dict['init_time'] = time2 - time1

  # Uncomment if delay after intiator() call is necessary
  #  time.sleep(5)
  ####################################################################################################################

  # Provision to use custom initial weights. Custom initial weight needs to be in 'initial_wt.csv' in the same directory.
  if if_verbose:
    if if_custom_init_wt:
      try:
        blobber('weights_iter{}.csv'.format(0), 'initial_wt.csv', bucket1, 'upload')
        print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
        print('---- \"initial_wt.csv\" found. Proceeding with algorithm using custom initial weights.')
        logit('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        logit('---- \"initial_wt.csv\" found. Proceeding with algorithm using custom initial weights.')
        initial_wt = Misc.load_dataset('initial_wt.csv', 0)['data']
        print('---- Initial weights used: \n', initial_wt)
        logit('---- Initial weights used: %s'%initial_wt)
      except:
        print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
        print('---- \"initial_wt.csv\" not found. Proceeding with algorithm using randomly initialized weights.')
        logit('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        logit('---- \"initial_wt.csv\" not found. Proceeding with algorithm using randomly initialized weights.')
        blobber('weights_iter{}.csv'.format(0), 'initial_wt.csv', bucket1, 'download')
        initial_wt = Misc.load_dataset('initial_wt.csv', 0)['data']
        print('---- Initial weights used: \n', initial_wt)
        logit('---- Initial weights used: %s'%initial_wt)
  else:
    if if_custom_init_wt:
      try:
        blobber('weights_iter{}.csv'.format(0), 'initial_wt.csv', bucket1, 'upload')
        print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
        print('---- \'initial_wt.csv\' found. Proceeding with algorithm using custom initial weights.')
        logit('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        logit('---- \'initial_wt.csv\' found. Proceeding with algorithm using custom initial weights.')        
        #initial_wt = Misc.load_dataset('initial_wt.csv', 0)['data']
      except:
        print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
        print('---- \'initial_wt.csv\' not found. Proceeding with algorithm using randomly initialized weights.')
        logit('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        logit('---- \'initial_wt.csv\' not found. Proceeding with algorithm using randomly initialized weights.')        
        blobber('weights_iter{}.csv'.format(0), 'initial_wt.csv', bucket1, 'download')
        #initial_wt = Misc.load_dataset('initial_wt.csv', 0)['data']

  #######################################################################################################################
  #######################################################################################################################

  # RUNNING THE LOOP

  # List to hold the weights obtained, confusion matrices and cost after every iteration
  wt_transition = []
  cost_transition = []
  conf_mat_transition = []
  test_conf_mat_transition = []

  blobber('assgn_matrix.csv', './{}/assgn_matrix.csv'.format(folder_name), bucket2, 'download')

  # List to hold number of nodes used in each iteration while doing algo = 'time_strag'
  nodes_num_used = []

  iterno = 0
  while iterno < NO_OF_ITER:

    # Gathering URLs of all the nodes in google cloud functions. Printing and logging relevant information
    urls = ['https://asia-south1-phonic-command-314914.cloudfunctions.net/function-n{}?dataset_name=dataset_{}.csv&iter_no={}&task={}&n_labels={}'.format((i+51), (i+1), iterno, VARIETY, no_of_unique_labels) for i in range(NO_OF_NODES)]
    print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('\n\tIteration number: {}\n'.format(iterno+1))
    logit('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    logit('\tIteration number: %s'%(iterno+1))

    # Dictionary to hold each node against time it took to finish execution.(format --> node:time(in s))
    time_dict = {}

    # To be completed as required for the task we are implementing. Provision to change assign matrix
    if if_change_assgn:
      pass

    # Calling all the nodes simultaneously using concurrent.futures
    if if_verbose:
      with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as executor:
        #time1 = time.time()
        future_to_url = {executor.submit(load_url, url, TIMEOUT):url for url in urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
          timex = time.time()
          temp = future_to_url[future]
          temp1 = temp.split('?')[0].split('-')[-1]
          try:
            data = future.result()
            print('---- The outcome of \"{0}\" is \"{1}\". \nCompleted in {2:.5f}s.\n'.format(temp1, data, timex-time_start_dict[temp1]))
            logit('---- The outcome of \"%s\" is \"%s\". Completed in %ss.'%(temp1, data, timex-time_start_dict[temp1]))
            time_dict[temp1] = timex-time_start_dict[temp1]
          except Exception as exc:
            data = str(exc)
            print('---- Error encountered in \"{0}\"" : \"{1}\". \nCompleted in {2:.5f}s.\n'.format(temp1, data, timex-time_start_dict[temp1]))
            logit('---- Error encountered in \"%s\"" : \"%s\". Completed in %ss.'%(temp1, data, timex-time_start_dict[temp1]))
            time_dict[temp1] = timex-time_start_dict[temp1]

      time_dict = {k: v for k, v in sorted(time_dict.items(), key=lambda item: item[1])}
      http_time = http_time + list(time_dict.values())[-1]
      print('#####   Total time taken for HTTP calls: {0:.5f}s   #####\n'.format(list(time_dict.values())[-1]))
      print('---- Dict of time taken for the nodes:\n', time_dict)
      logit('#####   Total time taken for HTTP calls: %ss   #####'%(list(time_dict.values())[-1]))
      logit('---- Dict of time taken for the nodes: %s'%time_dict)
    else:
      with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as executor:
        #time1 = time.time()
        future_to_url = {executor.submit(load_url, url, TIMEOUT):url for url in urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
          timex = time.time()
          temp = future_to_url[future]
          temp1 = temp.split('?')[0].split('-')[-1]
          try:
            data = future.result()
            time_dict[temp1] = timex-time_start_dict[temp1]
          except Exception as exc:
            data = str(exc)
            time_dict[temp1] = timex-time_start_dict[temp1]

      time_dict = {k: v for k, v in sorted(time_dict.items(), key=lambda item: item[1])}
      http_time = http_time + list(time_dict.values())[-1]
      print('#####   Total time taken for HTTP calls: {0:.5f}s   #####'.format(list(time_dict.values())[-1]))
      logit('#####   Total time taken for HTTP calls: %ss   #####'%(list(time_dict.values())[-1]))

    all_time_data_dict['http_iter_{}_w'.format(iterno+1)] = list(time_dict.values())[-1]
    # Saving the time_dict for this iteration
    with open("./{}/time_dict_iterno_{}.csv".format(folder_name, iterno+1), 'w') as csv_file:  
      writer = csv.writer(csv_file)
      for key, value in time_dict.items():
        writer.writerow([key, value])    

    #################
    # To read back into dict
    # with open('dict.csv') as csv_file:
    #   reader = csv.reader(csv_file)
    #   mydict = dict(reader)
    #################
    ########################################################################################################################

    # # Inverting time_dict for use
    # inv_dict = {v: k for k, v in time_dict.items()}

    # # Compiling the list of nodes whose outputs will be used in the weight update
    # if algo == 'time_strag':
    #   valid_nodes = []
    #   for x in sorted(time_dict.values()):
    #     if float(x) < STRAG:
    #       valid_nodes.append(inv_dict[x])
    #   print('---- We will be using gradients from ({}/{}) nodes in this iteration.'.format(len(valid_nodes), NO_OF_NODES))
    #   logit('---- We will be using gradients from (%s/%s) nodes in this iteration.'%(len(valid_nodes), NO_OF_NODES))
    #   nodes_num_used.append(float(len(valid_nodes)))
    # elif algo == 'percent_strag':
    #   temp1 = sorted(time_dict.values())[:VALID_COUNT]
    #   valid_nodes = []
    #   for x in temp1:
    #     valid_nodes.append(inv_dict[x])      
    # elif algo == 'ErasHead':
    #   n = np.shape(X)[0]
    #   l = NO_OF_NODES*c//n
    #   p = comb(NO_OF_NODES-l, VALID_COUNT)/comb(NO_OF_NODES, VALID_COUNT)
    #   if iterno == 0:
    #     LRATE = LRATE/(1-p)
    #   register = np.zeros((n//c,))
    #   temp = sorted(time_dict.values())
    #   count = 0
    #   valid_nodes = []
    #   for x in temp:
    #     if count < VALID_COUNT and np.sum(register) < n//c:
    #       if register[(int(inv_dict[x].strip('n'))-1)//l] == 0:
    #         valid_nodes.append(inv_dict[x])
    #         register[(int(inv_dict[x].strip('n'))-1)//l] = 1
    #     else:
    #       break
    #     count = count+1
    #########################################################################################################

    # Compiling the list of nodes whose outputs will be used in the weight update
    if algo == 'time_strag':
      valid_nodes = []
      for x in time_dict.keys():
        if float(time_dict[x]) < STRAG:
          valid_nodes.append(x)
      print('---- We will be using gradients from ({}/{}) nodes in this iteration.'.format(len(valid_nodes), NO_OF_NODES))
      logit('---- We will be using gradients from (%s/%s) nodes in this iteration.'%(len(valid_nodes), NO_OF_NODES))
      nodes_num_used.append(float(len(valid_nodes)))

    elif algo == 'percent_strag' or algo == 'StocGD':
      valid_nodes = []
      valid_nodes = list(time_dict.keys())[:VALID_COUNT]
      
    elif algo == 'ErasHead':
      n = np.shape(X)[0]
      l = NO_OF_NODES*c//n
      p = comb(NO_OF_NODES-l, VALID_COUNT)/comb(NO_OF_NODES, VALID_COUNT)
      if iterno == 0:
        LRATE = LRATE/(1-p)
      register = np.zeros((n//c,))
      count = 0
      valid_nodes = []
      for x in time_dict.keys():
        if count < VALID_COUNT and np.sum(register) < n//c:
          if register[(int(x.strip('n'))-51)//l] == 0:
            valid_nodes.append(x)
            register[(int(x.strip('n'))-51)//l] = 1
        else:
          break
        count = count+1

    else:
      print('algo given not valid! Exiting...')
      sys.exit()

    all_time_data_dict['http_iter_{}_wout'.format(iterno+1)] = time_dict[valid_nodes[-1]]
    # Saving the valid nodes for this iteration into csv
    with open("./{}/valid_nodes_iterno_{}.csv".format(folder_name, iterno+1), 'w') as f:    
      write = csv.writer(f)
      write.writerow(valid_nodes)


    # Printing and logging information regarding node outputs that will actually be used for weight update
    print('\n---- Time spent on nodes(excluding stragglers) in this iteration: {0:.5f}s'.format(time_dict[valid_nodes[-1]]))
    print('---- The nodes whose gradient outputs will be used for the weight update:\n {}\n'.format(valid_nodes))
    logit('---- Time spent on nodes(excluding stragglers) in this iteration: %ss'%(time_dict[valid_nodes[-1]]))
    logit('---- The nodes whose gradient outputs will be used for the weight update: %s'%(valid_nodes))

    sum_time = sum_time + time_dict[valid_nodes[-1]]
    ########################################################################################################################

    # Gathering all the gradient files that were returned by nodes in this iteration in 'grads'.
    grads = storage_client.list_blobs('data_tum_node-to-master', prefix='grad_iterno_{}_'.format(iterno))

    # List to store the filenames of all the gradient files that are downloaded in this iteration
    grad_downloads = []

    # Downloading the gradient files from valid_nodes only.
    if if_verbose:
      for x in grads:
        temp = x.name
        print('Available: ', temp)
        logit('Available: %s'%temp)
        if temp.split('_')[3] in valid_nodes:
          print('Downloading...\n')
          logit('Downloading... %s'%temp)
          grad_downloads.append(x.name)
          blobber(x.name, x.name, bucket3, 'download')
        else:
          print('Not downloading.\n')
          logit('Not downloading %s'%temp)          

      print('\n---- We use the following gradients to update the weights: \n{}'.format(grad_downloads))
      logit('---- We use the following gradients to update the weights: %s'%(grad_downloads))
    else:
      for x in grads:
        temp = x.name
        if temp.split('_')[3] in valid_nodes:
          grad_downloads.append(x.name)
          blobber(x.name, x.name, bucket3, 'download')
    
    # Updating the weight
    blobber('weights_iter{}.csv'.format(iterno), 'wt.csv', bucket1, 'download')
    wt = Misc.load_dataset('wt.csv', 0)['data']
    wt_transition.append(wt)

    if iterno == 0:
      if VARIETY == 'logistic':
        Yhat = sigmoid(np.dot(X,wt))
        cost_transition.append(np.sum(-np.multiply(np.log10(Yhat), np.reshape(Y,(-1, 1))) -np.multiply( np.log10(1-Yhat),1-np.reshape(Y,(-1, 1))))/np.shape(X)[0])
        Y_prediction = np.round(Yhat)
        conf_mat_transition.append(plot_confusion_matrix(Y, Y_prediction, -1, folder_name, 'train'))
        Y_test_prediction = np.round(sigmoid(np.dot(Xtest,wt)))
        test_conf_mat_transition.append(plot_confusion_matrix(Ytest, Y_test_prediction, -1, folder_name, 'test'))
      
      elif VARIETY == 'multi-logistic':
        temp = softmax(np.dot(X,wt))
        cost = 0
        for i in range(no_of_unique_labels):
          idx = np.where(Y == i)
          Yhat = np.squeeze(temp[idx, i])
          cost += np.sum(-np.log10(Yhat))
        
        cost_transition.append(cost/np.shape(X)[0])
        Y_prediction = np.argmax(temp, axis=1)
        conf_mat_transition.append(plot_confusion_matrix(Y, Y_prediction, -1, folder_name, 'train'))
        Y_test_prediction = np.argmax(softmax(np.dot(Xtest, wt)), axis=1)
        test_conf_mat_transition.append(plot_confusion_matrix(Ytest, Y_test_prediction, -1, folder_name, 'test'))

    wt_new = wt
    for blob in grad_downloads:
      grad = Misc.load_dataset(blob, 0)['data']    
      if algo == 'StocGD':
        if if_variable_lrate:
          wt_new = wt_new - ( LRATE*compute_alpha(iterno+1)/(red*(1-THRESHOLD/NO_OF_NODES)) )*grad
        else:
          wt_new = wt_new - ( LRATE/(red*(1-THRESHOLD/NO_OF_NODES)) )*grad

      else:
        if if_variable_lrate:
          wt_new = wt_new - LRATE*compute_alpha(iterno+1)*grad  
        else:
          wt_new = wt_new - LRATE*grad
    

    if VARIETY == 'logistic':
      Y_prediction = np.round(sigmoid(np.dot(X, wt_new)))
      conf_mat_transition.append(plot_confusion_matrix(Y, Y_prediction, iterno, folder_name, 'train'))
      Y_test_prediction = np.round(sigmoid(np.dot(Xtest, wt_new)))
      test_conf_mat_transition.append(plot_confusion_matrix(Ytest, Y_test_prediction, iterno, folder_name, 'test'))

    elif VARIETY == 'multi-logistic':
      temp = softmax(np.dot(X,wt_new))
      Y_prediction = np.argmax(temp, axis=1)
      conf_mat_transition.append(plot_confusion_matrix(Y, Y_prediction, iterno, folder_name, 'train'))
      Y_test_prediction = np.argmax(softmax(np.dot(Xtest, wt_new)), axis=1)
      test_conf_mat_transition.append(plot_confusion_matrix(Ytest, Y_test_prediction, iterno, folder_name, 'test'))

    print('\nUPDATED WEIGHT IN THIS ITERATION: \n{}'.format(wt_new))
    logit('UPDATED WEIGHT IN THIS ITERATION: %s'%(wt_new))

    # Calculating cost with the updated weight and printing, logging.
    if VARIETY == 'logistic':
      Yhat = sigmoid(np.dot(X, wt_new))
      cost = np.sum(-np.multiply(np.log10(Yhat), np.reshape(Y,(-1, 1))) -np.multiply( np.log10(1-Yhat),1-np.reshape(Y,(-1, 1))))/np.shape(X)[0]
      cost_transition.append(cost)
    elif VARIETY == 'multi-logistic':
      temp = softmax(np.dot(X,wt_new))
      cost = 0
      for i in range(no_of_unique_labels):
        idx = np.where(Y == i)
        Yhat = temp[idx, i]
        cost += np.sum(-np.log10(Yhat))
      cost_transition.append(cost/np.shape(X)[0])

    print('\nCOST AFTER THIS ITERATION: {0:.5f}'.format(cost))
    logit('COST AFTER THIS ITERATION: %s'%(cost))    

    # Uploading the updated weight
    np.savetxt("wt_new.csv", wt_new, delimiter=",")
    blobber('weights_iter{}.csv'.format(iterno+1), 'wt_new.csv', bucket1, 'upload')

    # Deleting the gradient csv files that we downloaded for this iteration
    if if_verbose:
      print('\nDeleting the gradient files downloaded during this iteration...')
      logit('Deleting the gradient files downloaded during this iteration...')
    for grad_file in grad_downloads:
      os.remove(grad_file)

    # Incrementing iteration number
    iterno = iterno+1

  ############################################################################################################
  ############################################################################################################

  # Deleting the 2 temporary files used to perform weight updates and the initial weight file.
  os.remove('wt.csv')
  os.remove('wt_new.csv')
  os.remove('./{}/test_dataset.zip'.format(folder_name))
  os.remove('./{}/dataset.zip'.format(folder_name))

  if if_custom_init_wt:
    os.replace(os.path.join(cwd, 'initial_wt.csv'), os.path.join(path, 'initial_wt.csv'))

  if algo == 'time_strag':
    np.savetxt('./{}/time_strag-num_nodes_used.csv'.format(folder_name), np.array(nodes_num_used), delimiter=",")

  # Printing and logging some relevant time information and the transition of weights.
  print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
  logit('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
  wt_transition.append(wt_new)
  time_end = time.time()

  for i in range(NO_OF_ITER+1):
    np.savetxt('./{}/conf_mat_iter_{}.csv'.format(folder_name, i), np.array(conf_mat_transition[i]), delimiter=",")    

  for i in range(NO_OF_ITER+1):
    np.savetxt('./{}/test_conf_mat_iter_{}.csv'.format(folder_name, i), np.array(test_conf_mat_transition[i]), delimiter=",") 

  all_time_data_dict['total_time'] = time_end - time_start
  all_time_data_dict['total_http_time'] = http_time
  all_time_data_dict['total_http_time_wout_strag'] = sum_time

  # Saving the time_dict for this iteration
  with open("./{}/all_time_data_dict.csv".format(folder_name), 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in all_time_data_dict.items():
      writer.writerow([key, value])    

  #################
  # To read back into dict
  # with open('dict.csv') as csv_file:
  #   reader = csv.reader(csv_file)
  #   mydict = dict(reader)
  #################
  # np.savetxt("./{}/wt_transition.csv".format(folder_name), np.squeeze(np.array(wt_transition)), delimiter=",")

  for i in range(NO_OF_ITER+1):
    np.savetxt("./{}/wt_iter_{}.csv".format(folder_name, i), wt_transition[i], delimiter=",")
  np.savetxt("./{}/cost_transition.csv".format(folder_name), np.array(cost_transition), delimiter=",")  
  np.savetxt("./{}/acc_transition.csv".format(folder_name), np.array(acc_transition), delimiter=",")  
  np.savetxt("./{}/test_acc_transition.csv".format(folder_name), np.array(test_acc_transition), delimiter=",")  

  ##########################################################################

  print('\n\tTotal time taken for entire run: {0:.5f}s'.format(time_end - time_start))
  print('\tTotal time taken for initiator and nodes(excluding stragglers): {0:.5f}s'.format(sum_time))
  print('\tTotal time taken for initiator and nodes: {0:.5f}s'.format(http_time))
  print('\tTime saved by excluding stragglers: {0:.5f}s ({1:.3f} percent of {2:.5f})'.format(http_time - sum_time, 100*(http_time - sum_time)/http_time, http_time))
  if algo == 'time_strag':
    print('\tNumber of nodes used for weight updates on average: ({}/{})'.format(sum(nodes_num_used)/len(nodes_num_used), NO_OF_NODES))
  print('\n Transition of weights: ')

  logit('Total time taken for entire run: %ss'%(time_end - time_start))
  logit('Total time taken for initiator and nodes(excluding stragglers): %ss'%(sum_time))
  logit('Total time taken for initiator and nodes: %ss'%(http_time))
  logit('Time saved by excluding stragglers: %ss (%s percent of %s)'%(http_time - sum_time, 100*(http_time - sum_time)/http_time, http_time))
  if algo == 'time_strag':
    logit('\tNumber of nodes used for weight updates on average: (%s/%s)'%(sum(nodes_num_used)/len(nodes_num_used), NO_OF_NODES))
  logit('Transition of weights: ')

  for i in range(NO_OF_ITER+1):
    print('   Iteration No: {}'.format(i))
    print(wt_transition[i])
    logit('   Iteration No: %s'%(i))
    logit(wt_transition[i])

  print('\n Plotting the cost transition... ')
  logit('Plotting the cost transition... ')

  plt.figure()
  plt.plot(range(NO_OF_ITER+1), cost_transition)
  plt.title('Transition of cost')
  plt.xlabel(r'Number of iterations $\rightarrow$')
  plt.ylabel(r'Cost(Logarithmic) $\rightarrow$')
  plt.grid()
  plt.savefig('./{}/cost-v-iter.png'.format(folder_name), bbox_inches='tight')
  # plt.show()
  plt.close()

  print('\n Plotting the accuracy transition... ')
  logit('Plotting the accuracy transition... ')

  plt.figure()
  plt.plot(range(NO_OF_ITER+1), acc_transition, 'r', label='train')
  plt.plot(range(NO_OF_ITER+1), test_acc_transition, 'b', label='test')
  plt.title('Transition of accuracy')
  plt.xlabel(r'Number of iterations $\rightarrow$')
  plt.ylabel(r'Accuracy(%) $\rightarrow$')
  plt.grid()
  plt.legend()
  plt.savefig('./{}/acc-v-iter.png'.format(folder_name), bbox_inches='tight')
  plt.show()

  print('\n+++++++++++++++++++++++++++++++++++  The Algorithm has finished executing!!  +++++++++++++++++++++++++++++++++++')
  print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
  logit('+++++++++++++++++++++++++++++++++++  The Algorithm has finished executing!!  +++++++++++++++++++++++++++++++++++')
  logit('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

  logging.shutdown()
  return folder_name


#master(dataset_name, no_of_nodes, no_of_iter, lrate, strag, choice, c, factor, connections, timeout, algo, if_ones, if_custom_init_wt, if_change_assgn, if_verbose):
#master('log_train_norm.csv',  50, 70, 12, 0, 'partition', 40, 15, 50, 20, 'ErasHead', 0, False, False, False)

'''

def model_optimize(w, b, X, Y):
  m = X.shape[0]
  
  #Prediction
  final_result = sigmoid_activation(np.dot(w,X.T)+b)
  Y_T = Y.T
  cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))
  #
  
  #Gradient calculation
  dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T))
  db = (1/m)*(np.sum(final_result-Y.T))
  
  grads = {"dw": dw, "db": db}
  
  return grads, cost


def model_predict(w, b, X, Y, learning_rate, no_iterations):
  costs = []
  for i in range(no_iterations):
    #
    grads, cost = model_optimize(w,b,X,Y)
    #
    dw = grads["dw"]
    db = grads["db"]
    #weight update
    w = w - (learning_rate * (dw.T))
    b = b - (learning_rate * db)
    #
        
    if (i % 100 == 0):
      costs.append(cost)
      #print("Cost after %i iteration is %f" %(i, cost))

  #final parameters
  coeff = {"w": w, "b": b}
  gradient = {"dw": dw, "db": db}
  
  return coeff, gradient, costs


def predict(final_pred, m):
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
    return y_pred
'''

