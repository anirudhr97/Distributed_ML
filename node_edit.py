#Importing the relevant libraries
from google.cloud import storage
from scipy.special import expit as sigmoid
import numpy as np
import pandas as pd
import os

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

def sigm(arg):
  '''
  Function to pass each element of 'arg' through sigmoid
  ###
  Arguments:
  arg       -- numpy array whose sigmoid we need
  ###
  Return:
  temp      -- sigmoid of 'arg'(numpy array)
  '''
  temp = 1/(1+np.exp(-arg))
  return temp

################################################################
################################################################

def node(request):

  request_json = request.get_json()
  if request.args and 'dataset_name' in request.args and 'iter_no' in request.args and 'task' in request.args and 'n_labels' in request.args:
    DATASET_NAME = request.args.get('dataset_name')
    ITER_NO = int(request.args.get('iter_no'))
    TASK = request.args.get('task')
    N_LABELS = int(request.args.get('n_labels'))

  elif request_json and 'dataset_name' in request_json and 'iter_no' in request_json and 'task' in request_json and 'n_labels' in request_json:
    DATASET_NAME = request_json['dataset_name']
    ITER_NO = int(request_json['iter_no'])
    TASK = request_json['task']
    N_LABELS = int(request_json['n_labels'])
  
  else:
    return f'Appropriate parameters not given. \'dataset_name\', \'task\', \'if_ones\',  and \'iter_no\' need to be given.'

  ##############################################################
  ident = int(os.environ['identity'])
  storage_client = storage.Client()
  bucket1 = storage_client.bucket('data_tum_base')
  bucket2 = storage_client.bucket('data_tum_master-to-node')
  bucket3 = storage_client.bucket('data_tum_node-to-master')
  ##############################################################

  blobber('assgn_matrix.csv', '/tmp/mat.csv', bucket2, 'download')
  blobber('weights_iter{}.csv'.format(ITER_NO), '/tmp/weights.csv', bucket1, 'download')
  blobber(DATASET_NAME, '/tmp/data.csv', bucket1, 'download')

  data = Misc.load_dataset('/tmp/data.csv', 0)['data']
  wt = Misc.load_dataset('/tmp/weights.csv', 0)['data']
  mat = Misc.load_dataset('/tmp/mat.csv', 0)['data']  
  ##############################################################

  # Extracting the relevant data points according to 'assgn_matrix.csv'.
  X = data[:, :-1]
  y = data[:, -1]

  # Calculating gradient based on if we are doing a linear/logistic regression task.
  if TASK == 'linear':
    grad = 2*np.transpose(X)@(np.dot(X, wt) - np.reshape(y, (-1, 1)))/np.shape(mat)[1]
    np.savetxt("/tmp/grad.csv", grad, delimiter=",")
  elif TASK == 'logistic':
    grad = np.transpose(X)@(sigmoid(np.dot(X, wt))-np.reshape(y, (-1, 1)))/np.shape(mat)[1]
    np.savetxt("/tmp/grad.csv", grad, delimiter=",")
  elif TASK == 'multi-logistic':
    grad = np.zeros((X.shape[1], N_LABELS))
    labels = np.unique(y)
    for i in labels:
      y1 = 1 * (y == i)
      h = sigmoid( X.dot((wt[:, int(i)])[:, None]) )
      X_diff = h - y1[:,None]
      X_grad = X * X_diff
      grad[:, int(i)] = np.sum(X_grad,axis=0)/np.shape(mat)[1]
    np.savetxt("/tmp/grad.csv", grad, delimiter=",")

  # Y_train1 = 1 * (Y_train == i)
  # h = g(X_train.dot(theta))
  # X_diff = h - Y_train1[:,None]
  # X_grad = X_train * X_diff
  # theta = theta - ((alpha/m) * np.sum(X_grad,axis=0))[:,None]

  else:
    print('Unknown \'task\' given.')
    return "Unknown \'task\' given."
  ##############################################################

  # Uploading the gradient computed to 'data_tum_node-to-master' bucket.
  gradname = "grad_iterno_{}_n{}_.csv".format(ITER_NO, ident)
  blobber(gradname, '/tmp/grad.csv', bucket3, 'upload')

  return "Node{} has finished its job for iteration {}".format(ident, ITER_NO)

