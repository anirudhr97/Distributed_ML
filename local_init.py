#Importing the relevant libraries...
from google.cloud import storage
import numpy as np
import pandas as pd
import sys

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

  @staticmethod
  def assgn_matrix(data, k, choice, c, red):
    '''
    Function to make the assign matrix for the algorithm.
    ###
    Arguments:
    data          -- Numpy array of the dataset
    k             -- Number of nodes
    choice        -- To choose the algorithm
                        -- 'partition'
                        -- 'FRC'
                        -- 'StocGD'
    c             -- parameter c for FRC
    red           -- Redundancy corresponding to Stochastic GD
    ###
    Return:
    mat           -- Assign matrix
    '''
    n = data.shape[0]
    mat = np.zeros((k, n))
    print('k = ', k, ', n = ', n)

    if choice == 'partition':
      index = -(-n // k)
      for i in range(k-1):
        mat[i, index*i:index*(i+1)] = np.ones((index))
      mat[k-1, index*(k-1):] = np.ones((n-index*(k-1)))

    elif choice == 'FRC':
      if n%c == 0 and k*c/n>=1 and (k*c)%n==0:
        l = k*c//n
        for i in range(n//c):
          mat[l*i:l*(i+1):, c*i:c*(i+1)] = np.ones((l, c))
      else:
        print('c value given not appropriate for FRC. Exiting...')
        sys.exit()
    
    elif choice == 'StocGD':
      if k>=red:
        rng = np.random.default_rng()
        arr = np.array([1] * red + [0] * (k-red))
        for i in range(n):
          rng.shuffle(arr)
          mat[:, i] = arr
      else:
        print('redundancy(d) greater than the number of nodes! Exiting...')
        sys.exit()

    else:
      print('Assignment matrix choice not valid. Exiting...')
      sys.exit()

    return mat


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


def initiator(NO_OF_NODES, DATASET_NAME, CHOICE, NO_OF_ITER, IF_ONES, c, FACTOR, VARIETY, red):

  ##############################################################
  storage_client = storage.Client()
  bucket1 = storage_client.bucket('data_tum_base')
  bucket2 = storage_client.bucket('data_tum_master-to-node')
  ##############################################################

  blobber(DATASET_NAME, './tmp/data.zip', bucket1, 'download')
  data = Misc.load_dataset('./tmp/data.zip', IF_ONES)['data']
  d = data.shape[1]-1

  if VARIETY == 'linear' or VARIETY == 'logistic':
    wt = FACTOR*np.random.rand(d)
    
  elif VARIETY == 'multi-logistic':
    labels = np.unique(data[:, -1])
    no_of_unique_labels = labels.shape[0]
    wt = np.zeros((d, no_of_unique_labels))
    for i in range(no_of_unique_labels):
      wt[:, i] = FACTOR*np.random.rand(d)

  np.savetxt("./tmp/weights.csv", wt, delimiter=",")

  mat = Misc.assgn_matrix(data, NO_OF_NODES, CHOICE, c, red)
  np.savetxt("./tmp/assgn_matrix.csv", mat, delimiter=",")

  for i in range(NO_OF_NODES):
    inds = np.where(mat[i,:] == 1)
    temp1 = data[inds]
    np.savetxt("./tmp/temp1.csv", temp1, delimiter=",")
    blobber('dataset_{}.csv'.format(i+1), './tmp/temp1.csv', bucket1, 'upload')

  blobber('weights_iter{}.csv'.format(NO_OF_ITER), './tmp/weights.csv', bucket1, 'upload')
  blobber('assgn_matrix.csv', './tmp/assgn_matrix.csv', bucket2, 'upload')

  return "The initiator has executed its job!"

# dataset_name=&no_of_iter=0&no_of_nodes=10&choice=StocGD&if_ones=0&c=6&factor=1.0&variety=multi-logistic&d=5

ret = initiator(3, 'function0_2d.zip', 'StocGD', 0, 0, 3, 1, 'linear', 2)
print(ret)
