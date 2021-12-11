import numpy as np
import time
import os
import tensorflow as tf
from config import config
from train import train

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

tfe = tf.contrib.eager
tf.compat.v1.enable_eager_execution()

idx = 0

# dataset file name (except .npz)
config['data'] = 'MNIST_35_val'

config['pi_pl'] = 0.01
config['pi_pu'] = 0.5 - config['pi_pl']
config['pi_u'] = 1 - config['pi_pl']

config['n_h_y'] = 10
config['n_h_o'] = 2
config['lr_pu'] = 3e-4
config['lr_pn'] = 1e-5
config['num_epoch_pre'] = 100
config['num_epoch_step1'] = 400

config['num_epoch_step_pn1'] = 500
config['num_epoch_step_pn2'] = 600

config['num_epoch_step2'] = 500
config['num_epoch_step3'] = 700
config['num_epoch'] = 800

config['n_hidden_cl'] = []
config['n_hidden_pn'] = [300, 300, 300, 300]

config['batch_size_l'], config['batch_size_u'] = (10, 990)
config['batch_size_l_pn'], config['batch_size_u_pn'] = (10, 990)

config['alpha_gen'] = 0.1
config['alpha_disc'] = 0.1
config['alpha_gen2'] = 3
config['alpha_disc2'] = 3

np.random.seed(idx)
tf.set_random_seed(idx)
train(idx, config)
