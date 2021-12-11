import numpy as np
import time
import tensorflow as tf
from config import config
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#tfe = tf.contrib.eager
#tf.compat.v1.enable_eager_execution()

def analysis(model):
    # load PU data (x_tr_l = train labeled data, x_tr_u = train unlabeled data, x_te = test data, y_te = test label)
    model_config = model.config
    data_load = np.load('./pu_data/'+model_config['data']+'.npz')
    x_tr_l = data_load['x_tr_l']
    x_tr_u = data_load['x_tr_u']
    y_tr_l = data_load['y_tr_l']
    y_tr_u = data_load['y_tr_u']

    x_val = data_load['x_val']
    y_val = data_load['y_val']

    x_te = data_load['x_te']
    y_te = data_load['y_te']
    data_load.close()

    if 'MNIST' in model_config['data']:
        x_tr_l = (x_tr_l + 1.) / 2.
        x_tr_u = (x_tr_u + 1.) / 2.
        x_val = (x_val + 1.) / 2.
        x_te = (x_te + 1.) / 2.

    if 'conv' in model_config['data']:
        x_tr_l = (x_tr_l + 1.) / 2.
        x_tr_u = (x_tr_u + 1.) / 2.
        x_val = (x_val + 1.) / 2.
        x_te = (x_te + 1.) / 2.

    if 'conv' in model_config['data']:
        # CIFAR
        x_tr_l = np.transpose(x_tr_l.reshape(-1, 3, 32, 32), (0, 2, 3, 1))
        x_tr_u = np.transpose(x_tr_u.reshape(-1, 3, 32, 32), (0, 2, 3, 1))
        x_val = np.transpose(x_val.reshape(-1, 3, 32, 32), (0, 2, 3, 1))
        x_te = np.transpose(x_te.reshape(-1, 3, 32, 32), (0, 2, 3, 1))

    if 'news' in model_config['data']:
        x_tr_l = tf.cast(x_tr_l, tf.float32)
        x_tr_u = tf.cast(x_tr_u, tf.float32)
        x_val = tf.cast(x_val, tf.float32)
        x_te = tf.cast(x_te, tf.float32)

    # clustering of h_y
    o1 = tf.concat([tf.ones([x_tr_l.shape[0], 1]), tf.zeros([x_tr_l.shape[0], 1])], axis=1)
    o2 = tf.concat([tf.zeros([x_tr_u.shape[0], 1]), tf.ones([x_tr_u.shape[0], 1])], axis=1)
    o = tf.concat([o1, o2], axis=0)
    test_tsne(model, model_config, np.concatenate([x_tr_l, x_tr_u], axis=0), np.concatenate([0.5*np.ones_like(y_tr_l[:]), y_tr_u[:]], axis=0), o, 'train_tsne_add_obs_h_y', mode='h_y')
    test_tsne(model, model_config, np.concatenate([x_tr_l, x_tr_u], axis=0), np.concatenate([0.5*np.ones_like(y_tr_l[:]), y_tr_u[:]], axis=0), o, 'train_tsne_add_obs_h_o', mode='h_o')

def test_tsne(model, model_config, x, y, o, fname, makecolor=False, mode='h_y'):

    if makecolor == False:
        color = y
    else:
        color = []
        for i in range(len(y)):
            if y[i] == 3:
                color.append('g')
            if y[i] == 2:
                color.append('r')
            if y[i] == 1:
                color.append('y')
            if y[i] == -1:
                color.append('m')
            if y[i] == -2:
                color.append('b')


    h_y_mu, h_y_log_sig_sq, h_o_mu, h_o_log_sig_sq = model.model_en.encode(x, o)

    tsne = TSNE(n_components=2)

    if mode == 'h_y':
        trans = tsne.fit_transform(h_y_mu)
    elif mode == 'h_o':
        if model_config['n_h_o'] == 2:
            trans = h_o_mu
        else:
            trans = tsne.fit_transform(h_o_mu)
    else:
        NotImplementedError()

    plt.figure()
    plt.scatter(trans[:, 0], trans[:, 1], c=color, s=0.1)

    plt.savefig(model_config['directory'] + fname + '.png')
    plt.close()
