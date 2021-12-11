#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import os
import tensorflow as tf
from model import VAEencoder, VAEdecoder, discriminator, classifier_o, classifier_pn, myPU
from train_load import analysis
from config import config
import matplotlib.pyplot as plt

from keras import backend as K

#tfe = tf.contrib.eager
#tf.compat.v1.enable_eager_execution()

def plotLoss(lst, fname):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(lst)+1), lst)

    plt.savefig(fname)
    plt.close()

def train(num_exp, model_config, pretrain=True):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    K.tensorflow_backend.set_session(sess)

    model_config['directory'] = './result/'+model_config['data']+'/Exp' + str(num_exp) + '/'

    # load PU data (x_tr_l = train labeled data, x_tr_u = train unlabeled data, x_te = test data, y_te = test label)
    data_load = np.load('./pu_data/'+model_config['data']+'.npz')
    x_tr_l = data_load['x_tr_l']
    x_tr_u = data_load['x_tr_u']
    y_tr_l = data_load['y_tr_l']
    y_tr_u = data_load['y_tr_u']

    x_val = data_load['x_val']
    y_val = data_load['y_val']
    x_val = x_tr_l[:5]
    y_val = y_tr_l[:5]

    x_te = data_load['x_te']
    y_te = data_load['y_te']

    data_load.close()

    if 'MNIST' in model_config['data']:
        x_tr_l = (x_tr_l + 1.) / 2.
        x_tr_u = (x_tr_u + 1.) / 2.
        x_val = (x_val + 1.) / 2.
        x_te = (x_te + 1.) / 2.

    if 'news' in model_config['data']:
        x_tr_l = tf.cast(x_tr_l, tf.float32)
        x_tr_u = tf.cast(x_tr_u, tf.float32)
        x_val = tf.cast(x_val, tf.float32)
        x_te = tf.cast(x_te, tf.float32)

    x_demo = np.copy(x_tr_l[:int(model_config['batch_size_l'])])
    x_demo_u = np.copy(x_tr_u[:int(model_config['batch_size_u'])])

    # prepare dataset
    dataset_pl = tf.data.Dataset.from_tensor_slices(x_tr_l).shuffle(model_config['num_buff']).repeat().batch(int(model_config['batch_size_l']), True)
    dataset_u = tf.data.Dataset.from_tensor_slices(x_tr_u).shuffle(model_config['num_buff']).batch(int(model_config['batch_size_u']), True)
    dataset_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(model_config['num_buff']).batch(int(model_config['batch_size_val']), True)
    dataset_test = tf.data.Dataset.from_tensor_slices((x_te, y_te)).shuffle(model_config['num_buff']).batch(int(model_config['batch_size_test']), True)

    model_en = VAEencoder(model_config)
    model_de = VAEdecoder(model_config)
    model_disc = discriminator(model_config)
    model_cl = classifier_o(model_config)
    model_pn = classifier_pn(model_config)

    opt_en = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=model_config['lr_pu'])
    opt_de = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=model_config['lr_pu'])
    opt_disc = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=model_config['lr_disc'], beta1=model_config['beta1'], beta2=model_config['beta2'])
    opt_cl = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=model_config['lr_pu'])
    opt_pn = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=model_config['lr_pn'])
    ##### v2.0 [201122] #####
    # if you want to train the parameters (pi, mu, sigma)
    #opt_param = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=model_config['lr_pu'])
    ##### v2.0 [201122] #####

    ##### v2.0 [201122] #####
    model = myPU(model_config, model_en, model_de, model_disc, model_cl, model_pn, opt_en, opt_de, opt_disc, opt_cl, opt_pn)
    # if you want to train the parameters (pi, mu, sigma)
    #model = myPU(model_config, model_en, model_de, model_disc, model_cl, model_pn, opt_en, opt_de, opt_disc, opt_cl, opt_pn, opt_param)
    ##### v2.0 [201122] #####

    # set logger
    os.makedirs(model_config['directory'], exist_ok=True)
    log = open(model_config['directory'] + 'log.txt', 'w')
    log.write("config\n")
    log.write(str(model_config))
    log.write('\n')
    log.close()

    log2 = open(model_config['directory'] + 'log_PN' + '.txt', 'w')
    log2.write("config\n")
    log2.write(str(model_config))
    log2.write('\n')
    log2.close()

    lstLoss1 = []
    lstLoss2 = []
    lstLoss3 = []
    lstLoss4 = []
    lstLoss5 = []
    lstAcc = []
    lstVal = []

    preLoss1 = []
    preLoss2 = []
    if pretrain:
        pretrain_time = time.time()
        for epoch in range(1, model_config['num_epoch_pre'] + 1):
            print('[PRE-TRAIN] Exp: {} / Epoch: {}'.format(num_exp, epoch))

            start_time = time.time()
            lst_1 = []
            lst_2 = []

            for x_pl, x_u in zip(dataset_pl, dataset_u):
                l1 = model.pretrain(x_pl, x_u)
                lst_1.append(l1)
                if model_config['bool_pn_pre']:
                    l2 = model.train_step_pn_pre(x_pl, x_u)
                    lst_2.append(l2)

            if np.isnan(l1):
                print('nan occurs!')
                break

            #if epoch % 10 == 0:
            #    if 'MNIST' in model_config['data']:
            #        model.compare(model_config['directory'] + str(epoch) + '_pre.png', x_demo, x_demo_u, 5)

            #    if 'conv' in model_config['data']:
            #        model.compare(model_config['directory'] + str(epoch) + '_pre.png', x_demo, x_demo_u, 5)

            preLoss1.append(sum(lst_1) / len(lst_1))
            if model_config['bool_pn_pre']:
                preLoss2.append(sum(lst_2) / len(lst_2))

            end_time = time.time()
            print('[PRE-TRAIN] Remaining time: {} sec'.format((model_config['num_epoch_pre'] - epoch) * (end_time - start_time)))


    plotLoss(preLoss1, model_config['directory'] + 'loss_pretrain.png')
    if model_config['bool_pn_pre']:
       plotLoss(preLoss2, model_config['directory'] + 'loss_pretrain_pn.png')
    print('PRE-TRAIN finish!')
    model.findPrior(x_tr_l, x_tr_u)
    np.savez(model_config['directory']+'prior', mu=model.mu, var=model.var)

    # set checkpoint
    checkpoint_dir = model_config['directory'] + 'training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    ##### v2.0 [201122] #####
    checkpoint = tf.train.Checkpoint(model_en=model.model_en,
                                     model_de=model.model_de,
                                     model_disc=model.model_disc,
                                     model_cl=model.model_cl,
                                     model_pn=model.model_pn)
    # if you want to train the parameters (pi, mu, sigma)
    #checkpoint = tf.train.Checkpoint(model_en=model.model_en,
    #                                 model_de=model.model_de,
    #                                 model_disc=model.model_disc,
    #                                 model_cl=model.model_cl,
    #                                 model_pn=model.model_pn,
    #                                 model_p=model.p, model_mu=model.mu, model_var=model.var)
    ##### v2.0 [201122] #####

    manager = tf.contrib.checkpoint.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=100)
    #model.model_en = VAEencoder(model_config)
    #model.model_de = VAEdecoder(model_config)

    lsttime1 = []
    lsttime2 = []
    for epoch in range(1, model_config['num_epoch'] + 1):
        log = open(model_config['directory'] + 'log.txt', 'a')
        log2 = open(model_config['directory'] + 'log_PN' + '.txt', 'a')
        #print('Exp: {} / Epoch: {}'.format(num_exp, epoch))

        start_time = time.time()
        lst_1 = []
        lst_2 = []
        lst_3 = []
        lst_4 = []
        lst_5 = []

        for x_pl, x_u in zip(dataset_pl, dataset_u):
            if epoch <= model_config['num_epoch_step3']:
                if model_config['num_epoch_step1'] < epoch <= model_config['num_epoch_step2']:
                    pass
                else:
                    l3 = model.train_step_disc(x_pl, x_u)
                    lst_3.append(l3)
                    l1, l2, l4 = model.train_step_vae(x_pl, x_u, epoch)
                    lst_1.append(l1)
                    lst_2.append(l2)
                    lst_4.append(l4)

            if epoch > model_config['num_epoch_step1']:
                if model_config['num_epoch_step_pn1'] < epoch <= model_config['num_epoch_step_pn2']:
                    pass
                else:
                    l5 = model.train_step_pn(x_pl, x_u)
                    lst_5.append(l5)

        if np.isnan(l1):
            print('nan occurs!')
            break

        if epoch <= model_config['num_epoch_step3']:
            if epoch % 100 == 0:
                #if 'MNIST' in model_config['data']:
                #    model.compare(model_config['directory'] + str(epoch) + '.png', x_demo, x_demo_u, 5)

                log.write('epoch: {}\n'.format(epoch))
                d_x_pu, d_x_u = model.check_disc(x_pl, x_u)
                d_x_pu2, d_x_pl2 = model.check_pn(x_pl, x_u)

                log.write('d_x_pu: {}, d_x_u: {}\n'.format(np.average(d_x_pu), np.average(d_x_u)))
                tf.print('d_x_pu: {}, d_x_u: {}'.format(np.average(d_x_pu), np.average(d_x_u)))
                log.write('d_x_pu2: {}, d_x_pl2: {}\n'.format(np.average(d_x_pu2), np.average(d_x_pl2)))
                tf.print('d_x_pu2: {}, d_x_pl2: {}'.format(np.average(d_x_pu2), np.average(d_x_pl2)))
            if model_config['num_epoch_step1'] < epoch <= model_config['num_epoch_step2']:
                pass
            else:
                lstLoss1.append(sum(lst_1) / len(lst_1))
                lstLoss2.append(sum(lst_2) / len(lst_2))
                lstLoss3.append(sum(lst_3) / len(lst_3))
                lstLoss4.append(sum(lst_4) / len(lst_4))


        if epoch > model_config['num_epoch_step1']:
            if model_config['num_epoch_step_pn1'] < epoch <= model_config['num_epoch_step_pn2']:
                pass
            else:
                log2.write('epoch: {}, loss: {}'.format(epoch, sum(lst_5) / len(lst_5)) + '\n')
                print('epoch: {}, loss: {}'.format(epoch, sum(lst_5) / len(lst_5)))
                # acc test
                val_acc, val_pr, val_re = model.accuracy(dataset_val)
                lstAcc.append(val_acc)
                log2.write(
                    '...val: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}'.format(val_acc, val_pr, val_re) + '\n')
                print('...val: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}'.format(val_acc, val_pr, val_re))

                val_loss = model.loss_val(x_val[:20], x_val[20:])
                lstVal.append(val_loss)
                print(val_loss)

                lstLoss5.append(sum(lst_5) / len(lst_5))

        if epoch % model_config['save_epoch'] == 0:
            manager.save()
        log.close()
        log2.close()

        end_time = time.time()
        print('Exp: {} / Epoch: {:4} |||| Remaining time: {:.2f} sec'.format(num_exp, epoch, (model_config['num_epoch'] - epoch) * (end_time - start_time)))
        #print(''.format((model_config['num_epoch'] - epoch) * (end_time - start_time)))

        if epoch <= model_config['num_epoch_step3']:
            if model_config['num_epoch_step1'] < epoch <= model_config['num_epoch_step2']:
                pass
            else:
                lsttime1.append(end_time - start_time)

        if epoch > model_config['num_epoch_step1']:
            if model_config['num_epoch_step_pn1'] < epoch <= model_config['num_epoch_step_pn2']:
                pass
            else:
                lsttime2.append(end_time - start_time)

    print(np.mean(np.array(lsttime1[1:])), np.mean(np.array(lsttime2[1:])))

    plotLoss(lstLoss1, model_config['directory'] + 'loss_vae.png')
    plotLoss(lstLoss2, model_config['directory'] + 'loss_disc.png')
    plotLoss(lstLoss3, model_config['directory'] + 'loss_gen.png')
    plotLoss(lstLoss4, model_config['directory'] + 'loss_cl.png')

    np.savez(model_config['directory'] + 'PU_loss_val_VAEPU', loss=lstVal)

    log2 = open(model_config['directory'] + 'log_PN' + '.txt', 'a')
    acc, precision, recall = model.accuracy(dataset_test)
    log2.write('final test : acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}'.format(acc, precision, recall))
    print('final test : acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}'.format(acc, precision, recall))
    log2.close()

    plotLoss(lstLoss5, model_config['directory'] + 'loss_PN.png')
    plotLoss(lstAcc, model_config['directory'] + 'val_accuracy.png')

    #print(model.p)