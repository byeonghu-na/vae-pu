#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

#tfe = tf.contrib.eager
#tf.compat.v1.enable_eager_execution()


class VAEencoder(tf.keras.Model):

    def __init__(self, config):
        super(VAEencoder, self).__init__()
        self.config = config

        self.vae_en_y = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(self.config['n_input'],))])
        for i in range(len(self.config['n_hidden_vae_e'])):
            self.vae_en_y.add(tf.keras.layers.Dense(self.config['n_hidden_vae_e'][i]))
            self.vae_en_y.add(tf.keras.layers.BatchNormalization())
            self.vae_en_y.add(tf.keras.layers.LeakyReLU(0.2))
        self.vae_en_y.build()
        # self.vae_en_y.summary()

        self.vae_en_o = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(self.config['n_input'] + self.config['n_o'],))])
        for i in range(len(self.config['n_hidden_vae_e'])):
            self.vae_en_o.add(tf.keras.layers.Dense(self.config['n_hidden_vae_e'][i]))
            self.vae_en_o.add(tf.keras.layers.BatchNormalization())
            self.vae_en_o.add(tf.keras.layers.LeakyReLU(0.2))
        self.vae_en_o.build()
        # self.vae_en_o.summary()

        self.vae_en_y_mu = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(self.config['n_hidden_vae_e'][-1],))])
        self.vae_en_y_mu.add(tf.keras.layers.Dense(self.config['n_h_y']))
        self.vae_en_y_mu.build()
        # self.vae_en_y_mu.summary()

        self.vae_en_y_lss = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(self.config['n_hidden_vae_e'][-1],))])
        self.vae_en_y_lss.add(tf.keras.layers.Dense(self.config['n_h_y']))
        self.vae_en_y_lss.build()
        # self.vae_en_y_lss.summary()

        self.vae_en_o_mu = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(self.config['n_hidden_vae_e'][-1],))])
        self.vae_en_o_mu.add(tf.keras.layers.Dense(self.config['n_h_o']))
        self.vae_en_o_mu.build()
        # self.vae_en_o_mu.summary()

        self.vae_en_o_lss = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(self.config['n_hidden_vae_e'][-1],))])
        self.vae_en_o_lss.add(tf.keras.layers.Dense(self.config['n_h_o']))
        self.vae_en_o_lss.build()
        # self.vae_en_o_lss.summary()

    def encode(self, x, o):
        # separate each component
        hidden_y = self.vae_en_y(x)
        y_mu = self.vae_en_y_mu(hidden_y)
        y_lss = self.vae_en_y_lss(hidden_y)

        hidden_o = self.vae_en_o(tf.concat([x, o], axis=1))
        o_mu = self.vae_en_o_mu(hidden_o)
        o_lss = self.vae_en_o_lss(hidden_o)

        return y_mu, y_lss, o_mu, o_lss


class VAEdecoder(tf.keras.Model):

    def __init__(self, config):
        super(VAEdecoder, self).__init__()
        self.config = config

        # separate each component
        self.vae_de = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(self.config['n_h_y'] + self.config['n_h_o'],))])
        for i in range(len(self.config['n_hidden_vae_d'])):
            self.vae_de.add(tf.keras.layers.Dense(self.config['n_hidden_vae_d'][i]))
            self.vae_de.add(tf.keras.layers.BatchNormalization())
            self.vae_de.add(tf.keras.layers.LeakyReLU(0.2))
        # if the output is binary (0-1)
        if 'MNIST' in self.config['data']:
            self.vae_de.add(tf.keras.layers.Dense(self.config['n_input']))
        elif '01' in self.config['data']:
            self.vae_de.add(tf.keras.layers.Dense(self.config['n_input']))
        '''
        # if the output is continuous
        elif 'IIC' in self.config['data']:
            self.vae_de_mu = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(self.config['n_hidden_vae_d'][-1],))])
            self.vae_de_mu.add(tf.keras.layers.Dense(self.config['n_input'], activation='relu'))
            self.vae_de_mu.build()
        '''
        self.vae_de.build()
        #self.vae_de.summary()

    def decode(self, h_y, h_o, sigmoid=False):

        '''
        # if the output is continuous
        if 'IIC' in self.config['data']:
            recon = self.vae_de(tf.concat([h_y, h_o], axis=1))
            recon_mu = self.vae_de_mu(recon)
            #recon_lss = self.vae_de_lss(recon)
            #recon_lss = tf.clip_by_value(recon_lss, -10, 10)
            return recon_mu
        '''

        recon = self.vae_de(tf.concat([h_y, h_o], axis=1))
        if sigmoid:
            recon = tf.nn.sigmoid(recon)
        return recon


class discriminator(tf.keras.Model):

    def __init__(self, config):
        super(discriminator, self).__init__()
        self.config = config

        self.disc_u = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(self.config['n_input'],))])
        for i in range(len(self.config['n_hidden_disc'])):
            self.disc_u.add(tf.keras.layers.Dense(self.config['n_hidden_disc'][i], activation=tf.nn.leaky_relu))
            self.disc_u.add(tf.keras.layers.Dropout(0.3))
        self.disc_u.add(tf.keras.layers.Dense(1))

        self.disc_u.build()

    def discriminate(self, x, sigmoid=False):
        disc = self.disc_u(x)
        if sigmoid:
            disc = tf.nn.sigmoid(disc)
        return disc


class classifier_o(tf.keras.Model):

    def __init__(self, config):
        super(classifier_o, self).__init__()
        self.config = config

        self.classification = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.config['n_h_o'],)),
            ])
        for i in range(len(self.config['n_hidden_cl'])):
            self.classification.add(tf.keras.layers.Dense(self.config['n_hidden_cl'][i], kernel_regularizer=keras.regularizers.l2(1e-5)))
            self.classification.add(tf.keras.layers.BatchNormalization())
            self.classification.add(tf.keras.layers.LeakyReLU(0.2))
        self.classification.add(tf.keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(1e-5)))
        self.classification.build()
        #self.classification.summary()

    def classify(self, x, sigmoid=False):
        c = self.classification(x)
        if sigmoid:
            c = tf.nn.sigmoid(c)
        return c


class classifier_pn(tf.keras.Model):

    def __init__(self, config):
        super(classifier_pn, self).__init__()
        self.config = config

        self.classification = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.config['n_input'],)),
            ])
        for i in range(len(self.config['n_hidden_pn'])):
            self.classification.add(tf.keras.layers.Dense(self.config['n_hidden_pn'][i], kernel_regularizer=keras.regularizers.l2(1e-5)))
            self.classification.add(tf.keras.layers.BatchNormalization())
            self.classification.add(tf.keras.layers.LeakyReLU(0.2))
        self.classification.add(tf.keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(1e-5)))
        self.classification.build()
        #self.classification.summary()

    def classify(self, x, sigmoid=False):
        c = self.classification(x)
        if sigmoid:
            c = tf.nn.sigmoid(c)
        return c


class myPU():

    def __init__(self, config, model_en, model_de, model_disc, model_cl, model_pn, opt_en, opt_de, opt_disc, opt_cl, opt_pn, opt_param=None):
        self.config = config

        self.model_en = model_en
        self.model_de = model_de
        self.model_disc = model_disc
        self.model_cl = model_cl
        self.model_pn = model_pn

        self.opt_en = opt_en
        self.opt_de = opt_de
        self.opt_disc = opt_disc
        self.opt_cl = opt_cl
        self.opt_pn = opt_pn

        ##### v2.0 [201122] #####
        # if you want to train the parameters (pi, mu, sigma)
        #self.opt_param = opt_param
        ##### v2.0 [201122] #####

    def reparameterization(self, mu, log_sig_sq):
        eps = tf.random.normal(shape=mu.shape)
        return mu + tf.exp(log_sig_sq / 2.) * eps

    def concat_data(self, x_pl, x_u):
        x = tf.concat([x_pl, x_u], axis=0)

        o_pl = tf.concat([tf.ones([x_pl.shape[0], 1]), tf.zeros([x_pl.shape[0], 1])], axis=1)
        o_u = tf.concat([tf.zeros([x_u.shape[0], 1]), tf.ones([x_u.shape[0], 1])], axis=1)
        o = tf.concat([o_pl, o_u], axis=0)

        return x, o

    def pretrain(self, x_pl, x_u):

        x, o = self.concat_data(x_pl, x_u)

        with tf.GradientTape() as vae_en_tape, tf.GradientTape() as vae_de_tape:
            h_y_mu, h_y_log_sig_sq, h_o_mu, h_o_log_sig_sq = self.model_en.encode(x, o)
            h_y = self.reparameterization(h_y_mu, h_y_log_sig_sq)
            h_o = self.reparameterization(h_o_mu, h_o_log_sig_sq)
            if 'MNIST' in self.config['data']:
                recon_x = self.model_de.decode(h_y, h_o)
                loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=recon_x), axis=1))
            elif '01' in self.config['data']:
                recon_x = self.model_de.decode(h_y, h_o)
                loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=recon_x), axis=1))
            elif 'Swiss' in self.config['data']:
                recon_x = self.model_de.decode(h_y, h_o)
                loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=recon_x), axis=1))
            # if the output is continuous
            elif '20' in self.config['data']:
                recon_x_mu = self.model_de.decode(h_y, h_o)
                loss = tf.reduce_mean(tf.reduce_sum( 0.5 * tf.square(x - recon_x_mu), axis=1))
            # if the output is continuous
            elif 'CIFAR' in self.config['data']:
                recon_x_mu = self.model_de.decode(h_y, h_o)
                loss = tf.reduce_mean(tf.reduce_sum( 0.5 * tf.square(x - recon_x_mu), axis=1))
            else:
                NotImplementedError()

        gradients_of_vae_en = vae_en_tape.gradient(loss, self.model_en.trainable_variables)
        gradients_of_vae_de = vae_de_tape.gradient(loss, self.model_de.trainable_variables)
        self.opt_en.apply_gradients(zip(gradients_of_vae_en, self.model_en.trainable_variables))
        self.opt_de.apply_gradients(zip(gradients_of_vae_de, self.model_de.trainable_variables))

        return loss.numpy()

    def findPrior(self, x_tr_l, x_tr_u):
        from sklearn.mixture import GaussianMixture

        o_pl = tf.concat([tf.ones([x_tr_l.shape[0], 1]), tf.zeros([x_tr_l.shape[0], 1])], axis=1)
        o_u = tf.concat([tf.zeros([x_tr_u.shape[0], 1]), tf.ones([x_tr_u.shape[0], 1])], axis=1)

        h_y_u_mu, h_y_u_log_sig_sq, _, _ = self.model_en.encode(x_tr_u, o_u)
        h_y_u = h_y_u_mu
        h_y_l_mu, h_y_l_log_sig_sq, _, _ = self.model_en.encode(x_tr_l, o_pl)
        h_y_l = h_y_l_mu

        h_y = tf.concat([h_y_u, h_y_l], axis=0)

        gmm = GaussianMixture(n_components=2, covariance_type='diag')
        gmm.fit(h_y)

        self.p = gmm.weights_[1]
        self.mu = np.asarray(gmm.means_, dtype=np.float32)
        self.var = np.asarray(gmm.covariances_, dtype=np.float32)

        c0 = tf.reduce_mean(-0.5 * tf.truediv(tf.square(h_y_l - self.mu[0]), self.var[0]) - 0.5 * tf.math.log(self.var[0] + 1e-9), axis=1)
        c1 = tf.reduce_mean(-0.5 * tf.truediv(tf.square(h_y_l - self.mu[1]), self.var[1]) - 0.5 * tf.math.log(self.var[1] + 1e-9), axis=1)

        num0 = tf.reduce_sum(tf.cast(tf.greater(c0 - c1, 0.), dtype=tf.int32))
        frac0 = num0 / x_tr_l.shape[0]
        if frac0 > 0.5:
            self.p = gmm.weights_[0]
            self.mu[0], self.mu[1] = gmm.means_[1], gmm.means_[0]
            self.var[0], self.var[1] = gmm.covariances_[1], gmm.covariances_[0]
        self.mu = np.asarray(self.mu, dtype=np.float32)
        self.var = np.asarray(self.var, dtype=np.float32)

        ##### v2.0 [201122] #####
        # if you want to train the parameters (pi, mu, sigma)
        #self.p = tf.Variable(self.p.astype(np.float32))
        #self.mu = tf.Variable(self.mu)
        #self.var = tf.Variable(self.var)
        ##### v2.0 [201122] #####

        #print(self.p, self.mu, self.var)
        #print(tf.reduce_sum(c0) - tf.reduce_sum(c1))

    def train_step_vae(self, x_pl, x_u, epoch):

        alpha_gen = self.config['alpha_gen']
        alpha_gen2 = self.config['alpha_gen2']
        alpha_cl = self.config['alpha_cl']
        if self.config['pi_given'] == None:
            p = self.config['pi_pl'] + self.config['pi_pu']
        else:
            p = self.config['pi_given']

        # if you want to train the parameters (pi, mu, sigma)
        #p = self.p

        x, o = self.concat_data(x_pl, x_u)

        with tf.GradientTape() as vae_en_tape, tf.GradientTape() as vae_de_tape, tf.GradientTape() as cl_tape, tf.GradientTape() as param_tape:
            h_y_mu, h_y_log_sig_sq, h_o_mu, h_o_log_sig_sq = self.model_en.encode(x, o)
            h_y = self.reparameterization(h_y_mu, h_y_log_sig_sq)
            h_o = self.reparameterization(h_o_mu, h_o_log_sig_sq)

            c0 = -0.5 * tf.truediv(tf.square(h_y - self.mu[0]), self.var[0]) - 0.5 * tf.math.log(tf.maximum(self.var[0], 1e-9)) + tf.math.log(tf.maximum(1 - p, 1e-9))
            c1 = -0.5 * tf.truediv(tf.square(h_y - self.mu[1]), self.var[1]) - 0.5 * tf.math.log(tf.maximum(self.var[1], 1e-9)) + tf.math.log(tf.maximum(p, 1e-9))

            c0 = tf.reduce_sum(c0, axis=1, keep_dims=True)
            c1 = tf.reduce_sum(c1, axis=1, keep_dims=True)

            c = tf.concat([c0, c1], axis=1)
            c = tf.expand_dims(tf.nn.softmax(c, axis=1)[:, 1], 1)

            _, _, _, x_pu, _ = self.generate(x_pl, x_u, self.config['mode'])
            d_x_pu = self.model_disc.discriminate(x_pu, sigmoid=False)
            label = tf.ones_like(d_x_pu)

            gan_loss = alpha_gen * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=d_x_pu))

            if epoch <= self.config['num_epoch_step1']:
                gan_loss2 = 0 * tf.reduce_mean(tf.zeros(1))
            else:
                d_x_pu2 = self.model_pn.classify(x_pu, sigmoid=False)
                gan_loss2 = alpha_gen2 * tf.reduce_mean(self.sigmoid_loss(d_x_pu2, tf.ones_like(d_x_pu2)))

            loss1_0 = - tf.reduce_sum(0.5 * (tf.math.log(tf.maximum(self.var[0], 1e-9)) + tf.truediv(tf.exp(h_y_log_sig_sq) + tf.square(h_y_mu - self.mu[0]), self.var[0])), axis=1, keepdims=True)
            loss1_1 = - tf.reduce_sum(0.5 * (tf.math.log(tf.maximum(self.var[1], 1e-9)) + tf.truediv(tf.exp(h_y_log_sig_sq) + tf.square(h_y_mu - self.mu[1]), self.var[1])), axis=1, keepdims=True)

            loss1 = tf.reduce_mean(tf.multiply(1 - c, loss1_0) + tf.multiply(c, loss1_1))
            loss2 = - tf.reduce_mean(tf.reduce_sum(0.5 * (tf.exp(h_o_log_sig_sq) + tf.square(h_o_mu)), axis=1))

            if 'MNIST' in self.config['data']:
                recon_x = self.model_de.decode(h_y, h_o)
                loss3 = - tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=recon_x), axis=1))
            elif '01' in self.config['data']:
                recon_x = self.model_de.decode(h_y, h_o)
                loss3 = - tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=recon_x), axis=1))
            elif 'conv' in self.config['data']:
                recon_x = self.model_de.decode(h_y, h_o)
                loss3 = - tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=recon_x), axis=[1,2,3]))
            elif '20' in self.config['data']:
                recon_x = self.model_de.decode(h_y, h_o)
                loss3 = self.config['alpha_test'] * - tf.reduce_mean(tf.reduce_sum(0.5 * tf.square(x - recon_x), axis=1))
            elif 'CIFAR' in self.config['data']:
                recon_x = self.model_de.decode(h_y, h_o)
                loss3 = self.config['alpha_test'] * - tf.reduce_mean(tf.reduce_sum(0.5 * tf.square(x - recon_x), axis=1))
            else:
                NotImplementedError()

            loss4 = tf.reduce_mean(tf.reduce_sum(0.5 * (1 + h_y_log_sig_sq), axis=1))
            loss5 = tf.reduce_mean(tf.reduce_sum(0.5 * (1 + h_o_log_sig_sq), axis=1))

            loss6 = tf.reduce_mean(- c * tf.math.log((c / tf.maximum(p, 1e-9)) + 1e-9) - (1 - c) * tf.math.log(((1 - c) / tf.maximum(1-p, 1e-9)) + 1e-9))

            c_o = self.model_cl.classify(h_o, sigmoid=False)
            label = tf.expand_dims(o[:, 0], 1)
            loss7 = - tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=c_o))

            vade_loss = - self.config['alpha_vade'] * (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7)

            loss_t1 = vade_loss + gan_loss + gan_loss2

        gradients_of_vae_en = vae_en_tape.gradient(loss_t1, self.model_en.trainable_variables)
        gradients_of_vae_de = vae_de_tape.gradient(loss_t1, self.model_de.trainable_variables)
        gradients_of_cl = cl_tape.gradient(loss_t1, self.model_cl.trainable_variables)
        self.opt_en.apply_gradients(zip(gradients_of_vae_en, self.model_en.trainable_variables))
        self.opt_de.apply_gradients(zip(gradients_of_vae_de, self.model_de.trainable_variables))
        self.opt_cl.apply_gradients(zip(gradients_of_cl, self.model_cl.trainable_variables))

        ##### v2.0 [201122] #####
        # if you want to train the parameters (pi, mu, sigma)
        #gradients_of_param = param_tape.gradient(loss_t1, [self.p, self.mu, self.var])
        #self.opt_param.apply_gradients(zip(gradients_of_param, [self.p, self.mu, self.var]))
        ##### v2.0 [201122] #####

        return vade_loss.numpy(), gan_loss.numpy(), gan_loss2.numpy()

    def generate(self, x_pl, x_u, mode='near_o'):
        if mode == 'near_o':
            # nearest h_o
            o_pl = np.concatenate([np.ones([x_pl.shape[0], 1]), np.zeros([x_pl.shape[0], 1])], axis=1)
            o_u = np.concatenate([np.zeros([x_u.shape[0], 1]), np.ones([x_u.shape[0], 1])], axis=1)

            h_y_mu, h_y_log_sig_sq, h_o_mu, h_o_log_sig_sq = self.model_en.encode(x_pl, o_pl)
            h_y = self.reparameterization(h_y_mu, h_y_log_sig_sq)
            h_o = self.reparameterization(h_o_mu, h_o_log_sig_sq)

            _, _, h_o_mu_x, h_o_log_sig_sq_x = self.model_en.encode(x_u, o_u)
            h_o_x = self.reparameterization(h_o_mu_x, h_o_log_sig_sq_x)

            h_o_2 = tf.reduce_sum(tf.square(h_o), 1)
            h_o_x_2 = tf.reduce_sum(tf.square(h_o_x), 1)

            h_o_2 = tf.reshape(h_o_2, [-1, 1])
            h_o_x_2 = tf.reshape(h_o_x_2, [1, -1])

            distance = tf.sqrt(h_o_2 - 2 * tf.matmul(h_o, h_o_x, False, True) + h_o_x_2)
            lstIdx = tf.math.argmin(distance, 1)
            ne_h_o = tf.gather(h_o_x, lstIdx)

            x_u_select = tf.gather(x_u, lstIdx)

            if '20' not in self.config['data']:
                x_pu = self.model_de.decode(h_y, ne_h_o, sigmoid=True)
            elif '01' in self.config['data']:
                x_pu = self.model_de.decode(h_y, ne_h_o, sigmoid=True)
            elif 'MNIST' in self.config['data']:
                x_pu = self.model_de.decode(h_y, ne_h_o, sigmoid=True)
            else:
                x_pu_mu  = self.model_de.decode(h_y, ne_h_o)
                # x_pu = self.reparameterization(x_pu_mu, x_pu_lss)
                x_pu = x_pu_mu
            return h_y, h_o, ne_h_o, x_pu, x_u_select

        elif mode == 'near_y':
            o_pl = np.concatenate([np.ones([x_pl.shape[0], 1]), np.zeros([x_pl.shape[0], 1])], axis=1)
            o_u = np.concatenate([np.zeros([x_u.shape[0], 1]), np.ones([x_u.shape[0], 1])], axis=1)

            h_y_mu, h_y_log_sig_sq, h_o_mu, h_o_log_sig_sq = self.model_en.encode(x_pl, o_pl)
            h_y = self.reparameterization(h_y_mu, h_y_log_sig_sq)
            h_o = self.reparameterization(h_o_mu, h_o_log_sig_sq)

            h_y_mu_x, h_y_log_sig_sq_x, h_o_mu_x, h_o_log_sig_sq_x = self.model_en.encode(x_u, o_u)
            h_y_x = self.reparameterization(h_y_mu_x, h_y_log_sig_sq_x)
            h_o_x = self.reparameterization(h_o_mu_x, h_o_log_sig_sq_x)

            h_y_2 = tf.reduce_sum(tf.square(h_y), 1)
            h_y_x_2 = tf.reduce_sum(tf.square(h_y_x), 1)

            h_y_2 = tf.reshape(h_y_2, [-1, 1])
            h_y_x_2 = tf.reshape(h_y_x_2, [1, -1])

            distance = tf.sqrt(h_y_2 - 2 * tf.matmul(h_y, h_y_x, False, True) + h_y_x_2)
            lstIdx = tf.math.argmin(distance, 1)
            ne_h_o = tf.gather(h_o_x, lstIdx)

            x_u_select = tf.gather(x_u, lstIdx)

            if '20' not in self.config['data']:
                x_pu = self.model_de.decode(h_y, ne_h_o, sigmoid=True)
            elif '01' in self.config['data']:
                x_pu = self.model_de.decode(h_y, ne_h_o, sigmoid=True)
            elif 'MNIST' in self.config['data']:
                x_pu = self.model_de.decode(h_y, ne_h_o, sigmoid=True)
            else:
                x_pu_mu = self.model_de.decode(h_y, ne_h_o)
                x_pu = x_pu_mu
            return h_y, h_o, ne_h_o, x_pu, x_u_select

        elif mode == 'random':
            x_pu = x_pl
            return 0, 0, 0, x_pu, 0

        else:
            NotImplementedError()

    def train_step_disc(self, x_pl, x_u):

        alpha_disc = self.config['alpha_disc']
        alpha_disc2 = self.config['alpha_disc2']

        with tf.GradientTape() as disc_tape:
            _, _, _, x_pu, _ = self.generate(x_pl, x_u, self.config['mode'])

            d_x_pu = self.model_disc.discriminate(x_pu, sigmoid=False)
            d_x_u = self.model_disc.discriminate(x_u, sigmoid=False)

            label_pu = tf.zeros_like(d_x_pu)
            label_u = tf.ones_like(d_x_u)

            disc_loss1 = alpha_disc * (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_pu, logits=d_x_pu)) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_u, logits=d_x_u)))

            disc_loss = disc_loss1

        gradients_of_disc = disc_tape.gradient(disc_loss, self.model_disc.trainable_variables)
        self.opt_disc.apply_gradients(zip(gradients_of_disc, self.model_disc.trainable_variables))

        return disc_loss.numpy()

    def sigmoid_loss(self, t, y):
        return tf.nn.sigmoid(-t*y)

    def logistic_loss(self, t, y):
        return tf.math.softplus(-t*y)

    def ce_loss(self, t, y):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=t)

    def train_step_pn_pre(self, x_pl, x_u):

        pi_pl = self.config['pi_pl']
        pi_pu = self.config['pi_pu']
        pi_u = self.config['pi_u']

        pi_p = pi_pl + pi_pu

        with tf.GradientTape() as pn_tape:

            pn_x_pl = self.model_pn.classify(x_pl, sigmoid=False)
            pn_x_u = self.model_pn.classify(x_u, sigmoid=False)

            pu1_loss = tf.reduce_mean(self.sigmoid_loss(pn_x_pl, tf.ones_like(pn_x_pl)))
            pu2_loss = tf.reduce_mean(- pi_p * self.sigmoid_loss(pn_x_pl, -tf.ones_like(pn_x_pl)))
            u_loss = tf.reduce_mean(pi_u * self.sigmoid_loss(pn_x_u, -tf.ones_like(pn_x_u)))
            if tf.greater_equal(pu2_loss + u_loss, 0):
                pn_loss = pu1_loss + pu2_loss + u_loss
            else:
                pn_loss = - (pu2_loss + u_loss)
        gradients_of_pn = pn_tape.gradient(pn_loss, self.model_pn.trainable_variables)
        self.opt_pn.apply_gradients(zip(gradients_of_pn, self.model_pn.trainable_variables))

        return (pu1_loss).numpy()

    def train_step_pn(self, x_pl, x_u):

        pi_pl = self.config['pi_pl']

        ##### v2.0 [201122] #####
        pi_pu = self.config['pi_pu']

        # if you want to train the parameters (pi, mu, sigma)
        #pi_pu = self.p
        ##### v2.0 [201122] #####
        pi_u = self.config['pi_u']

        _, _, _, x_pu, _ = self.generate(x_pl, x_u, self.config['mode'])


        with tf.GradientTape() as pn_tape:

            pn_x_pl = self.model_pn.classify(x_pl, sigmoid=False)
            pn_x_pu = self.model_pn.classify(x_pu, sigmoid=False)
            pn_x_u = self.model_pn.classify(x_u, sigmoid=False)

            pl_loss = tf.reduce_mean(pi_pl * self.sigmoid_loss(pn_x_pl, tf.ones_like(pn_x_pl)))
            pu1_loss = tf.reduce_mean(pi_pu * self.sigmoid_loss(pn_x_pu, tf.ones_like(pn_x_pu)))
            pu2_loss = tf.reduce_mean(- pi_pu * self.sigmoid_loss(pn_x_pu, -tf.ones_like(pn_x_pu)))
            u_loss = tf.reduce_mean(pi_u * self.sigmoid_loss(pn_x_u, -tf.ones_like(pn_x_u)))

            if tf.greater_equal(pu2_loss + u_loss, 0):
                pn_loss = pl_loss + pu1_loss + pu2_loss + u_loss
            else:
                pn_loss = - (pu2_loss + u_loss)

        gradients_of_pn = pn_tape.gradient(pn_loss, self.model_pn.trainable_variables)
        self.opt_pn.apply_gradients(zip(gradients_of_pn, self.model_pn.trainable_variables))

        return (pl_loss + pu1_loss + pu2_loss + u_loss).numpy()

    def compare(self, fname, x_pl, x_u, n=1):

        lst_x_pu = []
        plt.figure(figsize=(10, 10))

        for _ in range(3):
            h_y, h_o, ne_h_o, x_pu, _ = self.generate(x_pl, x_u, self.config['mode'])
            lst_x_pu.append(x_pu)

        recon_x_pl = self.model_de.decode(h_y, h_o, sigmoid=True)

        if 'MNIST' in self.config['data']:

            for i in range(1, n + 1):
                plt.subplot(n, 5, (i - 1) * 5 + 1)
                plt.imshow(x_pl[i - 1].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
                plt.title("real")

                plt.subplot(n, 5, (i - 1) * 5 + 2)
                plt.imshow(recon_x_pl[i - 1].numpy().reshape(28, 28), vmin=0, vmax=1, cmap="gray")
                plt.title("fake_pl")

                plt.subplot(n, 5, (i - 1) * 5 + 3)
                plt.imshow(lst_x_pu[0][i - 1].numpy().reshape(28, 28), vmin=0, vmax=1, cmap="gray")
                plt.title("fake_pu1")

                plt.subplot(n, 5, (i - 1) * 5 + 4)
                plt.imshow(lst_x_pu[1][i - 1].numpy().reshape(28, 28), vmin=0, vmax=1, cmap="gray")
                plt.title("fake_pu2")

                plt.subplot(n, 5, (i - 1) * 5 + 5)
                plt.imshow(lst_x_pu[2][i - 1].numpy().reshape(28, 28), vmin=0, vmax=1, cmap="gray")
                plt.title("fake_pu3")

        else:
            for i in range(1, n + 1):
                plt.subplot(n, 5, (i - 1) * 5 + 1)
                plt.imshow(x_pl[i - 1], vmin=0, vmax=1, cmap="gray")
                plt.title("real")

                plt.subplot(n, 5, (i - 1) * 5 + 2)
                plt.imshow(recon_x_pl[i - 1], vmin=0, vmax=1, cmap="gray")
                plt.title("fake_pl")

                plt.subplot(n, 5, (i - 1) * 5 + 3)
                plt.imshow(lst_x_pu[0][i - 1], vmin=0, vmax=1, cmap="gray")
                plt.title("fake_pu1")

                plt.subplot(n, 5, (i - 1) * 5 + 4)
                plt.imshow(lst_x_pu[1][i - 1], vmin=0, vmax=1, cmap="gray")
                plt.title("fake_pu2")

                plt.subplot(n, 5, (i - 1) * 5 + 5)
                plt.imshow(lst_x_pu[2][i - 1], vmin=0, vmax=1, cmap="gray")
                plt.title("fake_pu3")

        plt.savefig(fname)
        plt.close()

    def check_disc(self, x_pl, x_u):

        _, _, _, x_pu, _ = self.generate(x_pl, x_u, self.config['mode'])

        d_x_pu = self.model_disc.discriminate(x_pu, sigmoid=True)
        d_x_u = self.model_disc.discriminate(x_u, sigmoid=True)

        return d_x_pu, d_x_u

    def check_pn(self, x_pl, x_u):

        _, _, _, x_pu, _ = self.generate(x_pl, x_u, self.config['mode'])

        d_x_pu = self.model_pn.classify(x_pu, sigmoid=True)
        d_x_pl = self.model_pn.classify(x_pl, sigmoid=True)

        return d_x_pu, d_x_pl

    def accuracy(self, dataset):

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for (x_val, y_val) in dataset:
            c = self.model_pn.classify(x_val, sigmoid=False)

            tp = tp + tf.reduce_sum(tf.cast(tf.greater(tf.boolean_mask(c, tf.equal(y_val, 1)), 0), tf.int32)).numpy()
            fn = fn + tf.reduce_sum(tf.cast(tf.less_equal(tf.boolean_mask(c, tf.equal(y_val, 1)), 0), tf.int32)).numpy()
            fp = fp + tf.reduce_sum(tf.cast(tf.greater(tf.boolean_mask(c, tf.equal(y_val, -1)), 0), tf.int32)).numpy()
            tn = tn + tf.reduce_sum(tf.cast(tf.less_equal(tf.boolean_mask(c, tf.equal(y_val, -1)), 0), tf.int32)).numpy()

        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)

        acc = (tp + tn) / (tp + tn + fp + fn)

        return acc, precision, recall

    def loss_val(self, x_pl, x_u):

        pi_pl = self.config['pi_pl']
        pi_pu = self.config['pi_pu']
        pi_u = self.config['pi_u']

        pi_p = pi_pl + pi_pu

        _, _, _, x_pu, _ = self.generate(x_pl, x_u, self.config['mode'])

        pn_x_pl = self.model_pn.classify(x_pl, sigmoid=False)
        pn_x_pu = self.model_pn.classify(x_pu, sigmoid=False)
        pn_x_u = self.model_pn.classify(x_u, sigmoid=False)

        pl_loss = tf.reduce_mean(pi_pl * self.sigmoid_loss(pn_x_pl, tf.ones_like(pn_x_pl)))
        pu1_loss = tf.reduce_mean(pi_pu * self.sigmoid_loss(pn_x_pu, tf.ones_like(pn_x_pu)))
        pu2_loss = tf.reduce_mean(- pi_pu * self.sigmoid_loss(pn_x_pu, -tf.ones_like(pn_x_pu)))
        u_loss = tf.reduce_mean(pi_u * self.sigmoid_loss(pn_x_u, -tf.ones_like(pn_x_u)))

        return (pl_loss + pu1_loss + pu2_loss + u_loss).numpy()

    def check_cl(self, x_pl, x_u):
        np.random.shuffle(x_pl)
        np.random.shuffle(x_u)
        x_pl = x_pl[:10]
        x_u = x_u[:10]

        _, h_o, ne_h_o, _ = self.generate(x_pl, x_u)

        c_h_o = self.model_cl.classify(h_o, sigmoid=True)
        c_ne_h_o = self.model_cl.classify(ne_h_o, sigmoid=True)

        _, _, h_o_mu_2, h_o_log_sig_sq_2 = self.model_en.encode(x_u)
        h_o_2 = self.reparameterization(h_o_mu_2, h_o_log_sig_sq_2)

        c_h_o_2 = self.model_cl.classify(h_o_2, sigmoid=True)

        return c_h_o, c_ne_h_o, c_h_o_2
