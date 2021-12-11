config = {
    'directory': './result/MNIST_35_y/Exp1/', 'data': 'MNIST',
    'num_epoch': 500, 'num_epoch_pn': 100, 'num_epoch_pre': 100, 'batch_size_l': 50, 'batch_size_u': 100,  'batch_size_l_pn': 50, 'batch_size_u_pn': 100, 'num_buff': 100000,
    'batch_size_val': 100, 'batch_size_test': 100,
    'n_input': 784, 'n_hidden_vae_e': [500, 500], 'n_h_y' : 100, 'n_h_o' : 100, 'n_hidden_vae_d': [500, 500], 'n_o': 2,
    'n_hidden_nevae_e' : [50], 'n_z': 50, 'n_hidden_nevae_d': [50], 'n_hidden_disc': [256], 'n_hidden_cl': [],  'n_hidden_pn': [300, 300, 300, 300],
    'alpha_gen': 1., 'alpha_disc': 1. , 'alpha_vae': 1., 'alpha_cl': 1., 'alpha_ne': 1., 'alpha_mi' : 1., 'alpha_o' : 1., 'alpha_vade' : 1., 'alpha_gen2': 1., 'alpha_disc2':1.,
    'pi_pl': 100/10000, 'pi_pu': 4900/10000, 'pi_u': 9900/10000,
    'lr_pu': 3*1e-4, 'lr_disc':3*1e-4, 'lr_pn': 3*1e-4, 'beta1': 0.9, 'beta2': 0.999, 'mode': 'near_o', 'k_gan': 1,
    'save_epoch': 1000, 'num_repeat': 1,
    'pi_given': None, 'bool_pn_pre': False
}