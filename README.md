# VAE-PU: Deep Generative Positive-Unlabeled Learning under Selection Bias (CIKM '20)

| [paper](https://dl.acm.org/doi/10.1145/3340531.3411971) | [data](https://drive.google.com/file/d/1iCcYyJeuvWLqvOSlOUItIvDDiQQ4oNKa/view?usp=sharing) | 

Official Tensorflow implementation for Deep Generative Positive-Unlabeled Learning under Selection Bias (CIKM '20).


## Requirements

This work was tested with CUDA 10.1, Python 3.5, Tensorflow 1.13.2, keras 2.2.4. \
Please refer to `requirements.txt`.

## Run code

1. Download train and validation data in `/<work_dir>/pu_data/` from above data link.
2. Run main.py (May need to change config dictionary)

## Hyperparameters

We use hyperparameters to balance of the VAE and Regularization losses. In this code, we implemented these hyperparameters by `config['alpha_gen'], config['alpha_disc'], config['alpha_gen2'], config['alpha_disc2']`. \
Below table provides the hyperparameters that we used for each dataset.

* Random Labelling

Hyperparameter | MNIST_35 | MNIST_EO | CIFAR-10 | 20News
--------- | --------- | --------- | --------- | ---------
alpha_gen | 0.1 | 0.1 | 0.3 | 0.01 |
alpha_disc | 0.1 | 0.1 | 0.3 | 0.01 |
alpha_gen2 | 3 | 1 | 1 | 1 |
alpha_disc2 | 3 | 1 | 1 | 1 |

* Bias Labelling

Hyperparameter | MNIST_35 | MNIST_EO | CIFAR-10 | 20News
--------- | --------- | --------- | --------- | ---------
alpha_gen | 1 | 1 | 3 | 1 |
alpha_disc | 1 | 1 | 3 | 1 |
alpha_gen2 | 10 | 3 | 1 | 1 |
alpha_disc2 | 10 | 3 | 1 | 1 |

## Citation

Please cite this work in your publications if it helps your research.

~~~
@inproceedings{10.1145/3340531.3411971,
author = {Na, Byeonghu and Kim, Hyemi and Song, Kyungwoo and Joo, Weonyoung and Kim, Yoon-Yeong and Moon, Il-Chul},
title = {Deep Generative Positive-Unlabeled Learning under Selection Bias},
year = {2020},
isbn = {9781450368599},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3340531.3411971},
doi = {10.1145/3340531.3411971},
abstract = {Learning in the positive-unlabeled (PU) setting is prevalent in real world applications. Many previous works depend upon theSelected Completely At Random (SCAR) assumption to utilize unlabeled data, but the SCAR assumption is not often applicable to the real world due to selection bias in label observations. This paper is the first generative PU learning model without the SCAR assumption. Specifically, we derive the PU risk function without the SCAR assumption, and we generate a set of virtual PU examples to train the classifier. Although our PU risk function is more generalizable, the function requires PU instances that do not exist in the observations. Therefore, we introduce the VAE-PU, which is a variant of variational autoencoders to separate two latent variables that generate either features or observation indicators. The separated latent information enables the model to generate virtual PU instances. We test the VAE-PU on benchmark datasets with and without the SCAR assumption. The results indicate that the VAE-PU is superior when selection bias exists, and the VAE-PU is also competent under the SCAR assumption. The results also emphasize that the VAE-PU is effective when there are few positive-labeled instances due to modeling on selection bias.},
booktitle = {Proceedings of the 29th ACM International Conference on Information &amp; Knowledge Management},
pages = {1155–1164},
numpages = {10},
keywords = {positive-unlabeled learning, variational autoencoders, selection bias},
location = {Virtual Event, Ireland},
series = {CIKM '20}
}
~~~
