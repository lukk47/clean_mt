# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""CIFAR-10 final evaluation"""

import logging
import sys
sys.path.append('./')
from experiments.run_context import RunContext
import tensorflow as tf

# from datasets import Cifar10ZCA
import datasets
from mean_teacher.mean_teacher import mean_teacher
from mean_teacher import minibatching
import os

LOG = logging.getLogger('main')

flags = tf.app.flags
flags.DEFINE_integer('gpu', 0, 'GPU_number')
flags.DEFINE_string('n_labeled', '1000', 'labeled data')
flags.DEFINE_integer('dataset_index', 0, 'datasets including Cifar10ZCA, SVHN')
flags.DEFINE_integer('n_runs', 1, 'number of runs')
FLAGS = flags.FLAGS

datasets_name = ['Cifar10ZCA','SVHN']
assert FLAGS.dataset_index<= len(datasets_name), 'wrong dataset index'
print('{} is loaded'.format(datasets_name[FLAGS.dataset_index]))
data_loader = getattr(datasets, datasets_name[FLAGS.dataset_index])

def parameters():
    test_phase = True
    n_labeled = FLAGS.n_labeled
    if n_labeled == 'all':
        n_runs = FLAGS.n_runs
    else:
        n_runs = FLAGS.n_runs
    for data_seed in range(2000, 2000 + n_runs):
        yield {
            'test_phase': test_phase,
            'n_labeled': n_labeled,
            'data_seed': data_seed
        }

def run(test_phase, n_labeled, data_seed):

    minibatch_size = 100

    data = data_loader(n_labeled=n_labeled,
                       data_seed=data_seed,
                       test_phase=test_phase)

    if n_labeled == 'all':
        n_labeled_per_batch =  minibatch_size
        max_consistency_cost = minibatch_size
    else:
        n_labeled_per_batch = 'vary'
        max_consistency_cost = minibatch_size* int(n_labeled) / data['num_train']

    hyper_dcit = {'input_dim': data['input_dim'],
                'label_dim': data['label_dim'],
                'cnn':'tower',
                'flip_horizontally':True,
                'max_consistency_cost': max_consistency_cost,
                'adam_beta_2_during_rampup': 0.999,
                'ema_decay_during_rampup': 0.999,
                'normalize_input': False,
                'rampdown_length': 25000,
                'training_length': 150000 }

    tf.reset_default_graph()
    model = mean_teacher(RunContext(__file__, data_seed), hyper_dcit)

    training_batches = minibatching.training_batches(data.training,
                                                     minibatch_size,
                                                     n_labeled_per_batch)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(data.evaluation,
                                                                    minibatch_size)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    for run_params in parameters():
        run(**run_params)
