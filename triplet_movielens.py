"""
Triplet loss network example for recommenders
"""


from __future__ import print_function

import numpy as np

import theano

import keras
from keras import backend as K
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Lambda
from keras.optimizers import Adagrad, Adam


import data
import metrics


def ranking_loss(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    loss = -K.sigmoid(pos-neg) # use loss = K.maximum(1.0 + neg - pos, 0.0) if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true


def get_graph(num_users, num_items, latent_dim):

    model = Graph()
    model.add_input(name='user_input', input_shape=(num_users,))
    model.add_input(name='positive_item_input', input_shape=(num_items,))
    model.add_input(name='negative_item_input', input_shape=(num_items,))

    model.add_node(layer=Dense(latent_dim, input_shape = (num_users,)),
                   name='user_latent',
                   input='user_input')
    model.add_shared_node(layer=Dense(latent_dim, input_shape = (num_items,)), 
                          name='item_latent', 
                          inputs=['positive_item_input', 'negative_item_input'],
                          merge_mode=None, 
                          outputs=['positive_item_latent', 'negative_item_latent'])

    model.add_node(layer=Activation('linear'), name='user_pos', inputs=['user_latent', 'positive_item_latent'], merge_mode='dot', dot_axes=1)
    model.add_node(layer=Activation('linear'), name='user_neg', inputs=['user_latent', 'negative_item_latent'], merge_mode='dot', dot_axes=1)

    model.add_output(name='triplet_loss_out', inputs=['user_pos', 'user_neg'])
    model.compile(loss={'triplet_loss_out': ranking_loss}, optimizer=Adam())#Adagrad(lr=0.1, epsilon=1e-06))

    return model

if __name__ == '__main__':

    num_epochs = 5

    # Read data
    train, test = data.get_movielens_data()
    num_users, num_items = train.shape

    # Prepare the test triplets
    test_uid, test_pid, test_nid = data.get_triplets(test)
    test_user_features, test_positive_item_features, test_negative_item_features = data.get_dense_triplets(test_uid,
                                                                                                           test_pid,
                                                                                                           test_nid,
                                                                                                           num_users,
                                                                                                           num_items)

    # Sample triplets from the training data
    uid, pid, nid = data.get_triplets(train)
    user_features, positive_item_features, negative_item_features = data.get_dense_triplets(uid, pid, nid, num_users, num_items)

    model = get_graph(num_users, num_items, 256)

    # Print the model structure
    print(model.summary())

    # Sanity check, should be around 0.5
    print('AUC before training %s' % metrics.full_auc(model, test))

    for epoch in range(num_epochs):

        print('Epoch %s' % epoch)

        model.fit({'user_input': user_features,
                   'positive_item_input': positive_item_features,
                   'negative_item_input': negative_item_features,
                   'triplet_loss_out': np.ones(len(uid))
                   },
                  validation_data=({'user_input': test_user_features,
                                   'positive_item_input': test_positive_item_features,
                                   'negative_item_input': test_negative_item_features,
                                   'triplet_loss_out': np.ones(test_user_features.shape[0])}), batch_size=512, nb_epoch=1, verbose=2,shuffle=True)

        print('AUC %s' % metrics.full_auc(model, test))
