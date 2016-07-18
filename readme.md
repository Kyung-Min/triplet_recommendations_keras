
# Recommendations in Keras using triplet loss
Note: These codes show how to implement ranking loss using Keras Graph function.
      The original work is https://github.com/maciejkula/triplet_recommendations_keras and I slightly revise codes to follow updated Keras version.
      The codes using Keras Model function will be uploaded soon.

Along the lines of BPR [1]. 

[1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence. AUAI Press, 2009.

This is implemented (more efficiently) in LightFM (https://github.com/lyst/lightfm). See the MovieLens example (https://github.com/lyst/lightfm/blob/master/examples/movielens/example.ipynb) for results comparable to this notebook.

## Set up the architecture
A simple dense layer for both users and items: this is exactly equivalent to latent factor matrix when multiplied by binary user and item indices. There are three inputs: users, positive items, and negative items. In the triplet objective we try to make the positive item rank higher than the negative item for that user.

Because we want just one single embedding for the items, we use shared weights for the positive and negative item inputs (a siamese architecture).

This is all very simple but could be made arbitrarily complex, with more layers, conv layers and so on. I expect we'll be seeing a lot of papers doing just that.



```python
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

```

## Load and transform data
We're going to load the Movielens 100k dataset and create triplets of (user, known positive item, randomly sampled negative item).

The success metric is AUC: in this case, the probability that a randomly chosen known positive item from the test set is ranked higher for a given user than a ranomly chosen negative item.


```python
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
user_features, positive_item_features, negative_item_features = data.get_dense_triplets(uid,
                                                                                        pid,
                                                                                        nid,
                                                                                        num_users,
                                                                                        num_items)

model = get_graph(num_users, num_items, 256)

# Print the model structure
print(model.summary())

# Sanity check, should be around 0.5
print('AUC before training %s' % metrics.full_auc(model, test))
```
    --------------------------------------------------------------------------------
    Layer (type)                       Output Shape        Param #     Connected to                     
    --------------------------------------------------------------------------------
    negative_item_input (InputLayer)   (None, 1683)        0                                            
    positive_item_input (InputLayer)   (None, 1683)        0                                            
    item_latent (Dense)                (None, 256)         431104      positive_item_input[0][0]        
                                                                       negative_item_input[0][0]        
    user_input (InputLayer)            (None, 944)         0                                            
    negative_item_latent (Layer)       (None, 256)         0           item_latent[1][0]                
    positive_item_latent (Layer)       (None, 256)         0           item_latent[0][0]                
    user_latent (Dense)                (None, 256)         241920      user_input[0][0]                 
    merge_inputs_for_user_neg (Merge)  (None, 1)           0           user_latent[0][0]                
                                                                       negative_item_latent[0][0]       
    merge_inputs_for_user_pos (Merge)  (None, 1)           0           user_latent[0][0]                
                                                                       positive_item_latent[0][0]       
    user_neg (Activation)              (None, 1)           0           merge_inputs_for_user_neg[0][0]  
    user_pos (Activation)              (None, 1)           0           merge_inputs_for_user_pos[0][0]  
    triplet_loss_out (Merge)           (None, 2)           0           user_pos[0][0]                   
                                                                       user_neg[0][0]                   
    --------------------------------------------------------------------------------
    Total params: 673024
    --------------------------------------------------------------------------------

    None
    AUC before training 0.50088630047


## Run the model
Run for a couple of epochs, checking the AUC after every epoch.


```python
for epoch in range(num_epochs):

    print('Epoch %s' % epoch)

    model.fit({'user_input': user_features,
               'positive_item_input': positive_item_features,
               'negative_item_input': negative_item_features,
               'triplet_loss_out': np.ones(len(uid))},
              validation_data={'user_input': test_user_features,
                               'positive_item_input': test_positive_item_features,
                               'negative_item_input': test_negative_item_features,
                               'triplet_loss': np.ones(test_user_features.shape[0])},
              batch_size=512,
              nb_epoch=1, 
              verbose=2,
              shuffle=True)

    print('AUC %s' % metrics.full_auc(model, test))
    
```

    Epoch 0
    Train on 49906 samples, validate on 5469 samples
    Epoch 1/1
    1s - loss: -5.5905e-01 - val_loss: -7.3272e-01
    AUC 0.813442134934
    Epoch 1
    Train on 49906 samples, validate on 5469 samples
    Epoch 1/1
    1s - loss: -8.0206e-01 - val_loss: -8.2955e-01
    AUC 0.845866109576
    Epoch 2
    Train on 49906 samples, validate on 5469 samples
    Epoch 1/1
    1s - loss: -8.3717e-01 - val_loss: -8.3969e-01
    AUC 0.845286876593
    Epoch 3
    Train on 49906 samples, validate on 5469 samples
    Epoch 1/1
    1s - loss: -8.4702e-01 - val_loss: -8.4421e-01
    AUC 0.843973516913
    Epoch 4
    Train on 49906 samples, validate on 5469 samples
    Epoch 1/1
    1s - loss: -8.5271e-01 - val_loss: -8.4662e-01
    AUC 0.842522501549


The AUC is in the mid-80s. At some point we start overfitting, so it would be a good idea to stop early or add some regularization.
