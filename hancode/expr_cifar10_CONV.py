import os
import sys
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from dataset import CIFAR10
from preprocess import SubtractMeanAndNormalizeH, ZCA
from layer import ReluConv2DLayer, MaxPoolingLayer, ReluLayer, LinearLayer, ZerobiasLayer
from classifier import LogisticRegression
from train import GraddescentMinibatch, Dropout
from params import save_params, load_params, set_params, get_params

import pdb


#######################
# SET SUPER PARAMETER #
#######################

batchsize = 100
momentum = 0.9

weightdecay = 0.01
finetune_lr = 1e-2
finetune_epc = 400

print " "
print "batchsize =", batchsize
print "momentum =", momentum
print "finetune:            lr = %f, epc = %d" % (finetune_lr, finetune_epc)

#############
# LOAD DATA #
#############

cifar10_data = CIFAR10()
train_x, train_y = cifar10_data.get_train_set()
test_x, test_y = cifar10_data.get_test_set()

print "\n... pre-processing"
preprocess_model = SubtractMeanAndNormalizeH(train_x.shape[1])
map_fun = theano.function([preprocess_model.varin], preprocess_model.output())
train_x = map_fun(train_x.astype(theano.config.floatX)).reshape((50000, 32, 32, 3), order='F').swapaxes(1, 3)
test_x = map_fun(test_x.astype(theano.config.floatX)).reshape((10000, 32, 32, 3), order='F').swapaxes(1, 3)

train_x = theano.shared(value=train_x, name='train_x', borrow=True)
train_y = theano.shared(value=train_y, name='train_y', borrow=True)
test_x = theano.shared(value=test_x, name='test_x', borrow=True)
test_y = theano.shared(value=test_y, name='test_y', borrow=True)
print "Done."

###############
# BUILD MODEL #
###############

print "... building model"
l0_n_in = (batchsize, 3, 32, 32)
l0_filter_shape=(128, l0_n_in[1], 5, 5)
l0_poolsize = (3, 3)
l0_stride = (2, 2)

l1_filter_shape=(192, l0_filter_shape[0], 5, 5)
l1_poolsize = (2, 2)
l1_stride = (2, 2)

l2_filter_shape=(192, l1_filter_shape[0], 3, 3)
l2_poolsize = (2, 2)
l2_stride = (2, 2)

npy_rng = numpy.random.RandomState(123)
model = ReluConv2DLayer(
    n_in=l0_n_in, filter_shape=l0_filter_shape, npy_rng=npy_rng
) + MaxPoolingLayer(
    pool_size=l0_poolsize, stride=l0_stride, ignore_border=True
) + ReluConv2DLayer(
    filter_shape=l1_filter_shape, npy_rng=npy_rng
) + MaxPoolingLayer(
    pool_size=l1_poolsize, stride=l1_stride, ignore_border=True
) + ReluConv2DLayer(
    filter_shape=l2_filter_shape, npy_rng=npy_rng
) + MaxPoolingLayer(
    pool_size=l2_poolsize, stride=l2_stride, ignore_border=True
) + ReluLayer(None, 1600, npy_rng=npy_rng
) + ReluLayer(None, 1600, npy_rng=npy_rng
) + LogisticRegression(None, 10, npy_rng=npy_rng)

model.print_layer()


# compile error rate counters:
index = T.lscalar()
truth = T.lvector('truth')
train_set_error_rate = theano.function(
    [index],
    T.mean(T.neq(model.models_stack[-1].predict(), truth)),
    givens = {model.varin : train_x[index * batchsize: (index + 1) * batchsize],
              truth : train_y[index * batchsize: (index + 1) * batchsize]},
)
def train_error():
    return numpy.mean([train_set_error_rate(i) for i in xrange(50000/batchsize)])

test_set_error_rate = theano.function(
    [index],
    T.mean(T.neq(model.models_stack[-1].predict(), truth)),
    givens = {model.varin : test_x[index * batchsize: (index + 1) * batchsize],
              truth : test_y[index * batchsize: (index + 1) * batchsize]},
)
def test_error():
    return numpy.mean([test_set_error_rate(i) for i in xrange(10000/batchsize)])
print "Done."
print "Initial error rate: train: %f, test: %f" % (train_error(), test_error())

#############
# FINE-TUNE #
#############

print "\n\n... BP through the whole network"
trainer = GraddescentMinibatch(
    varin=model.varin, data=train_x, 
    truth=model.models_stack[-1].vartruth, truth_data=train_y,
    supervised=True,
    cost=model.models_stack[-1].cost(), 
    params=model.params,
    batchsize=batchsize, learningrate=finetune_lr, momentum=momentum,
    rng=npy_rng
)

init_lr = trainer.learningrate
prev_cost = numpy.inf
for epoch in xrange(finetune_epc):
    cost = trainer.epoch()
    if prev_cost <= cost:
        if trainer.learningrate < (init_lr * 1e-7):
            break
        trainer.set_learningrate(trainer.learningrate*0.8)
    prev_cost = cost
    if epoch % 10 == 0:
        print "*** error rate: train: %f, test: %f" % (train_error(), test_error())
    try:
        if epoch % 100 == 0:
            save_params(model, 'CONV_3-4-3_32-48-64_256_1600-400-1600-10.npy')
    except:
        pass
print "***FINAL error rate: train: %f, test: %f" % (train_error(), test_error())
print "Done."

pdb.set_trace()
