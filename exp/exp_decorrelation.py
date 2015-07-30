
# coding: utf-8

# In[1]:
import matplotlib as mpl
mpl.use('Agg')
import os
from lasagne.generative.autoencoder import Autoencoder, greedy_learn_with_validation

from lasagne.easy import BatchOptimizer, LightweightModel
from lasagne.datasets.mnist import MNIST

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from lasagne import layers, updates, init, nonlinearities
import theano.tensor as T
from theano.sandbox import rng_mrg
import theano
import matplotlib.pyplot as plt
import numpy as np
from lasagne.layers import get_all_layers

from skimage.io import imread
from skimage.filter import threshold_otsu
from skimage.transform import resize

import matplotlib.pyplot as plt

from lasagne import easy

from lasagne.generative.capsule import Capsule
from lasagne.easy import BatchIterator
import glob
import os

from lasagne.easy import BatchOptimizer, LightweightModel
from lasagne import init
from collections import OrderedDict
from lasagne import init, layers, updates, nonlinearities
from lasagne.layers.helper import get_all_layers
from lasagne.layers import helper
import theano.tensor as T
from theano.sandbox import rng_mrg
from sklearn.cross_validation import train_test_split
from lasagne.datasets.fonts import Fonts
import theano
from collections import OrderedDict
import theano.tensor as T

from lasagne.generative.capsule import Capsule
from lasagne.layers import Layer
from lasagne.datasets.fonts import Fonts


from lightexperiments.light import Light
light = Light()

light.launch()
light.initials() # save the date and init the timer

light.file_snapshot() # save the content of the python file running
seed = 1234
np.random.seed(seed)
light.set_seed(seed) # save the content of the seed
light.tag("decorrelation") # for tagging your experiments
light.tag("vary_hidden_factors")


def binarize(X):
    X_b = np.empty(X.shape, dtype=X.dtype)
    for i in range(X.shape[0]):
        X_b[i] = 1. * (X[i] <= threshold_otsu(X[i]))
    return X_b

def resize_all(X, w, h):
    if X.shape[1] == w and X.shape[2] == h:
        return X
    X_b = np.empty((X.shape[0], w, h), dtype=X.dtype)
    for i in range(X.shape[0]):
        X_b[i] = resize(X[i], (w, h))
    return X_b  

class SumLayer(Layer):
    def __init__(self, 
                 incoming,
                 axis=1,
                 **kwargs):
        super(SumLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_for(self, input, **kwargs):
        return input.sum(axis=self.axis)
    
    def get_output_shape_for(self, input):
        shape = list(self.input_shape)
        del shape[self.axis]
        return tuple(shape)




# # Load & pre-process data

# In[2]:


dataset = "mnist"
light.set("dataset", dataset)


if dataset == "font":
	data = Fonts(kind="all_64", labels_kind="letters")
	data.load()
	X = data.X
	X = X.astype(np.float32)
	y = data.y.astype(np.int32)
	nb_outputs = 26
elif dataset == "mnist":
	data = MNIST()
	data.load()
	X = data.X
	X = X.astype(np.float32)
	y = data.y.astype(np.int32)
	nb_outputs = 10



# In[3]:

w, h = 28, 28
light.set("w", w)
light.set("h", h)


# In[4]:

rescale = False

if rescale:
	from skimage.filter import threshold_otsu
	from skimage.transform import resize
	X_b = np.zeros((X.shape[0], w, h))
	for i in range(X_b.shape[0]):
		X_b[i] = resize(X[i].reshape((64, 64)), (w, h))
	X = X_b
	#X = X <= threshold_otsu(X)
	X = X.astype(np.float32)
	X = X.reshape((X.shape[0], w*h))
	X=1-X

light.set("rescaled", rescale)


# In[5]:

from sklearn.preprocessing import label_binarize
y = label_binarize(y, np.arange(nb_outputs))
y = y.astype(np.float32)


# In[6]:

output_dim = y.shape[1]


# In[7]:

#plt.imshow(X[120].reshape((28, 28)), cmap="gray")


# In[8]:

X, y = shuffle(X, y, random_state=seed)
train, test = train_test_split(range(X.shape[0]), test_size=0.25)


# In[9]:

nb_samples_learning_curve = 1000
nb_tries_learning_curve = 10

light.set("nb_samples_learning_curve", nb_samples_learning_curve)
light.set("nb_tries_learning_curve", nb_tries_learning_curve)

class MyBatchOptimizer(BatchOptimizer):
    
    def iter_update(self, epoch, nb_batches, iter_update_batch):
        status = super(MyBatchOptimizer, self).iter_update(epoch, nb_batches, iter_update_batch)
        
        all_of_them = [
            "acc_train",
            "acc_valid",
            "rec_train",
            "rec_valid",
            "crosscor_train",
            "crosscor_valid"
        ]
        for a in all_of_them:
            status[a] = 0.
        for i in range(nb_tries_learning_curve):
        
            s = np.arange(len(train))
            np.random.shuffle(s)
            s_train = s[0:nb_samples_learning_curve]


            s = np.arange(len(test))
            np.random.shuffle(s)
            s_test = s[0:nb_samples_learning_curve]


            status["acc_train"] += (self.model.predict(X[train][s_train])==y[train][s_train].argmax(axis=1)).mean()
            status["acc_valid"] += (self.model.predict(X[test][s_test])==y[test][s_test].argmax(axis=1)).mean()

            status["rec_train"] += self.model.get_reconstruction_error(X[train][s_train])
            status["rec_valid"] += self.model.get_reconstruction_error(X[test][s_test])

            status["crosscor_train"] += self.model.get_cross_correlation(X[train][s_train])
            status["crosscor_valid"] += self.model.get_cross_correlation(X[test][s_test])
        for a in all_of_them:
            status[a] /= nb_tries_learning_curve
        for k, v in status.items():
            light.append(k, float(v))
        return status


# ## Model definition

# In[10]:

from lasagne.layers import cuda_convnet, Conv2DLayer

def cross_entropy(truth, pred):
    return -(truth * T.log(pred) + (1 - truth) * T.log(1 - pred)).sum(axis=1).mean()

def mse(truth, pred):
    return (((truth - pred) ** 2).sum(axis=1)).mean()

def loss_function_y(y_true, y_pred):
    return (T.nnet.categorical_crossentropy(y_pred, y_true)).mean()
        
def corrupted_masking_noise(rng, x, corruption_level):
    return rng.binomial(size=x.shape, n=1, p=1 - corruption_level) * x

def corrupted_salt_and_pepper(rng, x, corruption_level):
    selected = rng.binomial(size=x.shape, n=1, p=corruption_level, dtype=theano.config.floatX)
    return x * (1 - selected) + selected * rng.binomial(size=x.shape, n=1, p=0.5, dtype=theano.config.floatX)

rng = rng_mrg.MRG_RandomStreams(seed)
 
def corruption_function(X):
    return corrupted_salt_and_pepper(rng, X, 0.5)
    

class Model:
    def get_all_params(self, **t):
        return list(set(self.x_to_z.get_all_params(**t) + 
                        self.x_to_y.get_all_params(**t) + 
                        self.z_to_x.get_all_params(**t)))


# ### model type

# In[11]:

model_type = "convnet" # or "convnet"
light.set(model_type, model_type)

# ### Fully connected

# In[12]:
latent_size = 5
if model_type == "fully_connected":
    ## fully connected
    num_hidden_units = 2000
    light.set("latent_size", latent_size)
    light.set("num_hidden_units", num_hidden_units)

    l_in = layers.InputLayer((None, w*h))
    input_dim = w*h
    output_dim = y.shape[1]

    # encoder
    l_encoder1 = layers.DenseLayer(l_in, num_units=num_hidden_units)
    l_encoder2 = layers.DenseLayer(l_encoder1, num_units=num_hidden_units)
    l_encoder3 = layers.DenseLayer(l_encoder2, num_units=num_hidden_units)
    l_encoder4 = layers.DenseLayer(l_encoder3, num_units=num_hidden_units)

    # learned representation
    l_observed = layers.DenseLayer(l_encoder4, num_units=output_dim,
                                      nonlinearity=T.nnet.softmax)

    l_latent = layers.DenseLayer(l_encoder4, 
                                 num_units=latent_size,
                                 nonlinearity=None) # linear

    l_representation = layers.concat([l_observed, l_latent])

    # decoder
    l_decoder1 = layers.DenseLayer(l_representation, num_units=num_hidden_units)
    l_decoder2 = layers.DenseLayer(l_decoder1, num_units=num_hidden_units)
    l_decoder3 = layers.DenseLayer(l_decoder2, num_units=num_hidden_units)
    l_decoder4 = layers.DenseLayer(l_decoder3, num_units=num_hidden_units)
    l_decoder_out = layers.DenseLayer(l_decoder4, num_units=input_dim,
                                       nonlinearity=nonlinearities.sigmoid)

    x_to_z = LightweightModel([l_in], [l_latent])
    x_to_y = LightweightModel([l_in], [l_observed])
    z_to_x = LightweightModel([l_observed, l_latent], [l_decoder_out])
    model = Model()
    model.x_to_z = x_to_z
    model.x_to_y = x_to_y
    model.z_to_x = z_to_x


# ### Convnet

# In[34]:

if model_type == "convnet":

    ## CNN
    nb_filters=32
    size_filters=5
    nb_hidden=1000
    
    light.set("latent_size", latent_size)
    light.set("nb_filters", nb_filters)
    light.set("size_filters", size_filters)
    light.set("num_hidden_units", nb_hidden)

    nb_filters_encoder = nb_filters
    nb_filters_decoder = nb_filters
    size_filters_encoder = size_filters
    size_filters_decoder = size_filters

    l_in = layers.InputLayer((None, w*h))


    x_in_reshaped = layers.ReshapeLayer(l_in, ([0], 1, w, h))

    # conv1
    l_conv = cuda_convnet.Conv2DCCLayer(
        x_in_reshaped,
        num_filters=nb_filters_encoder,
        filter_size=(size_filters, size_filters_encoder),
        nonlinearity=nonlinearities.rectify,
        dimshuffle=True,
    )
    l_hid = layers.DenseLayer(
        l_conv,
        num_units=nb_hidden,
        nonlinearity=nonlinearities.rectify,
    )
    
    l_hid = layers.DenseLayer(
        l_hid,
        num_units=nb_hidden,
        nonlinearity=nonlinearities.rectify,
    )

    #code layer

    l_observed = layers.DenseLayer(l_hid, 
                                   num_units=output_dim,
                                    nonlinearity=T.nnet.softmax)

    l_latent = layers.DenseLayer(l_hid, 
                                 num_units=latent_size,
                                 nonlinearity=None) # linear

    hid = layers.ConcatLayer([l_latent, l_observed], axis=1)

    l_hid = layers.DenseLayer(
        hid,
        num_units=nb_hidden,
        nonlinearity=nonlinearities.rectify,
    )
    
    l_hid = layers.DenseLayer(
        l_hid,
        num_units=nb_hidden,
        nonlinearity=nonlinearities.rectify,
    )


    
    # unflatten layer
    hid = layers.DenseLayer(l_hid,
                            num_units=nb_filters_decoder * (w - size_filters_decoder + 1) * (h - size_filters_decoder + 1))
    hid = layers.ReshapeLayer(hid,
                              ([0], nb_filters_decoder, (w - size_filters_decoder + 1), (h - size_filters_decoder + 1)))

    l_unconv = Conv2DLayer(
        hid,
        num_filters=nb_filters,
        filter_size=(size_filters_decoder, size_filters_decoder),
        nonlinearity=nonlinearities.linear,
        border_mode="full"
    )
    l_unconv_sum = SumLayer(l_unconv, axis=1)

    l_decoder_out = layers.ReshapeLayer(l_unconv_sum, ([0], w*h))
    l_decoder_out = layers.NonlinearityLayer(l_decoder_out, nonlinearities.sigmoid)

    x_to_z = LightweightModel([l_in], [l_latent])
    x_to_y = LightweightModel([l_in], [l_observed])
    z_to_x = LightweightModel([l_observed, l_latent], [l_decoder_out])
    model = Model()
    model.x_to_z = x_to_z
    model.x_to_y = x_to_y
    model.z_to_x = z_to_x


# In[35]:

def cross_correlation(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return 0.5 * ((((a.dimshuffle(0, 'x', 1) * b.dimshuffle(0, 1, 'x'))).mean(axis=0))**2).sum()

input_variables = OrderedDict()
input_variables["X"] = dict(tensor_type=T.matrix)
input_variables["y"] = dict(tensor_type=T.matrix)
    

functions = dict(
    encode=dict(
        get_output=lambda model, X:model.x_to_z.get_output(X)[0],
        params=["X"]
    ),
    predict=dict(
        get_output=lambda model, X:(model.x_to_y.get_output(X)[0]).argmax(axis=1),
        params=["X"]
    ),
    reconstruct=dict(
        get_output=lambda model, X: l_decoder_out.get_output(X),
        params=["X"]
    ),
    get_reconstruction_error=dict(
        get_output=lambda model, X: ((X-l_decoder_out.get_output(X))**2).sum(axis=1).mean(),
        params=["X"]
    ),
    get_cross_correlation=dict(
        get_output=lambda model, X: cross_correlation(model.x_to_z.get_output(X)[0],
                                                      model.x_to_y.get_output(X)[0]),
        params=["X"]
    ),
    predict_proba=dict(
        get_output=lambda model, X: model.x_to_y.get_output(X)[0],
        params=["X"]
    )

)

learning_rate = 0.0001
batch_optimizer = MyBatchOptimizer(
    verbose=1,
    max_nb_epochs=100,
    batch_size=100,
    optimization_procedure=(updates.rmsprop, 
                            {"learning_rate": learning_rate})
)
light.set("learning_rate", learning_rate)
light.set("max_nb_epochs", batch_optimizer.max_nb_epochs)
light.set("batch_size", batch_optimizer.batch_size)
light.set("optimization_method", "rmsprop")
light.set("learning_rate", learning_rate)

loss_rec_coef = 1
loss_supervised_coef = 10
loss_crosscor_coef = 10

def loss_function(model, tensors):
    x_to_z, x_to_y, z_to_x = model.x_to_z, model.x_to_y, model.z_to_x
    X_batch, y_batch = tensors["X"], tensors["y"]
    
    z, = x_to_z.get_output(X_batch)

    
    y_hat, = x_to_y.get_output(X_batch)
    X_hat, = z_to_x.get_output(y_hat, z)
    
    loss_rec = ((X_hat - X_batch) ** 2).sum(axis=1).mean()
    loss_supervised = ((y_hat - y_batch)**2).sum(axis=1).mean()
    
    return  loss_rec_coef * loss_rec + loss_supervised_coef*loss_supervised + loss_crosscor_coef * cross_correlation(z, y_hat)
    
capsule = Capsule(
    input_variables, 
    model,
    loss_function,
    functions=functions,
    batch_optimizer=batch_optimizer,
)
Z_batch = T.matrix("z_batch")
capsule.decode = theano.function([Z_batch, capsule.v_tensors["y"]], 
                                  l_decoder_out.get_output({l_latent: Z_batch, 
                                                            l_observed: capsule.v_tensors["y"]}))


# ## Model graph visualization

# In[36]:

#from lasagne.misc.draw_net import draw_to_file
#draw_to_file(get_all_layers(l_decoder_out), "model.svg")

#from IPython.display import SVG
#SVG("model.svg")


# ## Training 

# In[37]:

try:
    capsule.fit(X=X[train], y=y[train])
except KeyboardInterrupt:
    print("interruption...")




# In[38]:



# ## Visualization of features

# In[39]:

from lasagne.misc.plot_weights import grid_plot

if model_type == "convnet":
    layers_enc = get_all_layers(model.x_to_z.output_layers[0])
    layers_dec = get_all_layers(model.z_to_x.output_layers[0])
    for l in layers_enc[2], layers_dec[-4]:
        #plt.clf()
        W = l.W.get_value()[:, 0]
        light.append("features", light.insert_blob(W.tolist()))
        #grid_plot(W, imshow_options={"cmap": "gray"})
        #plt.show()
elif model_type == "fully_connected":
    layers = get_all_layers(l_decoder_out)
    for W in (layers[1].W.get_value().T, layers[-1].W.get_value()):
        #plt.clf()
        W = W.reshape((W.shape[0], w, h))
        light.append("features", light.insert_blob(W.tolist()))
        #grid_plot(W, imshow_options={"cmap": "gray"}, nbrows=10, nbcols=10)
        #plt.show()


# ## Statistics

# In[40]:

from lasagne.easy import get_stat
# ## Interactive sliders

# In[50]:


use_examples = True # init/work with examples for dataset or not
nb = 100 # nb of examples to consider
max_nb_sliders = 10
T_ = train
x = X[T_][0:nb]


nb_outputs = y.shape[1]


from IPython.html.widgets import (interact, interactive, 
                                  IntSliderWidget, IntSlider, FloatSliderWidget,
                                  ButtonWidget
                                  )
from IPython.display import display # Used to display widgets in the notebook

from IPython.html.widgets import *
from IPython.html import widgets

z = capsule.encode(x)

if z.shape[1] > max_nb_sliders:
    params = np.random.choice(z.shape[1],
                              size=10)
else:
    params = np.arange(z.shape[1])

boundaries = OrderedDict()
z_mean = z.mean(axis=0)
z_std = z.std(axis=0)
for p in (params):
    boundaries["{0}".format(p)] = FloatSliderWidget(min=z_mean[p]-2*z_std[p],
                                                    max=z_mean[p]+2*z_std[p],
                                                    step=0.001,
                                                    value=0.)
d = 0
l = y[T_][d].argmax() 

def draw(**all_params):
    if use_examples is True:
        example = all_params["example"]
        del all_params["example"]
    label = all_params["label"]
    del all_params["label"]
    params = all_params
    
    if use_examples is True:
        z = capsule.encode(x[example:example + 1])
    else:
        z = np.zeros((1, latent_size), dtype="float32")
        z[0, :] = 0
    
    y_ = np.zeros(nb_outputs, dtype='float32')
    y_[label] = 1.
    y_ = y_[np.newaxis, :]
        
    for k, v in params.items():
        z[0][int(k)] = v
    plt.imshow(capsule.decode(z, y_)[0].reshape((w, h)), cmap="gray")
    

p = dict()
p.update(boundaries)

label_selector = IntSliderWidget(min=0,max=output_dim-1,step=1,value=l)
p["label"] = label_selector

if use_examples is True:
    example_selector = IntSliderWidget(min=0,max=nb-1,step=1,value=d)
    p["example"] = example_selector

i = interact(**p)



def on_button_clicked(b):
    
    example = example_selector.get_state()["value"]
    z = capsule.encode(x[example:example + 1])
    for p in params:
        w = boundaries["{0}".format(p)]
        state = w.get_state()
        state["value"] = z[0, int(p)]
        w.set_state(state)
        w.send_state(state)
        
    state = label_selector.get_state()
    state["value"] = y[example].argmax()
    label_selector.set_state(state)
    label_selector.send_state(state)
    
    
draw_i = i(draw)

if use_examples is True:
    button = widgets.ButtonWidget(description="fit!")
    display(button)
    button.on_click(on_button_clicked)

from scipy.stats import kurtosistest
_, pvalues = kurtosistest(z)
light.set("hidfactkurtosispvalues",  pvalues)


# ## Covariance matrix of hidden factors

# In[91]:

#plt.matshow(np.cov(z.T), cmap="gray")
light.set("hidfactcov", np.cov(z.T).tolist())
light.set("hidfactcorr", np.corrcoef(z.T).tolist())
#corr=(np.corrcoef(z.T))


#print(np.abs((corr)))
#print(np.abs(corr-np.diag(np.diag(corr))).max())


# In[92]:

#plt.hist(corr[(1-np.eye(corr.shape[0])).astype(np.bool)], normed=True)


# In[68]:

#c=(np.cov(z.T))

#a= (np.diag(c).mean())
#b=( np.abs(c - np.diag(np.diag(c))).sum() /   (c.shape[0]*c.shape[1]-c.shape[0])  )
#print(a/b)


# ## Visualization of samples when varying hidden factors

# In[47]:
"""
C = np.cov(z.T)
eig = np.linalg.eigvals(np.dot(C, C.T))
latent_order = np.argsort(eig)[::-1]

from lasagne.misc.plot_weights import grid_plot
labels = np.arange(output_dim)# labels to consider (by default, all)
std_units=2# nb of std units of values of latent dim around the mean to consider
labels = np.arange(output_dim)
L = latent_size

x_ = X[train][0:100]
z_ = capsule.encode(x_)
latent_std = np.std(z_, axis=0)
latent_mean = np.mean(z_, axis=0)

k = 1
for latent_dim in latent_order:
    print("hidden factor : {0}".format(latent_dim))
    ys = np.eye(output_dim)[labels].repeat(nb, axis=0)
    seq = np.linspace(latent_mean[latent_dim] - latent_std[latent_dim]*std_units,
                      latent_mean[latent_dim] + latent_std[latent_dim]*std_units,
                      nb)
    z = np.zeros((nb, L))
    z[:, latent_dim] = seq
    z = z.repeat(len(labels), axis=0)
    z = z.reshape((nb, len(labels), L))
    z = z.transpose((1, 0, 2))
    z = z.reshape((nb*len(labels), L))
    z = z.astype(np.float32)
    ys = ys.astype(np.float32)

    c = capsule.decode(z, ys)
    c = c.reshape((c.shape[0], w, h))
    plt.clf()
    grid_plot(c, imshow_options=dict(cmap="gray"), nbrows=len(labels), nbcols=nb)
    plt.savefig("hidfactor{0}.png".format(k))
    plt.show()
    k += 1
"""
light.endings() # save the duration
light.store_experiment() # update the DB
light.close() # close
