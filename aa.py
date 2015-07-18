from hp_toolkit.hp import Param, Model
import utils

from lasagne.easy import BatchOptimizer, LightweightModel
from lasagne import layers, updates, nonlinearities
from lasagne.layers.helper import get_all_layers
import theano.tensor as T
from theano.sandbox import rng_mrg
from lasagne.generative.capsule import Capsule
import theano

def corrupted_masking_noise(rng, x, corruption_level):
    return rng.binomial(size=x.shape, n=1, p=1 - corruption_level) * x

def corrupted_salt_and_pepper(rng, x, corruption_level):
    selected = rng.binomial(size=x.shape, n=1, p=corruption_level, dtype=theano.config.floatX)
    return x * (1 - selected) + selected * rng.binomial(size=x.shape, n=1, p=0.5, dtype=theano.config.floatX)

class AA(Model):

    params = dict(
        nb_units=Param(initial=100, interval=[100, 800], type='int'),
        corruption=Param(initial=0, interval=[0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], type='choice'),
        nb_layers=Param(initial=3, interval=[1, 2, 3, 4], type='choice'),
        corruption_type=Param(initial='salt_and_pepper', interval=['masking_noise', 'salt_and_pepper'], type='choice')
    )
    params.update(utils.params_batch_optimizer)

    def fit(self, X, X_valid=None):
        m = 1

        # define the model
        x_in = layers.InputLayer((None, X.shape[1]))
        hid = x_in

        for i in range(self.nb_layers):
            hid = layers.DenseLayer(hid, num_units=self.nb_units*m/(2**i),
                                    nonlinearity=nonlinearities.sigmoid)
        for i in range(self.nb_layers - 1):
            k = self.nb_layers - 2 - i
            hid = layers.DenseLayer(hid, num_units=self.nb_units*m/(2**k),
                                    nonlinearity=nonlinearities.sigmoid)
        o = layers.DenseLayer(hid, num_units=X.shape[1],
                            nonlinearity=nonlinearities.sigmoid)
        model = LightweightModel([x_in], [o])

        all_layers = get_all_layers(o)
        self.all_layers = all_layers
        rng = rng_mrg.MRG_RandomStreams()

        def get_reconstruction_error(model, X, x_hat=None):
            if x_hat is None:
                x_hat, = model.get_output(X)

            return (-(X * T.log(x_hat) +
                    (1 - X) * T.log(1 - x_hat)).sum(axis=1).mean())

        def loss_function(model, tensors):
            X = tensors["X"]
            X_noisy = X
            #X_noisy = X * (rng.uniform(X.shape) < (1 - self.corruption))
            #if self.corruption_type == "masking_noise":
            #    X_noisy = corrupted_masking_noise(rng, X, self.corruption)
            #elif self.corruption_type == "salt_and_pepper":
            #    X_noisy = corrupted_salt_and_pepper(rng, X, self.corruption)
            x_hat, = model.get_output(X_noisy)
    #       l1 = 0.01 * sum( T.abs_(layer.W).sum() for layer in all_layers[1:-1])
            l1 = 0
            diversity = 0
            return get_reconstruction_error(model, X, x_hat) + diversity + l1

        input_variables = dict(
            X=dict(tensor_type=T.matrix),
        )

        functions = dict(
            predict=dict(
                get_output=lambda model, X:model.get_output(X)[0],
                params=["X"]
            ),
            get_reconstruction_error=dict(
                get_output=get_reconstruction_error,
                params=["X"]
            )
        )

        for i, layer in enumerate(all_layers[1:-1]):
            functions["get_layer_{0}".format(i + 1)] = dict(get_output=lambda model, X: model.get_output(X)[0],
                                                            params=["X"])

        class MyBatchOptimizer(BatchOptimizer):

            def iter_update(self, epoch, nb_batches, iter_update_batch):
                status = super(MyBatchOptimizer, self).iter_update(epoch, nb_batches, iter_update_batch)
                status["reconstruction_error_train"] = capsule.get_reconstruction_error(X[0:100])
                if X_valid is not None:
                    status["reconstruction_error_valid"] = capsule.get_reconstruction_error(X_valid[0:100])
                return status
                for i, layer in enumerate(all_layers[1:-1]):
                    getter = getattr(capsule, "get_layer_{0}".format(i + 1))
                    activations = getter(X)
                    status["activations_{0}_mean".format(i + 1)] = activations.mean()
                    status["activations_{0}_std".format(i + 1)] = activations.std()

                return status

        batch_optimizer = MyBatchOptimizer(
            verbose=1,
            max_nb_epochs=self.max_epochs,
            batch_size=self.batch_size,
            optimization_procedure=(
                   updates.adagrad,
                   {"learning_rate": self.learning_rate}
            ),
            whole_dataset_in_device=True
        )
        capsule = Capsule(
            input_variables, model,
            loss_function,
            functions=functions,
            batch_optimizer=batch_optimizer,
        )

        capsule.fit(X=X)
        self.capsule = capsule

    def get_reconstruction_error(self, X):
        return self.capsule.get_reconstruction_error(X)
