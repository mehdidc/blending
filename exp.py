import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils import load_textures, load_images
from lasagne.misc.plot_weights import grid_plot

from lightexperiments.light import Light
from lasagne.easy import get_stat
from hp_toolkit.hp import find_best_hp, minimize_fn_with_hyperopt
from aa import AA
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

import os

def launch():
    seed = 1234
    np.random.seed(seed)
    light = Light()
    light.initials()
    light.file_snapshot()
    light.set_seed(seed)

    w, h = 28, 28
    num_units = 1000
    fast_test = False
    test_ratio = 0.25
    valid_ratio = 0.25

    light.set("w", w)
    light.set("h", h)
    light.set("num_units", num_units)
    light.set("test_ratio", test_ratio)
    light.set("valid_ratio", valid_ratio)

    images = load_images()
    X = images

    # prepare
    X = shuffle(X)
    if fast_test is True:
        max_evaluations_hp = 1
        default_params = dict(
               max_epochs=2
        )
        X = X[0:100]
    else:
        default_params = dict()
        max_evaluations_hp = 20

    eval_function = lambda model, X_v, _: float(model.get_reconstruction_error(X_v))
    X_train_full, X_test = train_test_split(X, test_size=test_ratio)
    X_train, X_valid = train_test_split(X_train_full, test_size=valid_ratio)

    # show original data
    #X_ =  X.reshape((X.shape[0], im[0], im[1]))
    #X_ = X_[0:10]
    #grid_plot(X_, imshow_options={"cmap": "gray"})
    #plt.savefig(dirname+"/orig.png")
    #plt.show()

    best_hp, best_score = find_best_hp(
        AA,
        minimize_fn_with_hyperopt,
        X_train,
        X_valid,
        None,
        None,
        max_evaluations=max_evaluations_hp,
        default_params=default_params,
        eval_function=eval_function
    )
    best_hp.update(default_params)
    aa = AA(**best_hp)
    aa.fit(X_train_full, X_test)
    best_model = aa

    light.set("best_hp", best_hp)
    light.set("best_score", best_score)
    #light.set("best_model", light.insert_blob(best_model))
    names = best_model.capsule.batch_optimizer.stats[0].keys()
    stats = dict()
    for name in names:
        stats[name] =  get_stat(name, best_model.capsule.batch_optimizer.stats)
    light.set("best_model_stats", stats)
    light.set("nb_layers", aa.nb_layers * 2 - 1)

def write_report(folder, e):
    html = []

    html.append("<html><body>")
    names = set([ name[0:len(name)-5]
                for name in e.get("best_model_stats").keys()
                if name.endswith("train") or name.endswith("valid")])

    if not os.path.exists("{0}/img".format(folder)):
        os.mkdir("{0}/img".format(folder))

    for name in names:
        html.append("<h1>{0}</h1>".format(name))

        epochs = e.get("best_model_stats")["epoch"]
        plt.clf()
        if name + "train" in e.get("best_model_stats"):
            stats = e.get("best_model_stats")[name + "train"]
            plt.plot(epochs, stats, label="train")

        if name + "valid" in e.get("best_model_stats"):
            stats = e.get("best_model_stats")[name + "valid"]
            plt.plot(epochs, stats, label="test")
        plt.xlabel("epochs")
        plt.ylabel(name[0:-1])
        plt.legend()
        plt.savefig("{0}/img/{1}.png".format(folder, name))
        html.append("<img src='img/{0}.png'></img>".format(name))

    for i in range(1, e.get("nb_layers") + 1):
        html.append("<h1>Layer {0}</h1>".format(i))
        means = e.get("best_model_stats").get("activations_{0}_mean".format(i))
        stds = e.get("best_model_stats").get("activations_{0}_std".format(i))
        plt.clf()
        plt.errorbar(epochs, means, yerr=stds, label="layer {0}".format(i))
        plt.savefig("{0}/img/layer{1}.png".format(folder, i))
        html.append("<img src='img/layer{0}.png'></img>".format(i))

    html.append("<h1>Story</h1>")

    html.append("</body></html>")
    text = "\n".join(html)
    fd = open(folder + "/index.html", "w")
    fd.write(text)
    fd.close()

if __name__ == "__main__":
    light = Light()
    light.tag("aa_experiment")
    launch()
    write_report("report", light.cur_experiment)
    light.store_experiment()
    light.close()
