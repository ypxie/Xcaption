"""
Example execution script. The dataset parameter can
be modified to bladder
"""
from Core import *
import os
import argparse

from Core.train import train

parser = argparse.ArgumentParser()
parser.add_argument("--attn_type",  default="deterministic",
                    help="type of attention mechanism")


parser.add_argument("changes",  nargs="*",
                    help="Changes to default values", default="")


def main(params):
    # see documentation in capgen.py for more details on hyperparams
    _, validerr, _ = train(saveto=params["model"],
                           attn_type=params["attn-type"],
                           reload_=params["reload"],
                           dim_word=params["dim-word"],
                           ctx_dim=params["ctx-dim"],
                           dim=params["dim"],
                           n_layers_att=params["n-layers-att"],
                           n_layers_out=params["n-layers-out"],
                           n_layers_lstm=params["n-layers-lstm"],
                           n_layers_init=params["n-layers-init"],
                           n_words=params["n-words"],
                           lstm_encoder=params["lstm-encoder"],
                           decay_c=params["decay-c"],
                           alpha_c=params["alpha-c"],
                           prev2out=params["prev2out"],
                           ctx2out=params["ctx2out"],
                           lrate=params["learning-rate"],
                           optimizer=params["optimizer"],
                           selector=params["selector"],
                           patience=10,
                           maxlen=100,
                           batch_size=64,
                           valid_batch_size=params["valid_batch_size"],
                           validFreq=2000,
                           dispFreq=100,
                           saveFreq=1000,
                           sampleFreq=50,
                           dataset="bladder",
                           data_path = os.path.join('..','Data','TrainingData','flickr30k'),
                           use_dropout=params["use-dropout"],
                           use_dropout_lstm=params["use-dropout-lstm"],
                           save_per_epoch=params["save-per-epoch"])
    print "Final cost: {:.2f}".format(validerr.mean())


if __name__ == "__main__":
    # These defaults should more or less reproduce the soft
    # alignment model for the MS COCO dataset
    defaults = {"model": "my_caption_model.npz",
                "attn-type": "dynamic",
                "dim-word": 512,
                "ctx-dim": 512,
                "dim": 1800,
                "shift_range":3,
                "mem_dim": 512,
                "n-layers-att": 2,
                "n-layers-out": 1,
                "n-layers-lstm": 1,
                "n-layers-init": 2,
                "n-words": 50,
                "lstm-encoder": False,
                "decay-c": 0.,
                "alpha-c": 1.,
                "prev2out": True,
                "ctx2out": True,
                "learning-rate": 0.01,
                "optimizer": "adam",
                "selector": True,
                "use-dropout": True,
                "use-dropout-lstm": False,
                "save-per-epoch": True,
                "reload": True,
                "valid_batch_size":2}
    # get updates from command line
    args = parser.parse_args()
    #for k,v in args:
    #    defaults[k] = v
    for change in args.changes:
        defaults.update(eval("dict({})".format(change)))
    main(defaults)
