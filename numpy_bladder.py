"""
Example execution script. The dataset parameter can
be modified to bladder
"""
import argparse
import os
import sys
#os.environ['THEANO_FLAGS'] = 'device=gpu, optimizer=fast_compile,optimizer=None,force_device=True, exception_verbosity=high,allow_gc=False'
os.environ['THEANO_FLAGS'] = 'device=gpu0, optimizer=fast_run,force_device=True, exception_verbosity=high,allow_gc=False'
os.environ['debug_mode'] = 'True'

from Core.train import train

CopyRoot  = os.path.join('..','..','..')
projroot = os.path.join('..')
#dataroot = os.path.join(projroot, 'Data')

home = os.path.join(CopyRoot,'..')
dataroot = os.path.join(home,'DataSet/Bladder_Caption/Augmented/')
data_path = os.path.join(dataroot, 'Feat_conv')

dataset =  "bladder"
modelfolder = os.path.join('..','Data','Model','bladder')
saveto = os.path.join(modelfolder, "my_caption_model.npz")
if not os.path.exists(modelfolder):
        os.makedirs(modelfolder)  

parser = argparse.ArgumentParser()
#parser.add_argument("--attn_type",  default="deterministic",
#                    help="type of attention mechanism")
parser.add_argument("changes",  nargs="*",
                    help="Changes to default values", default="")


def main(params):
    # see documentation in capgen.py for more details on hyperparams
    _, validerr, _ =  train(**params) 
    print "Final cost: {:.2f}".format(validerr.mean())

if __name__ == "__main__":
    
    defaults = {"saveto": saveto,
                "attn_type": "dynamic" ,#"dynamic",
                #"attn_type": "deterministic" ,#"dynamic",
                "addressing": "ntm",
                "dim_word": 128,
                "ctx_dim": 512,
                "proj_ctx_dim": 512,
                "dim": 128,
                "shift_range":3,
                "n_layers_att": 2,
                "n_layers_out": 1,
                "n_layers_lstm": 1,
                "n_layers_init": 2,
                "n_words": 100,
                "lstm_encoder": False,
                "decay_c": 0.,
                "alpha_c": 1.,
                "prev2out": True,
                "ctx2out": True,
                "lrate": 0.0002 ,#0.01,
                "optimizer": "adam", #RMSprop
                "selector": True,
                "use_dropout": 0.5,
                "lstm_dropout": 0.25,
                "save_per_epoch": False,
                "reload": False,
                "valid_batch_size":2,
                "patience":10,
                "maxlen":100,
                "batch_size":64,
                "validFreq":100,
                "dispFreq":100,
                "saveFreq":100,
                "sampleFreq":100,
                "dataset": dataset,
                "data_path" : data_path
                 } 
    # get updates from command line
    args = parser.parse_args()
    for change in args.changes:
        defaults.update(eval("dict({})".format(change)))
    main(defaults)


