import argparse
import os
import sys
#os.environ['THEANO_FLAGS'] = 'device=gpu, optimizer=fast_compile,optimizer=None,force_device=True,allow_gc=False'
os.environ['THEANO_FLAGS'] = 'device=gpu0, optimizer=fast_run,force_device=False, allow_gc=True'
os.environ['debug_mode'] = 'False'
os.environ['homogeneous_data'] = 'False'


CopyRoot  = os.path.join('..','..','..','..')
projroot = os.path.join('..','..')
#dataroot = os.path.join(projroot, 'Data')
home = os.path.join(CopyRoot,'..')
dataroot = os.path.join(home,'DataSet/Bladder_Caption/Augmented/')
data_path = os.path.join(dataroot)
sys.path.insert(0, '..')

from Core.train import train

dataset =  "bladder"
#modelfolder = os.path.join('..','Data','Model','bladder','dynamic_softmax')
modelfolder = os.path.join(projroot,'Data','Model','bladder','labtory')
#modelfolder = os.path.join(projroot,'Data','Model','bladder','deep_ntm_e2e')

saveto = os.path.join(modelfolder, "my_caption_model.npz")
if not os.path.exists(modelfolder):
    os.makedirs(modelfolder)  

keras_model_path = os.path.join(CopyRoot,'WorkStation','fullyConv','Data','Model','Bladder','deep_conclusion_bladder')

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
                "ctx_dim_list": [512, 256],
                "proj_ctx_dim": 256,
                'project_context': True,
                "dim": 256,
                "shift_range":3,
                "n_layers_att": 1,
                "n_layers_out": 1,
                "n_layers_lstm": 1,
                "n_layers_init": 1,
                "n_words": 50,
                "lstm_encoder": False,
                "decay_c": 1e-8,
                "alpha_c": 0.001,
                "prev2out": True,
                "ctx2out": True,
                "lrate": 0.01,
                "optimizer": "adam", #RMSprop
                "use_dropout": 0.25,
                "lstm_dropout": 0.25,
                "save_per_epoch": False,
                "reload": False,
                "valid_batch_size":2,
                "patience":400,
                "maxlen":400,
                "batch_size": 16,
                "validFreq":100,
                "dispFreq":100,
                "saveFreq":100,
                "sampleFreq":100,
                "dataset": dataset,
                "data_path" : data_path,
                'keras_model_path': keras_model_path,
                'train_feat_encoder': True,
                'print_training': False ,
                'print_validation': True,
                'online_feature':  True,
                'hard_sampling' : True,
                "clipnorm":0,
	            "clipvalue":10
                 } 
    # get updates from command line
    args = parser.parse_args()
    for change in args.changes:
        defaults.update(eval("dict({})".format(change)))
    main(defaults)


