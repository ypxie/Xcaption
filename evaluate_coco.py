"""
Example execution script. The dataset parameter can
be modified to coco/flickr30k/flickr8k
"""
import argparse
import os
import sys
os.environ['THEANO_FLAGS'] = 'device=gpu,optimizer=None,force_device=True, exception_verbosity=high,allow_gc=False'
#os.environ['THEANO_FLAGS'] = 'device=gpu, optimizer=fast_run,force_device=True, allow_gc=True, exception_verbosity=high'
os.environ['debug_mode'] =  'False'
os.environ['homogeneous_data'] = 'False'

CopyRoot  = os.path.join('..','..','..')
projroot = os.path.join('..')
#dataroot = os.path.join(CopyRoot,'WorkStation','MIA_stru', 'Data')
dataroot = os.path.join(projroot, 'Data')

#kerasversion = 'keras-1'
##kerasversion = 'keras_classical'
#sys.path.insert(0, os.path.join(CopyRoot, 'Code', kerasversion))
#sys.path.insert(0, os.path.join(CopyRoot, 'Code', kerasversion,'keras'))
#sys.path.insert(0, os.path.join(CopyRoot, 'Code', kerasversion,'keras','layers'))
#sys.path.insert(0, '.')


from Core.train import train

parser = argparse.ArgumentParser()
#parser.add_argument("--attn_type",  default="deterministic",
#                    help="type of attention mechanism")


parser.add_argument("changes",  nargs="*",
                    help="Changes to default values", default="")


def main(params):
    # see documentation in capgen.py for more details on hyperparams
    _, validerr, _ = train(**params) 
#    saveto=params["saveto"],
#                           attn_type=params["attn-type"],
#                           reload_=params["reload"],
#                           dim_word=params["dim-word"],
#                           ctx_dim=params["ctx-dim"],
#                           dim=params["dim"],
#                           shift_range = params['shift_range'],
#                           n_layers_att=params["n-layers-att"],
#                           n_layers_out=params["n-layers-out"],
#                           n_layers_lstm=params["n-layers-lstm"],
#                           n_layers_init=params["n-layers-init"],
#                           n_words=params["n-words"],
#                           lstm_encoder=params["lstm-encoder"],
#                           decay_c=params["decay-c"],
#                           alpha_c=params["alpha-c"],
#                           prev2out=params["prev2out"],
#                           ctx2out=params["ctx2out"],
#                           lrate=params["learning-rate"],
#                           optimizer=params["optimizer"],
#                           selector=params["selector"],
#                           patience=10,
#                           maxlen=100,
#                           batch_size=64,
#                           valid_batch_size=params["valid_batch_size"],
#                           validFreq=2000,
#                           dispFreq=1,
#                           saveFreq=1000,
#                           sampleFreq=5,
#                           dataset="flickr30k",
#                           data_path = os.path.join('..','Data','TrainingData','flickr30k'),
#                           use_dropout=params["use-dropout"],
#                           use_dropout_lstm=params["use-dropout-lstm"],
#                           save_per_epoch=params["save-per-epoch"])
                           
    print "Final cost: {:.2f}".format(validerr.mean())

dataset = "flickr30k"
if __name__ == "__main__":
    # These defaults should more or less reproduce the soft
    # alignment model for the MS COCO dataset

    modelfolder = os.path.join('..','Data','Model','flickr30k')
    if not os.path.exists(modelfolder):
        os.makedirs(modelfolder) 
    data_path = os.path.join(dataroot,'TrainingData','flickr30k')
    saveto = os.path.join(modelfolder, "my_caption_model.npz")
    
    defaults = {#"saveto": saveto,
#                "attn_type": "dynamic" ,#"dynamic",
#                #"attn_type": "deterministic" ,#"dynamic",
#                "addressing": "ntm",
#                "dim_word": 512,
#                "ctx_dim": 512,
#                "proj_ctx_dim": 512,
#                "dim": 1800,
#                "shift_range":3,
#                "n_layers_att": 2,
#                "n_layers_out": 1,
#                "n_layers_lstm": 1,
#                "n_layers_init": 2,
#                "n_words": 10000,
#                "lstm_encoder": False,
#                "decay_c": 0.,
#                "alpha_c": 1.,
#                "prev2out": True,
#                "ctx2out": True,
#                "lrate": 0.01,
#                "optimizer": "adam", #RMSprop
#                "selector": True,
#                "use_dropout": 0.5,
#                "lstm_dropout": 0,
#                "save_per_epoch": False,
#                "reload": False,
#                "valid_batch_size":2,
#                "patience":10,
#                "maxlen":100,
#                "batch_size":64,
#                "validFreq":100,
#                "dispFreq":100,
#                "saveFreq":100,
#                "sampleFreq":100,
#                "dataset": "flickr30k",
#                "data_path" : data_path,
#                'print_training': True ,
#                'print_validation': False,
#                 'clipnorm': 0,
#                  'clipvalue':0
               "saveto": saveto,
                "attn_type": "dynamic" ,#"dynamic",
                #"attn_type": "dynamic" ,#"dynamic",
                "addressing": "softmax",
                #"attn_type": "stochastic",
                "dim_word": 64,
                "dim": 128,
                "ctx_dim": 512,
                'atten_num' : 196,
                "project_context": False,
                "proj_ctx_dim": 512,      
                "shift_range":3,
                "n_layers_att": 1,
                "n_layers_out": 1,
                "n_layers_lstm": 1,
                "n_layers_init": 1,
                "n_words": 50,
                "lstm_encoder": False,
                "decay_c": 1e-8,
                "alpha_c": 0.05,
                "prev2out": True,
                "ctx2out": True,
                "lrate": 0.01,
                "optimizer": "adam", #RMSprop
                "selector": True,
                "use_dropout": 0,
                "lstm_dropout": 0,
                "save_per_epoch": False,
                "reload": False, 
                "valid_batch_size":1,
                "patience":400,
                "maxlen":400,
                "batch_size": 1,
                "validFreq":100,
                "dispFreq":100,
                "saveFreq":100,
                "sampleFreq":100,
                "dataset": dataset,
                "data_path" : data_path,
                'online_feature':  False,
                'hard_sampling' : True,
                'print_training': False ,
                'print_validation': True,
                "clipnorm":0.1,
	            "clipvalue":0 
                 } 
    # get updates from command line
    args = parser.parse_args()
    for change in args.changes:
        defaults.update(eval("dict({})".format(change)))
    main(defaults)
