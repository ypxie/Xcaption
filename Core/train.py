'''
Source code for an attention based image caption generation system described
in:

Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
International Conference for Machine Learning (2015)
http://arxiv.org/abs/1502.03044

Comments in square brackets [] indicate references to the equations/
more detailed explanations in the above paper.
'''
#import theano
#import theano.tensor as tensor
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import backend.export as T

from fuel import datasets
from Core.engine import *
#from Core.original_optimizer import adadelta, adam, rmsprop, sgd
from Core.optimizers import *

if not 'homogeneous_data' in os.environ:
    os.environ['homogeneous_data'] == 'True'
if os.environ['homogeneous_data'] == 'True':
    from fuel.homogeneous_data import HomogeneousData
else:
    from fuel.inhomogeneous_data import HomogeneousData

from backend.export import npwrapper

import pickle as pkl
import numpy as np
import copy
import os
import time

from sklearn.cross_validation import KFold

import random
import warnings
# [see Section (4.3) for explanation]

# supported optimizers


# dataset iterators
def get_dataset(name):
    return datasets[name][0], datasets[name][1]

def validate_options(options):
    # Put friendly reminders here
    if options['dim_word'] > options['dim']:
        warnings.warn('dim_word should only be as large as dim.')

    if options['lstm_encoder']:
        warnings.warn('Note that this is a 1-D bidirectional LSTM, not 2-D one.')

    if options['lstm_dropout']:
        warnings.warn('dropout in the lstm seems not to help')

    # Other checks:
    if options['attn_type'] not in ['stochastic', 'deterministic', 'dynamic']:
        raise ValueError("specified attention type is not correct")

    return options

def pred_probs(f_log_probs, options, worddict, prepare_data, data, iterator, verbose=False, train=0,
               generate_sample_params = None):
    """ Get log probabilities of captions
    Parameters
    ----------
    f_log_probs : theano function
        compute the log probability of a x given the context
    options : dict
        options dictionary
    worddict : dict
        maps words to one-hot encodings
    prepare_data : function
        see corresponding dataset class for details
    data : np array
        output of load_data, see corresponding dataset class
    iterator : KFold
        indices from scikit-learn KFold
    verbose : boolean
        if True print progress
    Returns
    -------
    probs : np array
        array of log probabilities indexed by example
    """
    n_samples = len(data[0])
    probs = np.zeros((n_samples, 1)).astype('float32')

    n_done = 0

    max_sample = 20
    sample_size = 1
    sample_count = 0
    if generate_sample_params is not None and options['print_validation'] == True:
        print('Generating validation samples!')
    for _, valid_index in iterator:
        x, mask, ctx = prepare_data([data[0][t] for t in valid_index],
                                     data[1],
                                     worddict,
                                     maxlen=None,
                                     n_words=options['n_words'],
                                     online_feature=options['online_feature'])
        pred_probs = f_log_probs(x,mask,ctx,np.uint8(train))

        probs[valid_index] = pred_probs[:,None]

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples computed'%(n_done,n_samples)
        if generate_sample_params is not None and options['print_validation'] == True:
            if sample_count != max_sample:
                print_generated_sample(generate_sample_params, ctx, truth = x, 
                                       minSample = sample_size, mask = mask)
                sample_count += min(sample_size, len(ctx))

    return probs

def print_generated_sample(generate_sample_params, ctx_s, truth = None, minSample = 10, mask=None):
    '''
    ctx_s: nsample*slot*dim
    truth: time_step * nsample * 1
    '''
    tparams = generate_sample_params.tparams 
    gen_sample = generate_sample_params.gen_sample 
    f_init = generate_sample_params.f_init
    f_next = generate_sample_params.f_next
    model_options = generate_sample_params.model_options
    word_idict = generate_sample_params.word_idict
    rng = generate_sample_params.rng
    stochastic = generate_sample_params.stochastic
    beam_k = generate_sample_params.beam_k
    maxlen = generate_sample_params.maxlen
    
 
    minSample = len(ctx_s) if minSample is None else minSample
    # generate and decode a subset of the current training batch
    sample_rand_Ind = range(len(ctx_s))
    random.shuffle(sample_rand_Ind)
    acutal_sample = np.minimum(minSample, len(ctx_s) )
    
    for jj in sample_rand_Ind[0:acutal_sample]:
        sample, score = gen_sample(tparams, f_init, f_next, ctx_s[jj], model_options,
                                rng=rng, k=beam_k, maxlen=maxlen, stochastic=stochastic)
        # Decode the sample from encoding back to words
        if truth is not None:
            if truth.shape[1] != len(ctx_s):
                raise Exception('x_s must has the same sample size as ctx_s!')
            print 'Truth ',jj,': ',
            # if len(truth[:,jj]) == 0:
            #     print('sample id: {s}\n'.format(s= str(jj)))
            #     print ctx_s[jj]
            #     raise Exception('truth is empty please check if sth is wrong!!')
            # print truth[:,jj]
            # if mask is not None:
            #     print mask[:,jj] 
            for vv in truth[:,jj]:  
                if vv == 0:
                    print '<eos>'
                    break
                    #pass
                if vv in word_idict:
                    print word_idict[vv],
                else:
                    print 'UNK',
        print    
        for kk, ss in enumerate([sample[0]]):
            print 'Sample (', kk,') ', jj, ': ',
            for vv in ss:
                if vv == 0:
                    #pass
                    print '<eos>'
		    break
                if vv in word_idict:
                    print word_idict[vv],
                else:
                    print 'UNK',
        print('\n')

def BUILD_MODEL(tparams, params, model_options):
    # build_model_single
    if os.environ['debug_mode'] == 'True':
        # start of debuging
        #from Core.train import  get_dataset
        #from fuel.homogeneous_data import HomogeneousData
        batch_size, maxlen = model_options['batch_size'], 400
        load_data, prepare_data = get_dataset(model_options['dataset'])
        train, valid, test, worddict = load_data(root_path = model_options['data_path'], online_feature=model_options['online_feature'])
        train_iter = HomogeneousData(train, batch_size=batch_size, maxlen=maxlen)
        
        for caps in train_iter:
            x, mask, featSource = prepare_data(caps,
                                            train[1],
                                            worddict,
                                            maxlen=maxlen,
                                            online_feature=model_options['online_feature'],
                                            n_words=model_options['n_words'])
            x = npwrapper(x)
            mask = npwrapper(mask)
            featSource = npwrapper(featSource)

            # the following is for sampling
            ctx_2d = T.npwrapper(featSource[0]) 
            x_1d = T.npwrapper(x[0,0:1])
            mask_1d = T.npwrapper(mask[0,0:1]) 


            break
    else:
        # description string: #words x #samples,
        if model_options['online_feature'] == True:
            featSource  = T.placeholder(shape=(None, 3, None, None), name='ctx')       
        else:
            featSource  = T.placeholder(shape=(None, None, model_options['ctx_dim']), name='ctx')
        x = T.placeholder(ndim=2, name='x', dtype='int64')
        mask = T.placeholder(ndim=2, name='mask')
        # context: #samples x #annotations x dim
        
        
        # the following is for sampler
        ctx_2d = T.placeholder(shape=(None,model_options['ctx_dim']), name= 'ctx_sampler')
        x_1d = T.placeholder(ndim=1, name = 'x_sampler', dtype='int64')
        mask_1d = T.placeholder(ndim=1, name = 'mask_sampler', dtype='int64')

        init_state = [T.placeholder(ndim=2, name='init_state')]
        init_memory = [T.placeholder(ndim=2, name='init_memory')]
        init_alpha = [T.placeholder(ndim=2, name='init_alpha')]

        if model_options['n_layers_lstm'] > 1:
            for lidx in xrange(1, options['n_layers_lstm']):
                init_state.append(T.placeholder(ndim=2, name='init_state_' + str(lidx)))
                init_memory.append(T.placeholder(ndim=2, name='init_memory_' + str(lidx))) 
                init_alpha.append(T.placeholder(ndim=2, name='init_alpha_' + str(lidx))) 

    import time
    time_start = time.time()          
    rng,use_noise,inps, alphas, alphas_sample,\
          cost, \
          opt_outs = \
          build_model_single(tparams, model_options, x, mask, featSource, params,
          prefix = 'atten_model')

    # Initialize (or reload) the parameters using 'model_options'
    # then build the Theano graph
    trainable_param = OrderedDict()
    for k, v in tparams.iteritems():
        if getattr(v, 'trainable', True):
            trainable_param[k] = v

    # To sample, we use beam search: 1) f_init is a function that initializes
    # the LSTM at time 0 [see top right of page 4], 2) f_next returns the distribution over
    # words and also the new "initial state/memory" see equation
    print 'Buliding sampler'

    (init_outputs, f_init_clousure), (next_outputs, f_next_clousure) = \
    build_sampler(tparams, model_options, rng, x_1d, ctx_2d,  sampling=True, dropoutrate = 0.5,allow_input_downcast=True)


    # init_outputs , f_init_clousure= build_init(tparams, model_options,  rng, ctx_2d)
    # init_values = init_outputs[1:]
    # ctx_encoded = init_values[0]
    # next_outputs, f_next_clousure= build_next(tparams, model_options, rng, x_1d, ctx_2d, ctx_encoded,  init_values = init_values)
    
    print('Start building sampler function!')
    if os.environ['debug_mode'] == 'False':
        f_init = T.function([ ctx_2d, T.learning_phase()], init_outputs,  allow_input_downcast=True,on_unused_input='ignore')
        ctx_encoded = init_outputs[0]
        init_values = init_outputs[1:]
        f_next = T.function([x_1d, ctx_encoded, T.learning_phase()] + init_values, next_outputs, allow_input_downcast=True,on_unused_input='ignore')

    else:
        f_init =  f_init_clousure
        f_next =  f_next_clousure
    print('Finished building sampler function!')
    #f_init, f_next = build_sampler(tparams, model_options, x_1d, ctx_2d, rng, allow_input_downcast=True)

    # we want the cost without any the regularizers
    inps += [T.learning_phase()] 
    if model_options['attn_type']=='stochastic' or model_options['hard_sampling']==True:
        updates=opt_outs['attn_updates']
    else:
        updates = None
    print updates
    f_log_probs = T.function(inps, -cost, profile=False, 
                                        updates= updates,
                                        allow_input_downcast=True)
    cost = cost.mean()

    decay_c = model_options['decay_c']
    alpha_c = model_options['alpha_c']
    alpha_entropy_c = model_options['alpha_entropy_c']
    optimizer = model_options['optimizer']
    lrate = model_options['lrate']
    # add L2 regularization costs
    if decay_c > 0.:
        decay_c = T.variable(np.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in trainable_param.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay
    
    # Doubly stochastic regularization
    if alpha_c > 0.:
        alpha_c = T.variable(np.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * ((1.-alphas.sum(0))**2).sum(0).mean()
        cost += alpha_reg

    hard_attn_updates = []
    # Backprop!
    if model_options['hard_sampling'] != True and model_options['attn_type'] != 'stochastic':
        wrt =itemlist(trainable_param)
        grads = T.grad(cost, wrt)
    else:
        # shared variables for hard attention
        baseline_time = T.variable(np.float32(0.), name='baseline_time')
        opt_outs['baseline_time'] = baseline_time
        alpha_entropy_c = T.variable(np.float32(alpha_entropy_c), name='alpha_entropy_c')
        alpha_entropy_reg = alpha_entropy_c * (alphas*T.log(alphas)).mean()
        # [see Section 4.1: Stochastic "Hard" Attention for derivation of this learning rule]
        if model_options['RL_sumCost']:
            grads = T.grad(cost, wrt=itemlist(trainable_param),
                                 disconnected_inputs='raise',
                                known_grads={alphas:(baseline_time-opt_outs['masked_cost'].mean(0))[None,:,None]/10.*
                                            (-alphas_sample/(alphas+0.00000001)) + alpha_entropy_c*(T.log(alphas+0.00000001) + 1)})
        else:
            grads = T.grad(cost, wrt=itemlist(trainable_param),
                            disconnected_inputs='raise',
                            known_grads={alphas:opt_outs['masked_cost'][:,:,None]/10.*
                            (alphas_sample/(alphas+0.00000001)) + alpha_entropy_c*(T.log(alphas+0.00000001) + 1)})
        # [equation on bottom left of page 5]
        hard_attn_updates += [(baseline_time, baseline_time * 0.9 + 0.1 * opt_outs['masked_cost'].mean())]
        # updates from scan
        hard_attn_updates += opt_outs['attn_updates']
        
        print hard_attn_updates
    # to getthe cost after regularization or the gradients, use this
    if not 'clipvalue' in model_options:
        model_options['clipvalue'] = 0
    if not 'clipnorm' in model_options:
        model_options['clipnorm'] = 0.5

    opt = eval(optimizer)(lr=lrate,clipvalue = model_options['clipvalue'],clipnorm = model_options['clipnorm'])
    training_updates = opt.get_updates(itemlist(trainable_param),grads)
    updates = hard_attn_updates + training_updates

    f_train = T.function(inps,cost, updates=updates,on_unused_input='warn',allow_input_downcast=True)
    
    #    if options['debug'] == 0:
    #        import theano
    #        theano.printing.pydotprint(f_train, outfile='aa.png',with_ids=True,return_image=False, format='png')
    #        from IPython.display import SVG
    #        SVG(theano.printing.pydotprint(f_train, return_image=True,
    #                                       format='svg'))
    #lr = T.scalar(name='lr')    
    #f_grad_shared, f_update = eval(optimizer)(lr, trainable_param, grads, inps, cost, hard_attn_updates)

    total_time = time.time() - time_start
    print "Building Model takes {}".format(total_time)
    return f_train, f_init, f_next, f_log_probs,[rng,use_noise, alphas, alphas_sample]

"""Note: all the hyperparameters are stored in a dictionary model_options (or options outside train).
   train() then proceeds to do the following:
       1. The params are initialized (or reloaded)
       2. The computations graph is built symbolically using Theano.
       3. A cost is defined, then gradient are obtained automatically with tensor.grad :D
       4. With some helper functions, gradient descent + periodic saving/printing proceeds
"""
def train(dim_word=100,  # word vector dimensionality
          ctx_dim=512,  # context vector dimensionality
          project_context = True,
          proj_ctx_dim = 512, # projected context vector dimensionality
          dim=1000,  # the number of LSTM units
          shift_range = 3, # how many shift for ntm memory location
          attn_type='stochastic',  # [see section 4 from paper]
          addressing= "softmax",  #addressing types
          k_activ = "tanh",# only used for ntm addressing for generating key
          atten_num = 196,  # number of attention positions
          n_layers_att=1,  # number of layers used to compute the attention weights
          n_layers_out=1,  # number of layers used to compute logit
          n_layers_lstm=1,  # number of lstm layers
          n_layers_init=1,  # number of layers to initialize LSTM at time 0
          lstm_encoder=False,  # if True, run bidirectional LSTM on input units
          prev2out=False,  # Feed previous word into logit
          ctx2out=False,  # Feed attention weighted ctx into logit
          alpha_entropy_c=0.002,  # hard attn param
          RL_sumCost=True,  # hard attn param
          semi_sampling_p=0.5,  # hard attn param
          temperature=1.,  # hard attn param
          patience=10,
          max_epochs=5000,
          dispFreq=100,
          decay_c=0.,  # weight decay coeff
          alpha_c=1.,  # doubly stochastic coeff
          lrate=0.01,  # used only for SGD
          selector=False,  # selector (see paper)
          n_words=10000,  # vocab size
          maxlen=100,   #maximum length of the description
          optimizer='rmsprop',
          batch_size = 16,
          valid_batch_size = 16,
          saveto='model.npz',  # relative path of saved model file
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq updates
          dataset= None,
          data_path = None,
          dictionary= None,  # word dictionary
          use_dropout= 0.5,  # setting this true turns on dropout at various points
          lstm_dropout= None,  # dropout on lstm gates
          reload=False,
          hard_sampling = True,
          online_feature = True,
          save_per_epoch=False, print_training = True,print_validation=True, 
          clipvalue=0, clipnorm=0,**kwargs): # this saves down the model every epoch

    # hyperparam dict
    model_options = locals().copy()
    options = model_options
    model_options = validate_options(model_options)

    # reload options
    #if reload and os.path.exists(saveto):
    #    print "Reloading options"
    #    with open('%s.pkl'%saveto, 'rb') as f:
    #        model_options = pkl.load(f)
    
    tparams =  OrderedDict()
    params = OrderedDict()
    #params = init_params(model_options)
    if reload and os.path.exists(saveto):
        print "Reloading model"
        params = load_params(saveto, params)
    #tparams = init_tparams(params)


    print "Using the following parameters:"
    print  model_options

    print 'Loading data'
    load_data, prepare_data = get_dataset(dataset)
    print data_path
    train, valid, test, worddict = load_data(root_path = data_path, online_feature=model_options['online_feature'])
    train_iter = HomogeneousData(train, batch_size=batch_size, maxlen=maxlen)
    
    # index 0 and 1 always code for the end of sentence and unknown token
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'
    
    print 'Building model'

    f_train, f_init, f_next,f_log_probs, [rng,use_noise, alphas, alphas_sample] = BUILD_MODEL(tparams, params, model_options)

    print 'Optimization'

    # [See note in section 4.3 of paper]
    if valid:
        kf_valid = KFold(len(valid[0]), n_folds=len(valid[0])/valid_batch_size, shuffle=False)
    if test:
        kf_test = KFold(len(test[0]), n_folds=len(test[0])/valid_batch_size, shuffle=False)

    # history_errs is a bare-bones training log that holds the validation and test error
    history_errs = []
    # reload history
    if reload and os.path.exists(saveto):
        history_errs = np.load(saveto)['history_errs'].tolist()
    best_p = None
    bad_counter = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size
    
    from Core.utils_func import obj as myobj
    generate_sample_params = myobj()
    generate_sample_params.tparams = tparams
    generate_sample_params.gen_sample = gen_sample
    generate_sample_params.f_init = f_init
    generate_sample_params.f_next = f_next
    generate_sample_params.model_options = model_options
    generate_sample_params.word_idict = word_idict
    generate_sample_params.rng = rng
    generate_sample_params.stochastic =False
    generate_sample_params.beam_k = 5
    generate_sample_params.maxlen = 100
    generate_sample_params.minSample  = 10
    
    uidx = 0
    estop = False
    for eidx in xrange(max_epochs):
        n_samples = 0

        print 'Epoch ', eidx

        for caps in train_iter:
            
            n_samples += len(caps)
            uidx += 1
            # turn on dropout
            use_noise.set_value(1.)

            # preprocess the caption, recording the time spent to help detect bottlenecks
            pd_start = time.time()
            x, mask, ctx = prepare_data(caps,
                                        train[1],
                                        worddict,
                                        maxlen=maxlen,
                                        n_words=n_words,
                                        online_feature=model_options['online_feature'])
            pd_duration = time.time() - pd_start
            
            #print x.shape, mask.shape, ctx.shape 
            
            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                continue

            # get the cost for the minibatch, and update the weights
            ud_start = time.time()
            
            #cost = f_grad_shared(x, mask, ctx, 1)
            #f_update(lrate)
            cost = f_train(x,mask,ctx, np.uint8(1.0))
            ud_duration = time.time() - ud_start # some monitoring for each mini-batch

            # Numerical stability check
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.
            if np.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'PD ', pd_duration, 'UD ', ud_duration
            # Checkpoint
            if np.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                if best_p is not None:
                    params = copy.copy(best_p)
                else:
                    params = unzip(tparams)
                np.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                print 'Done'

            # Print a generated sample as a sanity check
            if np.mod(uidx, sampleFreq) == 0:
                # turn off dropout first
                use_noise.set_value(0.)
                #K.set_learning_phase(0.)
                x_s = x
                mask_s = mask
                ctx_s = ctx                
                # generate and decode a subset of the current training batch
                if not 'print_training' in model_options:
                    model_options['print_training'] = False
                if model_options['print_training']:
                    print('Print a subset of the current training batch!!\n ')
                    print_generated_sample(generate_sample_params, ctx_s, truth = x_s, minSample = 10)

            # Log validation loss + checkpoint the model with the best validation log likelihood
            if np.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                train_err = 0
                valid_err = 0
                test_err = 0

                if valid:
                    valid_err = -pred_probs(f_log_probs, model_options, worddict, prepare_data, valid, 
                                            kf_valid, train=0, generate_sample_params = generate_sample_params).mean()
                if test:
                    test_err = -pred_probs(f_log_probs, model_options, worddict, prepare_data, test, 
                                           kf_test,train=0,generate_sample_params = generate_sample_params).mean()

                history_errs.append([valid_err, test_err])

                # the model with the best validation long likelihood is saved seperately with a different name
                if uidx == 0 or valid_err <= np.array(history_errs)[:,0].min():
                    best_p = unzip(tparams)
                    print 'Saving model with best validation ll'
                    params = copy.copy(best_p)
                    params = unzip(tparams)
                    np.savez(saveto+'_bestll', history_errs=history_errs, **params)
                    bad_counter = 0

                # abort training if perplexity has been increasing for too long
                if eidx > patience and len(history_errs) > patience and valid_err >= np.array(history_errs)[:-patience,0].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

        print 'Seen %d samples' % n_samples

        if estop:
            break

        if save_per_epoch:
            np.savez(saveto + '_epoch_' + str(eidx + 1), history_errs=history_errs, **unzip(tparams))

    # use the best nll parameters for final checkpoint (if they exist)
    if best_p is not None:
        zipp(best_p, tparams)

    #use_noise.set_value(0.)
    train_err = 0
    valid_err = 0
    test_err = 0
    if valid:
        valid_err = -pred_probs(f_log_probs, model_options, worddict, prepare_data, valid, kf_valid)
    if test:
        test_err = -pred_probs(f_log_probs, model_options, worddict, prepare_data, test, kf_test)

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

    params = copy.copy(best_p)
    np.savez(saveto, zipped_params=best_p, train_err=train_err,
                valid_err=valid_err, test_err=test_err, history_errs=history_errs,
                **params)

    return train_err, valid_err, test_err


if __name__ == '__main__':
    pass
