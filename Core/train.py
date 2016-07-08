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
from fuel.homogeneous_data import HomogeneousData
from fuel import datasets
from Core.engine import *

import pickle as pkl
import numpy as np
import copy
import os
import time

from sklearn.cross_validation import KFold

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

    if options['use_dropout_lstm']:
        warnings.warn('dropout in the lstm seems not to help')

    # Other checks:
    if options['attn_type'] not in ['stochastic', 'deterministic', 'dynamic']:
        raise ValueError("specified attention type is not correct")

    return options

def pred_probs(f_log_probs, options, worddict, prepare_data, data, iterator, verbose=False, train=0):
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

    for _, valid_index in iterator:
        x, mask, ctx = prepare_data([data[0][t] for t in valid_index],
                                     data[1],
                                     worddict,
                                     maxlen=None,
                                     n_words=options['n_words'])
        pred_probs = f_log_probs(x,mask,ctx,train)
        probs[valid_index] = pred_probs[:,None]

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples computed'%(n_done,n_samples)

    return probs

"""Note: all the hyperparameters are stored in a dictionary model_options (or options outside train).
   train() then proceeds to do the following:
       1. The params are initialized (or reloaded)
       2. The computations graph is built symbolically using Theano.
       3. A cost is defined, then gradient are obtained automatically with tensor.grad :D
       4. With some helper functions, gradient descent + periodic saving/printing proceeds
"""
def train(dim_word=100,  # word vector dimensionality
          ctx_dim=512,  # context vector dimensionality
          dim=1000,  # the number of LSTM units
          shift_range = 3, # how many shift for ntm memory location
          attn_type='stochastic',  # [see section 4 from paper]
          addressing= "softmax",  #addressing types
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
          alpha_c=0.,  # doubly stochastic coeff
          lrate=0.01,  # used only for SGD
          selector=False,  # selector (see paper)
          n_words=10000,  # vocab size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size = 16,
          valid_batch_size = 16,
          saveto='model.npz',  # relative path of saved model file
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq updates
          dataset='flickr8k',
          data_path = '../Data/TrainingData/flickr8k',
          dictionary= None,  # word dictionary
          use_dropout= 0.5,  # setting this true turns on dropout at various points
          lstm_dropout= None,  # dropout on lstm gates
          reload_=False,
          save_per_epoch=False, debug = True, **kwargs): # this saves down the model every epoch

    # hyperparam dict
    model_options = locals().copy()
    
    model_options = validate_options(model_options)

    # reload options
    if reload_ and os.path.exists(saveto):
        print "Reloading options"
        with open('%s.pkl'%saveto, 'rb') as f:
            model_options = pkl.load(f)

    print "Using the following parameters:"
    print  model_options

    print 'Loading data'
    load_data, prepare_data = get_dataset(dataset)
    print data_path
    train, valid, test, worddict = load_data(path = data_path)

    # index 0 and 1 always code for the end of sentence and unknown token
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # Initialize (or reload) the parameters using 'model_options'
    # then build the Theano graph
    print 'Building model'
    params = init_params(model_options)
    if reload_ and os.path.exists(saveto):
        print "Reloading model"
        params = load_params(saveto, params)
        #params = load_params('my_caption_model.npz_bestll.npz', params)

    # np arrays -> theano shared variables

    tparams = init_tparams(params)
    trainable_param = OrderedDict()
    for k, v in tparams.iteritems():
        if v.trainable:
           trainable_param[k] = v

    import time
    time_start = time.time()
    rng, use_noise, \
          inps, alphas, alphas_sample,\
          cost, \
          opt_outs = \
          build_model(tparams, model_options)
    

    # To sample, we use beam search: 1) f_init is a function that initializes
    # the LSTM at time 0 [see top right of page 4], 2) f_next returns the distribution over
    # words and also the new "initial state/memory" see equation
    print 'Buliding sampler'
    f_init, f_next = build_sampler(tparams, model_options, use_noise, rng)

    # we want the cost without any the regularizers
    inps += [T.learning_phase()] 
    f_log_probs = T.function(inps, -cost, profile=False,
                                        updates=opt_outs['attn_updates']
                                        if model_options['attn_type']=='stochastic'
                                        else None)
    
    
    cost = cost.mean()
    # add L2 regularization costs
    if decay_c > 0.:
        decay_c = T.shared(np.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in trainable_param.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay
    
    # Doubly stochastic regularization
    if alpha_c > 0.:
        alpha_c = T.shared(np.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * ((1.-alphas.sum(0))**2).sum(0).mean()
        cost += alpha_reg

    hard_attn_updates = []
    # Backprop!
    if model_options['attn_type'] == 'deterministic' or model_options['attn_type'] == 'dynamic':
        grads = T.grad(cost, wrt=itemlist(trainable_param))
    else:
        # shared variables for hard attention
        baseline_time = T.shared(np.float32(0.), name='baseline_time')
        opt_outs['baseline_time'] = baseline_time
        alpha_entropy_c = T.shared(np.float32(alpha_entropy_c), name='alpha_entropy_c')
        alpha_entropy_reg = alpha_entropy_c * (alphas*T.log(alphas)).mean()
        # [see Section 4.1: Stochastic "Hard" Attention for derivation of this learning rule]
        if model_options['RL_sumCost']:
            grads = T.grad(cost, wrt=itemlist(trainable_param),
                                 disconnected_inputs='raise',
                                known_grads={alphas:(baseline_time-opt_outs['masked_cost'].mean(0))[None,:,None]/10.*
                                            (-alphas_sample/alphas) + alpha_entropy_c*(T.log(alphas) + 1)})
        else:
            grads = T.grad(cost, wrt=itemlist(trainable_param),
                            disconnected_inputs='raise',
                            known_grads={alphas:opt_outs['masked_cost'][:,:,None]/10.*
                            (alphas_sample/alphas) + alpha_entropy_c*(T.log(alphas) + 1)})
        # [equation on bottom left of page 5]
        hard_attn_updates += [(baseline_time, baseline_time * 0.9 + 0.1 * opt_outs['masked_cost'].mean())]
        # updates from scan
        hard_attn_updates += opt_outs['attn_updates']

    # to getthe cost after regularization or the gradients, use this
    # f_cost = theano.function([x, mask, ctx], cost, profile=False)
    # f_grad = theano.function([x, mask, ctx], grads, profile=False)

    # f_grad_shared computes the cost and updates adaptive learning rate variables
    # f_update updates the weights of the model

    #lr = T.scalar(name='lr')
    #opt = optimizers.get(optimizer)
    opt = eval(optimizer)(lr=lrate)

    #opt = Adadelta(lr=lrate, rho=0.95, epsilon=1e-06)
    training_updates = opt.get_updates(trainable_param,grads)

    updates = hard_attn_updates + training_updates
    f_train = T.function(inps,cost, updates=updates)

    #f_grad_shared, f_update = eval(optimizer)(lr, trainable_param, grads, inps, cost, hard_attn_updates)

    total_time = time.time() - time_start
    print "Building Model takes {}".format(total_time)
    
    
    print 'Optimization'

    # [See note in section 4.3 of paper]
    train_iter = HomogeneousData(train, batch_size=batch_size, maxlen=maxlen)

    if valid:
        kf_valid = KFold(len(valid[0]), n_folds=len(valid[0])/valid_batch_size, shuffle=False)
    if test:
        kf_test = KFold(len(test[0]), n_folds=len(test[0])/valid_batch_size, shuffle=False)

    # history_errs is a bare-bones training log that holds the validation and test error
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = np.load(saveto)['history_errs'].tolist()
    best_p = None
    bad_counter = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

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

            # preprocess the caption, recording the
            # time spent to help detect bottlenecks
            pd_start = time.time()
            x, mask, ctx = prepare_data(caps,
                                        train[1],
                                        worddict,
                                        maxlen=maxlen,
                                        n_words=n_words)
            pd_duration = time.time() - pd_start
            
            #print x.shape, mask.shape, ctx.shape 
            
            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                continue

            # get the cost for the minibatch, and update the weights
            ud_start = time.time()
            #cost = f_grad_shared(x, mask, ctx, 1)
            #f_update(lrate)
            cost = f_train(x,mask,ctx,1.0)
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
                #use_noise.set_value(0.)
                #K.set_learning_phase(0.)
                x_s = x
                mask_s = mask
                ctx_s = ctx
                # generate and decode the a subset of the current training batch
                for jj in xrange(np.minimum(10, len(caps))):
                    sample, score = gen_sample(tparams, f_init, f_next, ctx_s[jj], model_options,
                                               rng=rng, k=5, maxlen=30, stochastic=False)
                    # Decode the sample from encoding back to words
                    print 'Truth ',jj,': ',
                    for vv in x_s[:,jj]:
                        if vv == 0:
                            break
                        if vv in word_idict:
                            print word_idict[vv],
                        else:
                            print 'UNK',
                    print
                    for kk, ss in enumerate([sample[0]]):
                        print 'Sample (', kk,') ', jj, ': ',
                        for vv in ss:
                            if vv == 0:
                                break
                            if vv in word_idict:
                                print word_idict[vv],
                            else:
                                print 'UNK',
                    print

            # Log validation loss + checkpoint the model with the best validation log likelihood
            if np.mod(uidx, validFreq) == 0:
                #use_noise.set_value(0.)
                train_err = 0
                valid_err = 0
                test_err = 0

                if valid:
                    valid_err = -pred_probs(f_log_probs, model_options, worddict, prepare_data, valid, kf_valid, train=0).mean()
                if test:
                    test_err = -pred_probs(f_log_probs, model_options, worddict, prepare_data, test, kf_test,train=0).mean()

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
