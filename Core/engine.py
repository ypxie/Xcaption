import numpy as np
import backend.export as T
from backend.export import RandomStreams
from collections import OrderedDict
import copy
import os
from Core.simple import *
from Core.utils_func import *
from Core.recurrent import *

def init_params(options):
    #from capgen import get_layer
    #init_fflayer = get_layer('ff')[0]
    params = OrderedDict()
    trainable = True
    # embedding: [matrix E in paper] get_name
    #params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    params = init_embeding(options, params, prefix='embeding',input_dim=options['n_words'],
                  output_dim=options['dim_word'], init='normal',trainable=trainable)
    ctx_dim = options['ctx_dim']
    proj_ctx_dim = options['proj_ctx_dim']
    lstm_cond_ndim = options['dim_word'] # originally, it accepts input from text.
    if options['lstm_encoder']: # potential feature that runs an LSTM over the annotation vectors
        # encoder: LSTM
        params = init_lstm(options, params, prefix='encoder',
                                      nin=options['dim_word'], dim=options['dim'], trainable = trainable)
        params = init_lstm(options, params, prefix='encoder_rev',
                                      nin=options['dim_word'], dim=options['dim'], trainable = trainable)
        ctx_dim = options['dim'] * 2
        lstm_cond_ndim = options['dim']
    # init_state, init_cell: [top right on page 4]
    for lidx in xrange(1, options['n_layers_init']):
        params = init_fflayer(options, params, prefix='ff_init_%d'%lidx, nin=ctx_dim, nout=ctx_dim, trainable = trainable)
    params = init_fflayer(options, params, prefix='ff_state', nin=ctx_dim, nout=options['dim'], trainable = trainable)
    params = init_fflayer(options, params, prefix='ff_memory', nin=ctx_dim, nout=options['dim'], trainable = trainable)
    # decoder: LSTM: [equation (1)/(2)/(3)]
    params = init_dynamic_lstm_cond(options, params, prefix='decoder',
                                       nin=lstm_cond_ndim, dim=options['dim'],
                                       ctx_dim=ctx_dim, proj_ctx_dim=proj_ctx_dim,
                                       trainable = trainable)
    # potentially deep decoder (warning: should work but somewhat untested)
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            params = init_fflayer(options, params, prefix='ff_state_%d'%lidx, nin=ctx_dim, nout=options['dim'], trainable = trainable)
            params = init_fflayer(options, params, prefix='ff_memory_%d'%lidx, nin=ctx_dim, nout=options['dim'], trainable = trainable)
            params = init_dynamic_lstm_cond(options, params, prefix='decoder_%d'%lidx,
                                               nin=options['dim'], dim=options['dim'],
                                               ctx_dim=ctx_dim, proj_ctx_dim=proj_ctx_dim,
                                               trainable = trainable)
    # readout: [equation (7)]
    params = init_fflayer(options, params, prefix='ff_logit_lstm', nin=options['dim'], 
                                nout=options['dim_word'], trainable = trainable)
    if options['ctx2out']:
        params = init_fflayer(options, params, prefix='ff_logit_ctx', nin=ctx_dim, 
                                    nout=options['dim_word'], trainable = trainable)
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            params = init_fflayer(options, params, prefix='ff_logit_h%d'%lidx, 
                                        nin=options['dim_word'], nout=options['dim_word'], trainable = trainable)
    params = init_fflayer(options, params, prefix='ff_logit', nin=options['dim_word'], 
                                nout=options['n_words'], trainable = trainable)

    return params



#from capgen import build_model
# build a training model
def build_model(tparams, options, sampling=True, dropoutrate = 0.5):
    """ Builds the entire computational graph used for training

    [This function builds a model described in Section 3.1.2 onwards
    as the convolutional feature are precomputed, some extra features
    which were not used are also implemented here.]

    Parameters
    ----------
    tparams : OrderedDict
        maps names of variables to theano shared variables
    options : dict
        big dictionary with all the settings and hyperparameters
    sampling : boolean
        [If it is true, when using stochastic attention, follows
        the learning rule described in section 4. at the bottom left of
        page 5]
    Returns
    -------
    trng: theano random number generator
        Used for dropout, stochastic attention, etc
    [x, mask, ctx]: theano variables
        x : words x #samples,
        mask: words * samples
        ctx: samples * annotation *dim
        
        Represent the captions, binary mask, and annotations
        for a single batch (see dimensions below)
    alphas: theano variables
        Attention weights
    alpha_sample: theano variable
        Sampled attention weights used in REINFORCE for stochastic
        attention: [see the learning rule in eq (12)]
    cost: theano variable
        negative log likelihood
    opt_outs: OrderedDict
        extra outputs required depending on configuration in options
    """
    options['regularizers'] = []
    
    from capgen import get_layer
    #dynamic_lstm_cond_layer =  get_layer('lstm_cond')[1]
    #fflayer = get_layer('ff')[1]
    
    rng = T.RandomStreams(1234)
    use_noise = T.variable(np.float32(0.))

    if os.environ['debug_mode'] == 'True':
        # start of debuging
        from Core.train import  get_dataset
        from fuel.homogeneous_data import HomogeneousData
        batch_size, maxlen = 12,100
        load_data, prepare_data = get_dataset(options['dataset'])
        train, valid, test, worddict = load_data(path = options['data_path'])
        train_iter = HomogeneousData(train, batch_size=batch_size, maxlen=maxlen)
        for caps in train_iter:
            x, mask, ctx = prepare_data(caps,
                                            train[1],
                                            worddict,
                                            maxlen=maxlen,
                                            n_words=options['n_words'])
            break
    else:
        # description string: #words x #samples,
        x = T.placeholder(ndim=2, name='x', dtype='int64')
        mask = T.placeholder(ndim=2, name='mask')
        # context: #samples x #annotations x dim
        ctx = T.placeholder(ndim=3, name='ctx')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # index into the word embedding matrix, shift it forward in time
    emb = embeding_layer(tparams, x, options, prefix='embeding',dropout=None)
    emb_shifted = T.zeros_like(emb)
    emb_shifted = T.assign_subtensor(emb_shifted, emb[:-1], slice(1,None,1))
    emb = emb_shifted
    
    #emb = tparams['embeding_W'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    #emb_shifted = T.zeros_like(emb)
    #emb_shifted = T.set_subtensor(emb_shifted[1:], emb[:-1])
    #emb = emb_shifted
    
    if options['lstm_encoder']:
        # encoder
        ctx_fwd = lstm_layer(tparams, ctx.dimshuffle(1,0,2),
                                       options, prefix='encoder')[0].dimshuffle(1,0,2)
        ctx_rev = lstm_layer(tparams, ctx.dimshuffle(1,0,2)[:,::-1,:],
                                       options, prefix='encoder_rev')[0][:,::-1,:].dimshuffle(1,0,2)
        ctx0 = T.concatenate((ctx_fwd, ctx_rev), axis=2)
    else:
        ctx0 = ctx

    # initial state/cell [top right on page 4]
    ctx_mean = ctx0.mean(1)
    for lidx in xrange(1, options['n_layers_init']):
        ctx_mean = fflayer(tparams, ctx_mean, options,
                                      prefix='ff_init_%d'%lidx, activation='softplus')
        if options['use_dropout']:
            ctx_mean = dropout_layer(ctx_mean, rng = rng, dropoutrate = options['use_dropout'])

    init_state = fflayer(tparams, ctx_mean, options, prefix='ff_state', activation='tanh')
    init_memory = fflayer(tparams, ctx_mean, options, prefix='ff_memory', activation='tanh')
    # lstm decoder
    # [equation (1), (2), (3) in section 3.1.2]
    attn_updates = []
    proj, updates = dynamic_lstm_cond_layer(tparams, emb, options,
                                              prefix='decoder', mask=mask, context=ctx0,
                                              one_step=False,
                                              init_state=init_state,
                                              init_memory=init_memory,
                                              rng=rng,
                                              sampling=sampling)
    attn_updates += updates
    print updates
    proj_h = proj[0]
    # optional deep attention
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            init_state = fflayer(tparams, ctx_mean, options, prefix='ff_state_%d'%lidx, activation='tanh')
            init_memory = fflayer(tparams, ctx_mean, options, prefix='ff_memory_%d'%lidx, activation='tanh')
            proj, updates = dynamic_lstm_cond_layer(tparams, proj_h, options,
                                                      prefix='decoder_%d'%lidx,
                                                      mask=mask, context=ctx0,
                                                      one_step=False,
                                                      init_state=init_state,
                                                      init_memory=init_memory,
                                                      rng=rng,
                                                      sampling=sampling)
            print updates            
            attn_updates += updates
            
            proj_h = proj[0]

    alphas = proj[2]
    alpha_sample = proj[3]
    ctxs = proj[4]

    # [beta value explained in note 4.2.1 "doubly stochastic attention"]
    if options['selector']:
        sels = proj[5]

    if options['use_dropout']:
        proj_h = dropout_layer(proj_h, rng=rng, dropoutrate = options['use_dropout'])

    # compute word probabilities
    # [equation (7)]
    logit = fflayer(tparams, proj_h, options, prefix='ff_logit_lstm', activation='linear')
    if options['prev2out']:
        logit += emb
    if options['ctx2out']:
        logit += fflayer(tparams, ctxs, options, prefix='ff_logit_ctx', activation='linear')
    logit = T.tanh(logit)
    if options['use_dropout']:
        logit = dropout_layer(logit, rng=rng, dropoutrate = options['use_dropout'])
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            logit = fflayer(tparams, logit, options, prefix='ff_logit_h%d'%lidx, activation='softplus')
            if options['use_dropout']:
                logit = dropout_layer(logit, rng=rng, dropoutrate = options['use_dropout'])

    # compute softmax
    logit = fflayer(tparams, logit, options, prefix='ff_logit', activation='linear')
    logit_shp = logit.shape
    probs = T.softmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    # Index into the computed probability to give the log likelihood
    x_flat = x.flatten()
    p_flat = probs.flatten()
    cost = -T.log(p_flat[T.arange(x_flat.shape[0])*probs.shape[1]+x_flat]+1e-8)
    cost = cost.reshape([x.shape[0], x.shape[1]])
    masked_cost = cost * mask
    cost = (masked_cost).sum(0)

    # optional outputs
    opt_outs = dict()
    if options['selector']:
        opt_outs['selector'] = sels
    if options['attn_type'] == 'stochastic':
        opt_outs['masked_cost'] = masked_cost # need this for reinforce later
        opt_outs['attn_updates'] = attn_updates # this is to update the rng

    return rng,use_noise,  [x, mask, ctx], alphas, alpha_sample, cost, opt_outs

# build a sampler
def build_sampler(tparams, options, rng, sampling=True,dropoutrate = 0.5):
    """ Builds a sampler used for generating from the model
    Parameters
    ----------
        See build_model function above, since we only run one step, so 
        the dimension of RNN input will be one dimension less.
    Returns
    -------
    f_init : theano function
        Input: annotation, Output: initial lstm state and memory
        (also performs transformation on ctx0 if using lstm_encoder)
    f_next: theano function
        Takes the previous word/state/memory + ctx0 and runs ne
        step through the lstm (used for beam search)
    ------
    f_init
    Parameters
    ----------
      ctx_2d: annotation * dimension
    Returns
      ctx0: the encoded context information, annotation * dim
      init_state:  list of tensor of shape 1*hidden dimension
      init_memory: list of tensor of shape 1*hidden
      [ctx0[0]]+init_state+init_memory
      
    f_next
    Parameters
    ----------
    ctx_2d: annotation * dimension     
    x : vector contains indexs of word for one image
        
    """
    
    if  os.environ['debug_mode'] == 'True':
        # start of debuging
        from Core.train import  get_dataset
        from fuel.homogeneous_data import HomogeneousData
        batch_size, maxlen = 12,100
        load_data, prepare_data = get_dataset(options['dataset'])
        train, valid, test, worddict = load_data(path = options['data_path'])
        train_iter = HomogeneousData(train, batch_size=batch_size, maxlen=maxlen)
        for caps in train_iter:
            x_init, mask_init, ctx_init = prepare_data(caps,
                                            train[1],
                                            worddict,
                                            maxlen=maxlen,
                                            n_words=options['n_words'])
            break
        ctx_2d = ctx_init[0]            
    else:
        # context: #annotations x dim, the features are NOT row by col by dim.
        ctx_2d = T.placeholder(ndim=2, name= 'ctx_sampler')
        # we need to make ctx compatible with lstm dimension requirement
        # now it is annotation*dim
    
    ctx = T.expand_dims(ctx_2d, dim=0)  

    if options['lstm_encoder']:
        # encoder
        ctx_fwd = lstm_layer(tparams, ctx.dimshuffle(1,0,2),
                                       options, prefix='encoder')[0].dimshuffle(1,0,2)
        ctx_rev = lstm_layer(tparams, ctx.dimshuffle(1,0,2)[:,::-1,:],
                                       options, prefix='encoder_rev')[0][:,::-1,:].dimshuffle(1,0,2)
        ctx0 = T.concatenate((ctx_fwd, ctx_rev), axis=2)
    else:
        ctx0 = ctx
    # initial state/cell [top right on page 4]
    ctx_mean = ctx0.mean(1)
    for lidx in xrange(1, options['n_layers_init']):
        ctx_mean = fflayer(tparams, ctx_mean, options,
                                      prefix='ff_init_%d'%lidx, activation='softplus')
        if options['use_dropout']:
            ctx_mean = dropout_layer(ctx_mean, rng = rng, dropoutrate = options['use_dropout'])

    init_state =  [fflayer(tparams, ctx_mean, options, prefix='ff_state', activation='tanh')]
    init_memory = [fflayer(tparams, ctx_mean, options, prefix='ff_memory', activation='tanh')]
    init_alpha =  [T.alloc(0., ctx.shape[0], ctx.shape[1])]

    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            init_state.append(fflayer(tparams, ctx_mean, options, prefix='ff_state_%d'%lidx, activation='tanh'))
            init_memory.append( fflayer(tparams, ctx_mean, options, prefix='ff_memory_%d'%lidx, activation='tanh'))
            init_alpha.append(T.alloc(0., ctx.shape[0], ctx.shape[1]))
    print 'Building f_init...',
    f_init = T.function([ctx_2d,T.learning_phase()], [ctx0[0]]+init_state+init_memory + init_alpha
                         , name='f_init', profile=False,on_unused_input='warn')
    print 'Done'
    
    # build f_next
    if os.environ['debug_mode'] == 'True':
        # start of debuging
        ctx_2d = ctx_init[0]
        ctx = T.expand_dims(ctx_2d, dim=0)    
        x   = x_init[0,0:1] #although we only want one time step * one sample, we make it to a vector of 
        #because we already have init_state and init_memory
    else:
        # context: #annotations x dim, the features are NOT row by col by dim.
        ctx_2d = T.placeholder(ndim=2, name='ctx_sampler')
        ctx = T.expand_dims(ctx_2d, dim=0)  
        x = T.placeholder(ndim=1, name = 'x_sampler', dtype='int64')
  
        init_state = [T.placeholder(ndim=2, name='init_state')]
        init_memory = [T.placeholder(ndim=2, name='init_memory')]
        init_alpha = [T.placeholder(ndim=2, name='init_alpha')]

        if options['n_layers_lstm'] > 1:
            for lidx in xrange(1, options['n_layers_lstm']):
                init_state.append(T.placeholder(ndim=2, name='init_state_' + str(lidx)))
                init_memory.append(T.placeholder(ndim=2, name='init_memory_' + str(lidx))) 
                init_alpha.append(T.placeholder(ndim=2, name='init_alpha_' + str(lidx))) 
                
    # for the first word (which is coded with -1), emb should be all zero
    #emb = T.switch(x[:,None] < 0, T.alloc(0., 1, tparams['Wemb'].shape[1]),
    #                    tparams['Wemb'][x])    
    emb = embeding_layer(tparams, x, options, prefix='embeding',dropout=None,
                    specifier=-1,filled_value=0.)

    proj = dynamic_lstm_cond_layer(tparams, emb, options,
                                     prefix='decoder',
                                     mask=None, context=ctx,
                                     one_step=True,
                                     init_state=init_state[0],
                                     init_memory=init_memory[0],
                                     init_alpha = init_alpha[0],
                                     rng=rng,
                                     sampling=sampling)

    next_state, next_memory, ctxs, next_alpha = [proj[0]], [proj[1]], [proj[4]], [proj[2]]
    proj_h = proj[0]
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            proj = dynamic_lstm_cond_layer(tparams, proj_h, options,
                                             prefix='decoder_%d'%lidx,
                                             context=ctx,
                                             one_step=True,
                                             init_state=init_state[lidx],
                                             init_memory=init_memory[lidx],
                                             init_alpha = init_alpha[lidx],
                                             rng=rng,
                                             sampling=sampling)
            next_state.append(proj[0])
            next_memory.append(proj[1])
            ctxs.append(proj[4])
            next_alpha.append(proj[2])
            proj_h = proj[0]

    if options['use_dropout']:
        proj_h = dropout_layer(proj[0], rng=rng, dropoutrate = options['use_dropout'])
    else:
        proj_h = proj[0]
    logit = fflayer(tparams, proj_h, options, prefix='ff_logit_lstm', activation='linear')
    if options['prev2out']:
        logit += emb
    if options['ctx2out']:
        logit += fflayer(tparams, ctxs[-1], options, prefix='ff_logit_ctx', activation='linear')
    logit = T.tanh(logit)
    if options['use_dropout']:
        logit = dropout_layer(logit, rng=rng, dropoutrate = options['use_dropout'])
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            logit = fflayer(tparams, logit, options, prefix='ff_logit_h%d'%lidx, activation='softplus')
            if options['use_dropout']:
                logit = dropout_layer(logit, rng=rng, dropoutrate = options['use_dropout'])
    logit = fflayer(tparams, logit, options, prefix='ff_logit', activation='linear')
    logit_shp = logit.shape
    next_probs = T.softmax(logit)
    next_sample = T.multinomial(p=next_probs).argmax(1)

    # next word probability
    print "Building f_next..."
    f_next = T.function([x, ctx_2d,T.learning_phase()]+init_state+init_memory + init_alpha,
                        [next_probs, next_sample]+next_state+next_memory + next_alpha, 
                        name='f_next', profile=False,on_unused_input='warn')
    print 'Done'
    return f_init, f_next

#generate sample
def gen_sample(tparams, f_init, f_next, ctx0, options, rng=None, k=1, maxlen=30, stochastic=False):
    """Generate captions with beam search.

    This function uses the beam search algorithm to conditionally
    generate candidate captions. Supports beamsearch and stochastic
    sampling.

    Parameters
    ----------
    tparams : OrderedDict()
        dictionary of theano shared variables represented weight
        matricies
    f_init : theano function
        input: annotation, output: initial lstm state and memory
        (also performs transformation on ctx0 if using lstm_encoder)
    f_next: theano function
        takes the previous word/state/memory + ctx0 and runs one
        step through the lstm
    ctx0 : np array
        annotation from convnet, of dimension #annotations x # dimension
        [e.g (196 x 512)]
    options : dict
        dictionary of flags and options
    trng : random number generator
    k : int
        size of beam search
    maxlen : int
        maximum allowed caption size
    stochastic : bool
        if True, sample stochastically

    Returns
    -------
    sample : list of list
        each sublist contains an (encoded) sample from the model
    sample_score : np array
        scores of each sample
    """
    if k > 1:
        assert not stochastic, 'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype('float32')
    hyp_states = []
    hyp_memories = []

    # only matters if we use lstm encoder
    learning_phase = np.uint8(0)
    rval = f_init(ctx0,learning_phase)
    ctx0 = rval[0]
    next_state = []
    next_memory = []
    next_alpha = []
    # the states are returned as a: (dim,) and this is just a reshape to (1, dim)
    for lidx in xrange(options['n_layers_lstm']):
        next_state.append(rval[1+lidx])
        next_state[-1] = next_state[-1].reshape([1, next_state[-1].size])
    for lidx in xrange(options['n_layers_lstm']):
        next_memory.append(rval[1+options['n_layers_lstm']+lidx])
        next_memory[-1] = next_memory[-1].reshape([1, next_memory[-1].size])
    for lidx in xrange(options['n_layers_lstm']):
        next_alpha.append(rval[1+2*options['n_layers_lstm']+lidx])
        next_alpha[-1] = next_alpha[-1].reshape([1, next_alpha[-1].size])

    # reminder: if next_w = -1, the switch statement
    # in build_sampler is triggered -> (empty word embeddings)
    next_w = -1 * np.ones((1,)).astype('int64')

    for ii in xrange(maxlen):
        # our "next" state/memory in our previous step is now our "initial" state and memory
        rval = f_next(*([next_w, ctx0, learning_phase]+next_state+next_memory+next_alpha))
        next_p = rval[0]
        next_w = rval[1]

        # extract all the states and memories
        next_state = []
        next_memory = []
        next_alpha = []
        for lidx in xrange(options['n_layers_lstm']):
            next_state.append(rval[2+lidx])
            next_memory.append(rval[2+options['n_layers_lstm']+lidx])
            next_alpha.append(rval[2+2*options['n_layers_lstm']+lidx])
        if stochastic:
            sample.append(next_w[0]) # if we are using stochastic sampling this easy
            sample_score += next_p[0,next_w[0]]
            if next_w[0] == 0:
                break
        else:
            cand_scores = hyp_scores[:,None] - np.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)] # (k-dead_k) np array of with min nll

            voc_size = next_p.shape[1]
            # indexing into the correct selected captions, the flat index is originally #sentence*voc_size
            trans_indices = ranks_flat / voc_size  
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat] # extract costs from top hypothesis

            # a bunch of lists to hold future hypothesis
            new_hyp_samples = []
            new_hyp_scores = np.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            new_hyp_memories = []
            new_hyp_alphas = []
            for lidx in xrange(options['n_layers_lstm']):
                new_hyp_states.append([])
                new_hyp_memories.append([])
                new_hyp_alphas.append([])
            # get the corresponding hypothesis and append the predicted word
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx]) # copy in the cost of that hypothesis
                for lidx in xrange(options['n_layers_lstm']):
                    new_hyp_states[lidx].append(copy.copy(next_state[lidx][ti]))
                    new_hyp_memories[lidx].append(copy.copy(next_memory[lidx][ti]))
                    new_hyp_alphas[lidx].append(copy.copy(next_alpha[lidx][ti]))
            # check the finished samples for <eos> character
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            hyp_memories = []
            hyp_alphas = []
            for lidx in xrange(options['n_layers_lstm']):
                hyp_states.append([])           
                hyp_memories.append([])
                hyp_alphas.append([])
            # check for completed sentences.
            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1 # completed sample!
                else:
                    new_live_k += 1 # collect collect correct states/memories
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    for lidx in xrange(options['n_layers_lstm']):
                        hyp_states[lidx].append(new_hyp_states[lidx][idx])
                        hyp_memories[lidx].append(new_hyp_memories[lidx][idx])
                        hyp_alphas[lidx].append(new_hyp_alphas[lidx][idx])
            hyp_scores = np.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = np.array([w[-1] for w in hyp_samples])
            next_state = []
            next_memory = []
            next_alpha = []
            for lidx in xrange(options['n_layers_lstm']):
                next_state.append(np.array(hyp_states[lidx]))
                next_memory.append(np.array(hyp_memories[lidx]))
                next_alpha.append(np.array(hyp_alphas[lidx]))
    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


