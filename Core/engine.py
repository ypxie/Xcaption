import numpy as np
import theano
import backend.export as T
from backend.export import RandomStreams
from collections import OrderedDict
import copy

from Core.simple import *
from Core.utils_func import *
from Core.recurrent import *
from Core.optimizers import adadelta, adam, rmsprop, sgd
from Core.common import get_layer


def init_params(options):
    params = OrderedDict()
    # embedding: [matrix E in paper]
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    ctx_dim = options['ctx_dim']
    lstm_cond_ndim = options['dim_word'] # originally, it accepts input from text.
    if options['lstm_encoder']: # potential feature that runs an LSTM over the annotation vectors
        # encoder: LSTM
        params = get_layer('lstm')[0](options, params, prefix='encoder',
                                      nin=options['dim_word'], dim=options['dim'], trainable = True)
        params = get_layer('lstm')[0](options, params, prefix='encoder_rev',
                                      nin=options['dim_word'], dim=options['dim'], trainable = True)
        ctx_dim = options['dim'] * 2
        lstm_cond_ndim = options['dim']
    # init_state, init_cell: [top right on page 4]
    for lidx in xrange(1, options['n_layers_init']):
        params = get_layer('ff')[0](options, params, prefix='ff_init_%d'%lidx, nin=ctx_dim, nout=ctx_dim, trainable = True)
    params = get_layer('ff')[0](options, params, prefix='ff_state', nin=ctx_dim, nout=options['dim'], trainable = True)
    params = get_layer('ff')[0](options, params, prefix='ff_memory', nin=ctx_dim, nout=options['dim'], trainable = True)
    # decoder: LSTM: [equation (1)/(2)/(3)]
    params = get_layer('lstm_cond')[0](options, params, prefix='decoder',
                                       nin=lstm_cond_ndim, dim=options['dim'],
                                       dimctx=ctx_dim, trainable = True)
    # potentially deep decoder (warning: should work but somewhat untested)
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            params = get_layer('ff')[0](options, params, prefix='ff_state_%d'%lidx, nin=options['dim'], nout=options['dim'], trainable = True)
            params = get_layer('ff')[0](options, params, prefix='ff_memory_%d'%lidx, nin=options['dim'], nout=options['dim'], trainable = True)
            params = get_layer('lstm_cond')[0](options, params, prefix='decoder_%d'%lidx,
                                               nin=options['dim'], dim=options['dim'],
                                               dimctx=ctx_dim, trainable = True)
    # readout: [equation (7)]
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm', nin=options['dim'], 
                                nout=options['dim_word'], trainable = True)
    if options['ctx2out']:
        params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx', nin=ctx_dim, 
                                    nout=options['dim_word'], trainable = True)
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            params = get_layer('ff')[0](options, params, prefix='ff_logit_h%d'%lidx, 
                                        nin=options['dim_word'], nout=options['dim_word'], trainable = True)
    params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim_word'], 
                                nout=options['n_words'], trainable = True)

    return params


      
# build a training model
def build_model(tparams, options, sampling=True):
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
    use_noise: theano shared variable
        flag that toggles noise on and off
    [x, mask, ctx]: theano variables
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
    
    rng = T.RandomStreams(1234)
    use_noise = T.shared(np.float32(0.))

    if options['debug'] == 1:
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
        x = T.matrix('x', dtype='int64')
        mask = T.matrix('mask', dtype='float32')
        # context: #samples x #annotations x dim
        ctx = T.tensor3('ctx', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # index into the word embedding matrix, shift it forward in time
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    emb_shifted = T.zeros_like(emb)
    emb_shifted = T.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted
    if options['lstm_encoder']:
        # encoder
        ctx_fwd = get_layer('lstm')[1](tparams, ctx.dimshuffle(1,0,2),
                                       options, prefix='encoder')[0].dimshuffle(1,0,2)
        ctx_rev = get_layer('lstm')[1](tparams, ctx.dimshuffle(1,0,2)[:,::-1,:],
                                       options, prefix='encoder_rev')[0][:,::-1,:].dimshuffle(1,0,2)
        ctx0 = T.concatenate((ctx_fwd, ctx_rev), axis=2)
    else:
        ctx0 = ctx

    # initial state/cell [top right on page 4]
    ctx_mean = ctx0.mean(1)
    for lidx in xrange(1, options['n_layers_init']):
        ctx_mean = get_layer('ff')[1](tparams, ctx_mean, options,
                                      prefix='ff_init_%d'%lidx, activ='rectifier')
        if options['use_dropout']:
            ctx_mean = dropout_layer(ctx_mean, use_noise, rng = rng)

    init_state = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state', activ='tanh')
    init_memory = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')
    # lstm decoder
    # [equation (1), (2), (3) in section 3.1.2]
    attn_updates = []
    proj, updates = get_layer('lstm_cond')[1](tparams, emb, options,
                                              prefix='decoder', mask=mask, context=ctx0,
                                              one_step=False,
                                              init_state=init_state,
                                              init_memory=init_memory,
                                              rng=rng,
                                              use_noise=use_noise,
                                              sampling=sampling)
    attn_updates += updates
    proj_h = proj[0]
    # optional deep attention
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            init_state = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state_%d'%lidx, activ='tanh')
            init_memory = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory_%d'%lidx, activ='tanh')
            proj, updates = get_layer('lstm_cond')[1](tparams, proj_h, options,
                                                      prefix='decoder_%d'%lidx,
                                                      mask=mask, context=ctx0,
                                                      one_step=False,
                                                      init_state=init_state,
                                                      init_memory=init_memory,
                                                      rng=rng,
                                                      use_noise=use_noise,
                                                      sampling=sampling)
            attn_updates += updates
            proj_h = proj[0]

    alphas = proj[2]
    alpha_sample = proj[3]
    ctxs = proj[4]

    # [beta value explained in note 4.2.1 "doubly stochastic attention"]
    if options['selector']:
        sels = proj[5]

    if options['use_dropout']:
        proj_h = dropout_layer(proj_h, use_noise, rng=rng)

    # compute word probabilities
    # [equation (7)]
    logit = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
    if options['prev2out']:
        logit += emb
    if options['ctx2out']:
        logit += get_layer('ff')[1](tparams, ctxs, options, prefix='ff_logit_ctx', activ='linear')
    logit = tanh(logit)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, rng=rng)
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit_h%d'%lidx, activ='rectifier')
            if options['use_dropout']:
                logit = dropout_layer(logit, use_noise, rng=rng)

    # compute softmax
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
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

    return rng, use_noise, [x, mask, ctx], alphas, alpha_sample, cost, opt_outs

# build a sampler
def build_sampler(tparams, options, use_noise, rng, sampling=True):
    """ Builds a sampler used for generating from the model
    Parameters
    ----------
        See build_model function above
    Returns
    -------
    f_init : theano function
        Input: annotation, Output: initial lstm state and memory
        (also performs transformation on ctx0 if using lstm_encoder)
    f_next: theano function
        Takes the previous word/state/memory + ctx0 and runs ne
        step through the lstm (used for beam search)
    """
    # context: #annotations x dim
    ctx = T.matrix('ctx_sampler', dtype='float32')
    if options['lstm_encoder']:
        # encoder
        ctx_fwd = get_layer('lstm')[1](tparams, ctx,
                                       options, prefix='encoder')[0]
        ctx_rev = get_layer('lstm')[1](tparams, ctx[::-1,:],
                                       options, prefix='encoder_rev')[0][::-1,:]
        ctx = T.concatenate((ctx_fwd, ctx_rev), axis=1)

    # initial state/cell
    ctx_mean = ctx.mean(0)
    for lidx in xrange(1, options['n_layers_init']):
        ctx_mean = get_layer('ff')[1](tparams, ctx_mean, options,
                                      prefix='ff_init_%d'%lidx, activ='rectifier')
        if options['use_dropout']:
            ctx_mean = dropout_layer(ctx_mean, use_noise, rng=rng)
    init_state = [get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state', activ='tanh')]
    init_memory = [get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')]
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            init_state.append(get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state_%d'%lidx, activ='tanh'))
            init_memory.append(get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory_%d'%lidx, activ='tanh'))

    print 'Building f_init...',
    f_init = theano.function([ctx], [ctx]+init_state+init_memory, name='f_init', profile=False)
    print 'Done'

    # build f_next
    ctx = T.matrix('ctx_sampler', dtype='float32')
    x = T.vector('x_sampler', dtype='int64')
    init_state = [T.matrix('init_state', dtype='float32')]
    init_memory = [T.matrix('init_memory', dtype='float32')]
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            init_state.append(T.matrix('init_state', dtype='float32'))
            init_memory.append(T.matrix('init_memory', dtype='float32'))

    # for the first word (which is coded with -1), emb should be all zero
    emb = T.switch(x[:,None] < 0, T.alloc(0., 1, tparams['Wemb'].shape[1]),
                        tparams['Wemb'][x])

    proj = get_layer('lstm_cond')[1](tparams, emb, options,
                                     prefix='decoder',
                                     mask=None, context=ctx,
                                     one_step=True,
                                     init_state=init_state[0],
                                     init_memory=init_memory[0],
                                     rng=rng,
                                     use_noise=use_noise,
                                     sampling=sampling)

    next_state, next_memory, ctxs = [proj[0]], [proj[1]], [proj[4]]
    proj_h = proj[0]
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            proj = get_layer('lstm_cond')[1](tparams, proj_h, options,
                                             prefix='decoder_%d'%lidx,
                                             context=ctx,
                                             one_step=True,
                                             init_state=init_state[lidx],
                                             init_memory=init_memory[lidx],
                                             rng=rng,
                                             use_noise=use_noise,
                                             sampling=sampling)
            next_state.append(proj[0])
            next_memory.append(proj[1])
            ctxs.append(proj[4])
            proj_h = proj[0]

    if options['use_dropout']:
        proj_h = dropout_layer(proj[0], use_noise, rng=rng)
    else:
        proj_h = proj[0]
    logit = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
    if options['prev2out']:
        logit += emb
    if options['ctx2out']:
        logit += get_layer('ff')[1](tparams, ctxs[-1], options, prefix='ff_logit_ctx', activ='linear')
    logit = tanh(logit)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, rng=rng)
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit_h%d'%lidx, activ='rectifier')
            if options['use_dropout']:
                logit = dropout_layer(logit, use_noise, rng=rng)
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    next_probs = T.softmax(logit)
    next_sample = T.multinomial(shape=(),pvals=next_probs, rng=rng).argmax(1)

    # next word probability
    print "Building f_next..."
    f_next = theano.function([x, ctx]+init_state+init_memory, [next_probs, next_sample]+next_state+next_memory, name='f_next', profile=False)
    print 'Done'
    return f_init, f_next

# generate sample
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
    rval = f_init(ctx0)
    ctx0 = rval[0]
    next_state = []
    next_memory = []
    # the states are returned as a: (dim,) and this is just a reshape to (1, dim)
    for lidx in xrange(options['n_layers_lstm']):
        next_state.append(rval[1+lidx])
        next_state[-1] = next_state[-1].reshape([1, next_state[-1].shape[0]])
    for lidx in xrange(options['n_layers_lstm']):
        next_memory.append(rval[1+options['n_layers_lstm']+lidx])
        next_memory[-1] = next_memory[-1].reshape([1, next_memory[-1].shape[0]])
    # reminder: if next_w = -1, the switch statement
    # in build_sampler is triggered -> (empty word embeddings)
    next_w = -1 * np.ones((1,)).astype('int64')

    for ii in xrange(maxlen):
        # our "next" state/memory in our previous step is now our "initial" state and memory
        rval = f_next(*([next_w, ctx0]+next_state+next_memory))
        next_p = rval[0]
        next_w = rval[1]

        # extract all the states and memories
        next_state = []
        next_memory = []
        for lidx in xrange(options['n_layers_lstm']):
            next_state.append(rval[2+lidx])
            next_memory.append(rval[2+options['n_layers_lstm']+lidx])

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
            # indexing into the correct selected captions
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat] # extract costs from top hypothesis

            # a bunch of lists to hold future hypothesis
            new_hyp_samples = []
            new_hyp_scores = np.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            for lidx in xrange(options['n_layers_lstm']):
                new_hyp_states.append([])
            new_hyp_memories = []
            for lidx in xrange(options['n_layers_lstm']):
                new_hyp_memories.append([])

            # get the corresponding hypothesis and append the predicted word
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx]) # copy in the cost of that hypothesis
                for lidx in xrange(options['n_layers_lstm']):
                    new_hyp_states[lidx].append(copy.copy(next_state[lidx][ti]))
                for lidx in xrange(options['n_layers_lstm']):
                    new_hyp_memories[lidx].append(copy.copy(next_memory[lidx][ti]))

            # check the finished samples for <eos> character
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            for lidx in xrange(options['n_layers_lstm']):
                hyp_states.append([])
            hyp_memories = []
            for lidx in xrange(options['n_layers_lstm']):
                hyp_memories.append([])

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
                    for lidx in xrange(options['n_layers_lstm']):
                        hyp_memories[lidx].append(new_hyp_memories[lidx][idx])
            hyp_scores = np.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = np.array([w[-1] for w in hyp_samples])
            next_state = []
            for lidx in xrange(options['n_layers_lstm']):
                next_state.append(np.array(hyp_states[lidx]))
            next_memory = []
            for lidx in xrange(options['n_layers_lstm']):
                next_memory.append(np.array(hyp_memories[lidx]))

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


def pred_probs(f_log_probs, options, worddict, prepare_data, data, iterator, verbose=False):
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
        pred_probs = f_log_probs(x,mask,ctx)
        probs[valid_index] = pred_probs[:,None]

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples computed'%(n_done,n_samples)

    return probs
