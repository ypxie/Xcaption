import numpy as np
import backend.export as T
from backend.export import RandomStreams
from collections import OrderedDict
import copy
import os
from Core.simple import *
from Core.utils_func import *
from Core.recurrent import *
from Core.convolution import Convolution2D, MaxPooling2D

from utils.regularizers import l2
from utils.activations import elu, relu

def get_conv_feature(tparams, options, inputs, params = None, weight_decay = 1e-7,prefix = 'conv_feat',
                      img_channels=3,dropoutrate = 0.5, trainable = True, belonging_Module=None, **kwargs):
    if not belonging_Module:
        belonging_Module = options['belonging_Module'] if 'belonging_Module' in options else None

    module_identifier = get_module_identifier(prefix)
    init_ModuleInfo(options, name = module_identifier)
    options['belonging_Module'] = module_identifier 

    params = OrderedDict() if params is None else params

    activ = relu
    inputs.name = 'input'
    if hasattr(inputs, '_keras_shape'):
        print inputs._keras_shape
        if inputs._keras_shape[-3] != img_channels:
            raise Exception('Wrong Inputs channel for conv_feat!')
    else:
        inputs._keras_shape = (None, img_channels, None, None)
        
    W_regularizer = l2(weight_decay)
    b_regularizer=l2(weight_decay)
    
    conv_1 = Convolution2D(options, 32,3,3, belonging_Module=module_identifier, prefix='conv_1',init='orthogonal', 
                            border_mode='same', activation=activ,  W_regularizer=W_regularizer, 
                            b_regularizer=b_regularizer, trainable=trainable)(inputs, tparams, options,params = params) 
    max_1 = MaxPooling2D(pool_size=(2,2))(conv_1)
    
    conv_2 = Convolution2D(options, 64,3,3, belonging_Module=module_identifier,prefix='conv_2',init='orthogonal', 
                            border_mode='same', activation=activ, W_regularizer=W_regularizer, 
                            b_regularizer=b_regularizer, trainable=trainable)(max_1,tparams, options,params = params)
    max_2 = MaxPooling2D( pool_size=(2,2))(conv_2)
    
    dp_0 =  dropout_layer(dropoutrate = 0.25)(max_2)
    
    conv_3 = Convolution2D(options, 128,3,3, belonging_Module=module_identifier, prefix='conv_3',init='orthogonal', 
                            border_mode='same', activation=activ, W_regularizer=W_regularizer, 
                            b_regularizer=b_regularizer, trainable=trainable)(dp_0,tparams, options,params = params)  # 25
    max_3 = MaxPooling2D(pool_size = (2,2))(conv_3)                                                    # 12
    
    conv_4 = Convolution2D(options, 256,3,3, belonging_Module=module_identifier,prefix='conv_4',init='orthogonal', 
                            border_mode='same', activation=activ, W_regularizer=W_regularizer, 
                            b_regularizer=b_regularizer, trainable=trainable)(max_3,tparams, options,params = params )  # 12
    max_4 = MaxPooling2D(pool_size = (2,2))(conv_4)            # 6
    
    dp_1 =  dropout_layer(dropoutrate = 0.25)(max_4)
    
    conv_5 = Convolution2D(options, 512,3,3,belonging_Module=module_identifier,prefix='conv_5',init='orthogonal', 
                            border_mode='same', activation=activ,W_regularizer=W_regularizer, 
                            b_regularizer=b_regularizer, trainable=trainable)(dp_1,tparams, options,params = params)  # 6

    max_5 = MaxPooling2D(pool_size = (2,2))(conv_5)
    dp = dropout_layer(dropoutrate = 0.25)(max_5)
    conv_6 = Convolution2D(options, 1024,3,3,belonging_Module=module_identifier,prefix='conv_6',init='orthogonal',
                            border_mode='same', activation=activ,W_regularizer=W_regularizer,
                            b_regularizer=b_regularizer, trainable=trainable)(dp,tparams, options,params = params)  # 3

    update_father_module(options,belonging_Module, module_identifier)

    options['belonging_Module'] = None 
    print('Finished build conv_feat module')
    return [conv_5, conv_4, conv_3]

def build_model_single(tparams, options, x, mask, featSource, params = None, prefix = 'atten_model',
                       sampling=True, dropoutrate = 0.5, belonging_Module = None, trainable = True):
    """ Builds the entire computational graph used for training
    
    if tparams, params are both empty OrderedDict, start from fresh.
    if params is given,  start from params
    else:
        start from tparam
    
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
    #activ = T.tanh
    init_activ = T.tanh
    if not belonging_Module:
        belonging_Module = options['belonging_Module'] if belonging_Module in options else None
    
    if not 'online_feature' in options:
        options['online_feature'] = False

    module_identifier = get_module_identifier(prefix)
    init_ModuleInfo(options, name = module_identifier)
    options['belonging_Module'] = module_identifier # make sure you make it None after this function
    
    if options['online_feature'] == True:
       # means we need to recompute the feature from ctx, which is actually row image.
       feat_pool = get_conv_feature(tparams, options, featSource, params = params, 
                                    img_channels=3,dropoutrate = 0.25, trainable = options['train_feat_encoder'])
       #feat_index = options['feat_index']
       feat_pool = feat_pool[0:options['num_attention']]
       context_list = []
       for ctx in feat_pool:
            ks = getattr(ctx, '_keras_shape', None)
            ctx = T.reshape(ctx, (ctx.shape[0], ctx.shape[1], -1))
            ctx = np.transpose(ctx, (0, 2, 1))
            ctx._keras_shape = (None, None, ks[1])
            context_list.append(ctx)
       
    elif options['online_feature'] == False:
       # means source is already comouted feature
       #ctx = featSource
       context_list = featSource
    options['regularizers'] = []
    #ctx_dim = options['ctx_dim']
    ctx_dim_list = options['ctx_dim_list']

    proj_ctx_dim = options['proj_ctx_dim']
    lstm_cond_ndim = options['dim_word'] # originally, it accepts input from text.

    rng = T.RandomStreams(1234)
    use_noise = T.variable(np.float32(0.))

    emb = Embedding(options, prefix='embeding',input_dim = options['n_words'], output_dim=options['dim_word'], 
                    init='normal',trainable=trainable)(x, tparams, options, params = params,)
    emb_shifted = T.zeros_like(emb)
    emb_shifted = T.assign_subtensor(emb_shifted, emb[:-1], slice(1,None,1))
    emb = emb_shifted

    ctx0_list = []
    ctx_mean_list = []
    for idx, ctx in enumerate(context_list):
        if options['lstm_encoder']:
            # encoder
            ctx_fwd = LSTM(options,prefix='encoder_cidx_%d'%idx, nin=ctx_dim_list[idx], dim= ctx_dim_list[idx]/2, trainable = trainable)\
                        (T.permute_dimensions(ctx, (1,0,2)),tparams,  options, params = params, )[0]
            ctx_fwd =  T.permute_dimensions(ctx_fwd,(1,0,2) )

            ctx_rev = LSTM(options,prefix='encoder_rev_cidx_%d'%idx, nin=ctx_dim_list[idx], dim=ctx_dim_list[idx]/2, trainable = trainable)\
                        (T.reverse(T.permute_dimensions(ctx, (1,0,2)), axis = 1),tparams, options, params = params)[0]
            ctx_rev =  T.reverse(ctx_rev,  axis = 1 )
            ctx_rev =  T.permute_dimensions(ctx_rev,(1,0,2) )
            
            ctx0 = T.concatenate((ctx_fwd, ctx_rev), axis=2)
        else:
            ctx0 = ctx
        ctx0_list.append(ctx0)
        
        # initial state/cell [top right on page 4]
        ctx_mean = T.mean(ctx0, axis=1) #ctx0.mean(1)
        #print ctx_mean._keras_shape
        for lidx in xrange(1, options['n_layers_init']):
            ctx_mean = Dense(options,prefix='ff_init_layer_init_{0}_cidx_{1}'.format(lidx, idx), nout=ctx_dim_list[idx], trainable = trainable,
                            activation=init_activ)(ctx_mean,tparams, options, params = params)
            if options['use_dropout']:
                ctx_mean = dropout_layer(rng = rng, dropoutrate = options['use_dropout'])(ctx_mean)
        ctx_mean_list.append(ctx_mean)
    ctx_mean_concate = T.concatenate(ctx_mean_list, axis=-1)
    init_state  = Dense(options, prefix='ff_state', nout=options['dim'], trainable = trainable, 
                        activation=init_activ)(ctx_mean_concate, tparams, options, params = params)
    init_memory = Dense(options, prefix='ff_memory', nout=options['dim'], trainable = trainable, 
                        activation=init_activ)(ctx_mean_concate, tparams, options, params = params)

    # lstm decoder
    # [equation (1), (2), (3) in section 3.1.2]
    attn_updates = {}
    proj, updates = cond_LSTM(  options,
                                prefix='decoder', 
                                nin=lstm_cond_ndim, 
                                dim=options['dim'], 
                                proj_ctx_dim=proj_ctx_dim,
                                trainable=trainable,  
                                mask=mask, context_list=ctx0_list, 
                                one_step=False,
                                init_state=init_state,
                                init_memory=init_memory,
                                rng=rng,
                                sampling=sampling)(emb, tparams, options, params = params)

    #attn_updates.updates(updates)   
    attn_updates = updates
    #attn_updates.extend(updates.items())   
    proj_h = proj[0]
    alphas_list = [proj[2]]
    alpha_sample_list = [proj[3]]
    ctxs_list = [proj[4]]
    # optional deep attention
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            init_state  = Dense(options, prefix='ff_state_%d'%lidx,  nout=options['dim'], trainable = trainable, 
                                activation= init_activ)(ctx_mean_concate, tparams, options, params = params)
            init_memory = Dense(options, prefix='ff_memory_%d'%lidx, nout=options['dim'], trainable = trainable, 
                                activation= init_activ)(ctx_mean_concate, tparams, options, params = params)
            
            proj, updates = cond_LSTM(  options,
                                        prefix='decoder_%d'%lidx,
                                        nin=lstm_cond_ndim, 
                                        dim=options['dim'], 
                                        proj_ctx_dim=proj_ctx_dim,
                                        trainable=trainable,  
                                        mask=mask, context_list=ctx0_list, 
                                        one_step=False,
                                        init_state=init_state,
                                        init_memory=init_memory,
                                        rng=rng,
                                        sampling=sampling)(proj_h,tparams, options, params = params )
            #attn_updates.extend(updates.items()) 

            #attn_updates.updates(updates)
            proj_h = proj[0]

            alphas_list.append(proj[2])
            alpha_sample_list.append(proj[3])
            ctxs_list.append(proj[4])


    # [beta value explained in note 4.2.1 "doubly stochastic attention"]
    #if options['selector']:
    #    sels_list = proj[5]
    if options['use_dropout']:
        proj_h = dropout_layer(rng=rng, dropoutrate = options['use_dropout'])(proj_h)
    
    # compute word probabilities
    # [equation (7)]
    logit = Dense(options,prefix='ff_logit_lstm',nin=options['dim'], nout=options['dim_word'], trainable = trainable, 
                  activation='linear')(proj_h, tparams,options, params = params)

    if options['prev2out']:
        logit += emb
    if options['ctx2out']:
        for idx, ctxs in enumerate(ctxs_list[-1]):
            logit += Dense(options, prefix='ff_logit_ctx_cidx_%d'%idx,  nout=options['dim_word'], trainable = trainable,
                            activation='linear')(ctxs, tparams, options, params = params)
                  
    logit = T.tanh(logit)
    
    if options['use_dropout']:
        logit = dropout_layer(rng=rng, dropoutrate = options['use_dropout'])(logit)
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            logit = Dense(options, prefix='ff_logit_h%d'%lidx, nin=options['dim_word'], nout=options['dim_word'], 
                          trainable = trainable, activation = 'tanh')(logit, tparams, options, params = params)
            
            if options['use_dropout']:
                logit = dropout_layer(rng=rng, dropoutrate = options['use_dropout'])(logit)

    # compute softmax
    logit = Dense(options, prefix='ff_logit', nin=options['dim_word'],nout=options['n_words'], 
                   trainable = trainable, activation='linear')(logit, tparams, options, params = params)

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
    #if options['selector']:
    #    opt_outs['selector'] = sels_list
    if options['attn_type'] == 'stochastic' or options['hard_sampling'] == True :
        opt_outs['masked_cost'] = masked_cost # need this for reinforce later
        opt_outs['attn_updates'] = attn_updates # this is to update the rng
    
    options['belonging_Module'] = None 
    
    return rng,use_noise,  [x, mask, featSource], expand_list(alphas_list), expand_list(alpha_sample_list), cost, opt_outs

def get_reward(probs, x_flat):
    '''

    Parameters
    ----------
    probs: times * batchsize * n_words
    x_flat: times * batchsize

    Returns
    the rewards.
    ------
    '''
    pass
def parse_init_values(options, init_values,reshape=False):
    next_state = []
    next_memory = []
    next_alpha = [] #should be list of list
    interv = options['num_attention']
    # the states are returned as a: (dim,) and this is just a reshape to (1, dim)
    for lidx in xrange(options['n_layers_lstm']):
        next_state.append(init_values[lidx])
        if T.ndim(next_state[-1]) == 1 and reshape:
            next_state[-1] = next_state[-1].reshape([1, next_state[-1].size])
    for lidx in xrange(options['n_layers_lstm']):
        next_memory.append(init_values[options['n_layers_lstm']+lidx])
        if T.ndim(next_memory[-1]) == 1 and reshape:
            next_memory[-1] = next_memory[-1].reshape([1, next_memory[-1].size])

    for lidx in xrange(options['n_layers_lstm']):
        this_start = 2*options['n_layers_lstm']+lidx*interv
        next_alpha.append(init_values[this_start:this_start+interv])
        for idx in range(len(next_alpha[-1])):
            if T.ndim(next_alpha[-1][idx]) == 1 and reshape:
                next_alpha[-1][idx] = next_alpha[-1][idx].reshape([1, next_alpha[-1][idx].size])
    
    return next_state, next_memory,next_alpha

# build a sampler
def build_sampler(tparams, options, rng, x_1d, featSource,  sampling=True, dropoutrate = 0.5,allow_input_downcast=True):
    #activ = T.tanh
    init_activ = T.tanh
    #http://stackoverflow.com/questions/8934772/assigning-to-variable-from-parent-function-local-variable-referenced-before-as
    interv = options['num_attention']
    def f_init_clousure(featSource=featSource, learning_phase = T.learning_phase()):
        if options['online_feature'] == True:
            featSource = T.expand_dims(featSource, dim=0)
           # means we need to recompute the feature from ctx, which is actually row image.
            feat_pool = get_conv_feature(tparams, options, featSource,
                                            img_channels=3,dropoutrate = 0.25)
            feat_pool = feat_pool[0:interv]

            context_list = []
            for ctx in feat_pool:
                ks = getattr(ctx, '_keras_shape', None)
                ctx = T.reshape(ctx, (ctx.shape[0], ctx.shape[1], -1))
                ctx = np.transpose(ctx, (0, 2, 1))
                ctx._keras_shape = (None, None, ks[1])
                context_list.append(ctx)
        elif options['online_feature'] == False:
            # means source is already comouted feature
            context_list = [f[None] for f in featSource] # in this case it should be list of tensors.
        ctx0_list = []
        ctx_mean_list = []
        for idx, ctx in enumerate(context_list):
            if options['lstm_encoder']:
                # encoder
                ctx_fwd = lstm_layer(tparams, T.permute_dimensions(ctx, (1,0,2)),
                                            options, prefix='encoder_cidx_%d'%idx)[0]
                ctx_fwd =  T.permute_dimensions(ctx_fwd,(1,0,2))
        
                ctx_rev = lstm_layer(tparams, T.reverse(T.permute_dimensions(ctx, (1,0,2)), axis = 1),
                                            options, prefix='encoder_rev_cidx_%d'%idx)[0]
                ctx_rev =  T.reverse(ctx_rev,  axis = 1 )
                ctx_rev =  T.permute_dimensions(ctx_rev,(1,0,2) )                               
                ctx0 = T.concatenate((ctx_fwd, ctx_rev), axis=2)
            else:
                ctx0 = ctx
            ctx0_list.append(ctx0)
            # initial state/cell [top right on page 4]
            ctx_mean = T.mean(ctx0, axis=1) #ctx0.mean(1)
            for lidx in xrange(1, options['n_layers_init']):
                ctx_mean = fflayer(tparams, ctx_mean, options,
                                            prefix='ff_init_layer_init_{0}_cidx_{1}'.format(lidx, idx), activation=init_activ)
                if options['use_dropout']:
                    ctx_mean = dropout_layer(rng = rng, dropoutrate = options['use_dropout'])(ctx_mean)
            ctx_mean_list.append(ctx_mean)

        ctx_mean_concate = T.concatenate(ctx_mean_list, axis=-1)
        init_state =  [fflayer(tparams, ctx_mean_concate, options, prefix='ff_state', activation= init_activ)]
        init_memory = [fflayer(tparams, ctx_mean_concate, options, prefix='ff_memory', activation= init_activ)]
        this_alpha_sampling_list = [T.alloc(0., (ctx0.shape[0], ctx0.shape[1]), broadcastable=False ) for ctx0 in ctx0_list]
        init_alpha_sampling_list = [this_alpha_sampling_list]

        if options['n_layers_lstm'] > 1:
            for lidx in xrange(1, options['n_layers_lstm']):
                init_state.append(fflayer(tparams, ctx_mean_concate, options, prefix='ff_state_%d'%lidx, activation=init_activ))
                init_memory.append( fflayer(tparams, ctx_mean_concate, options, prefix='ff_memory_%d'%lidx, activation=init_activ))
                this_alpha_sampling_list = [T.alloc(0., (ctx0.shape[0], ctx0.shape[1]), broadcastable=False ) for ctx0 in ctx0_list]
                init_alpha_sampling_list.append(this_alpha_sampling_list)
                
        print 'Building f_init...',
        outputs = [ct[0] for ct in ctx0_list] + init_state+init_memory + expand_list(init_alpha_sampling_list)
        return outputs

    init_outputs  = f_init_clousure(featSource, T.learning_phase())
    init_inputs = [featSource, T.learning_phase()]

    init_values = init_outputs[interv:]
    ctx_encoded_list = init_outputs[0:interv]

    def f_next_clousure(x_1d = x_1d, learning_phase = T.learning_phase(), *other_args):
        '''
        ctx_encoded_list should contain feature that are 1*num_slots*dim it is of 3 dimension.
        other_args: state, memory, flatten_list of alpha_Sampling, and last interv number of attention.
        '''
        other_args = list(other_args)
        assert T.ndim(x_1d) == 1, 'x_1d has to be 1 dimensional: sample'
        x = T.expand_dims(x_1d, dim=0, broadcastable = True)
        #ctx0 = T.expand_dims(ctx_2d, dim=0, broadcastable = True)
        ctx_encoded_list = other_args[0:interv]

        init_state, init_memory, init_alpha_sampling = parse_init_values(options, list(other_args[interv:]))

        emb = embeding_layer(tparams, x, options, prefix='embeding',dropout=None,
                    specifier=-1,filled_value=0.)
        proj, _ = dynamic_lstm_cond_layer(tparams, emb, options,
                                        prefix='decoder',
                                        mask=None, context_list=ctx_encoded_list,
                                        last_step = True,
                                        init_state=init_state[0],
                                        init_memory=init_memory[0],
                                        init_alpha_sampling_list = init_alpha_sampling[0],
                                        rng=rng,
                                        sampling=sampling)

        next_state, next_memory, ctxs_list, next_alpha_sampling_list = [proj[0]], [proj[1]], [proj[4]], [proj[3]]
        proj_h = proj[0]
        if options['n_layers_lstm'] > 1:
            for lidx in xrange(1, options['n_layers_lstm']):
                proj, _  = dynamic_lstm_cond_layer(tparams, proj_h, options,
                                                prefix='decoder_%d'%lidx,
                                                context_list=ctx_encoded_list,
                                                last_step = True,
                                                init_state=init_state[lidx],
                                                init_memory=init_memory[lidx],
                                                init_alpha_sampling_list = init_alpha_sampling[lidx],
                                                rng=rng,
                                                sampling=sampling)
                next_state.append(proj[0])
                next_memory.append(proj[1])
                ctxs_list.append(proj[4])
                next_alpha_sampling_list.append(proj[3])
                proj_h = proj[0]
        
        if options['use_dropout']:
            proj_h = dropout_layer( rng=rng, dropoutrate = options['use_dropout'])(proj[0])
        else:
            proj_h = proj[0]
        logit = fflayer(tparams, proj_h, options, prefix='ff_logit_lstm', activation='linear')
        if options['prev2out']:
            logit += emb[0]
        if options['ctx2out']:
            for idx, ctxs in enumerate(ctxs_list[-1]):
                logit += fflayer(tparams, ctxs, options, prefix='ff_logit_ctx_cidx_%d'%idx, activation='linear')

        logit = T.tanh(logit)
        if options['use_dropout']:
            logit = dropout_layer(rng=rng, dropoutrate = options['use_dropout'])(logit)
        if options['n_layers_out'] > 1:
            for lidx in xrange(1, options['n_layers_out']):
                logit = fflayer(tparams, logit, options, prefix='ff_logit_h%d'%lidx, activation='tanh')
                if options['use_dropout']:
                    logit = dropout_layer(rng=rng, dropoutrate = options['use_dropout'])(logit)
        logit = fflayer(tparams, logit, options, prefix='ff_logit', activation='linear')
        logit_shp = logit.shape
        next_probs = T.softmax(logit)
        next_sample = T.multinomial(pvals =next_probs).argmax(1)
        
        outputs = [next_probs, next_sample] + next_state + next_memory + expand_list(next_alpha_sampling_list)
        return outputs
    other_args = ctx_encoded_list + init_values
    next_outputs = f_next_clousure(x_1d, T.learning_phase(), *other_args)
    next_inputs = expand_list([x_1d, T.learning_phase()] + other_args )

    init_tuple = [init_inputs, init_outputs, f_init_clousure]
    next_tuple = [next_inputs, next_outputs, f_next_clousure]
    
    return init_tuple, next_tuple

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
        or It can be 3*512*512 as the images.
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
    interv = options['num_attention']
    rval = f_init(ctx0,learning_phase)
    ctx0_list = rval[0:interv]

    next_state,next_memory,next_alpha =  parse_init_values(options, rval[interv:], reshape=True)
    
    # reminder: if next_w = -1, the switch statement
    # in build_sampler is triggered -> (empty word embeddings)
    next_w = -1 * np.ones((1,)).astype('int64')

    for ii in xrange(maxlen):
        # our "next" state/memory in our previous step is now our "initial" state and memory
        next_list = expand_list(ctx0_list + next_state + next_memory + next_alpha)
        rval = f_next(*([next_w, learning_phase]+ next_list))
        next_p = rval[0]
        next_w = rval[1]

        # extract all the states and memories
        #next_state = []
        #next_memory = []
        #next_alpha = []
        #for lidx in xrange(options['n_layers_lstm']):
        #    next_state.append(rval[2+lidx])
        #    next_memory.append(rval[2+options['n_layers_lstm']+lidx])
        #    next_alpha.append(rval[2+2*options['n_layers_lstm']+lidx])
        next_state,next_memory,next_alpha =  parse_init_values(options, rval[2:], reshape=False)

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
                new_hyp_alphas.append(multi_list([], interv))
            # get the corresponding hypothesis and append the predicted word
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx]) # copy in the cost of that hypothesis
                for lidx in xrange(options['n_layers_lstm']):
                    new_hyp_states[lidx].append(copy.copy(next_state[lidx][ti]))
                    new_hyp_memories[lidx].append(copy.copy(next_memory[lidx][ti]))
                    for cidx in range(interv):
                        new_hyp_alphas[lidx][cidx].append(copy.copy(next_alpha[lidx][cidx][ti]))
            #for lidx in xrange(options['n_layers_lstm']):
            #    for cidx in range(interv):
            #        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            #            new_hyp_alphas[lidx][cidx].append(copy.copy(next_alpha[lidx][cidx][ti]))

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
                hyp_alphas.append(multi_list([], interv))
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
                        for cidx in range(interv):
                            hyp_alphas[lidx][cidx].append(new_hyp_alphas[lidx][cidx][idx])

            #if new_hyp_samples[idx][-1] != 0:
            #    for lidx in xrange(options['n_layers_lstm']):
            #        for cidx in range(interv):
            #            for idx in xrange(len(new_hyp_samples)):
            #                hyp_alphas[lidx][cidx].append(new_hyp_alphas[lidx][cidx][idx])

            hyp_scores = np.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = np.array([w[-1] for w in hyp_samples])
            next_state = []
            next_memory = []
            next_alpha = multi_list(multi_list([],interv), options['n_layers_lstm'])
            for lidx in xrange(options['n_layers_lstm']):
                next_state.append(np.array(hyp_states[lidx]))
                next_memory.append(np.array(hyp_memories[lidx]))
                for cidx in range(interv):
                    next_alpha[lidx][cidx]=(np.array(hyp_alphas[lidx][cidx]))
    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


