import numpy as np
import backend.export as T
from  backend.export import npwrapper

from  Core.utils_func  import *

from utils import activations, initializations, regularizers
#-------------------------LSTM Layer------------------------------
# This function implements the lstm fprop
def init_lstm(options, params, prefix='lstm', nin=None, dim=None,
              init='norm_weight', inner_init='othogo_weight',
              forget_bias_init='one', trainable = True, **kwargs):
    init = initializations.get(init)
    inner_init = initializations.get(inner_init)
    forget_bias_init = initializations.get(forget_bias_init)
    
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']
    """
     Stack the weight matricies for all the gates
     for much cleaner code and slightly faster dot-prods
    """
    # input weights
    W = np.concatenate([ init((nin,dim),symbolic=False),
                         init((nin,dim),symbolic=False),
                         init((nin,dim),symbolic=False),
                         init((nin,dim),symbolic=False)], axis=1)
                        
    params[get_name(prefix,'W')] = npwrapper(W, trainable = trainable)
    # for the previous hidden activation
    U = np.concatenate([ inner_init((dim,dim),symbolic=False),
                         inner_init((dim,dim),symbolic=False),
                         inner_init((dim,dim),symbolic=False),
                         inner_init((dim,dim),symbolic=False)], axis=1)
    params[get_name(prefix,'U')] =  npwrapper(U, trainable = trainable)
    
    b =   np.hstack((np.zeros(dim),
                     forget_bias_init((dim),symbolic=False) + 4,
                     np.zeros(dim),
                     np.zeros(dim)))   
    #b  = (forget_bias_init((4 * dim,) + 4,symbolic=False)+ 4) 
    params[get_name(prefix,'b')] = npwrapper(b, trainable = trainable)

    return params

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, 
               init_memory=None, init_state=None, activation='tanh', 
               inner_activation='hard_sigmoid', **kwargs):
    '''
    tparams: contains the ordredDict of symbolic parameters.
    state_below: timestep * batchsize * input_dim
    options: model configuration
    '''
    def get_dropout(shapelist=[None,None], dropoutrate = 0):
        #if self.seed is None:
        if dropoutrate is not None:
            retain_prob = 1- dropoutrate

            W1 = T.binomial(shape= shapelist[0], p = retain_prob, dtype = T.floatX)/retain_prob
            W2 = T.binomial(shape= shapelist[1], p = retain_prob, dtype = T.floatX)/retain_prob
            return [W1, W2]
        else:
            return [None,None]

    activation = activations.get(activation)
    inner_activation = activations.get(inner_activation)

    nsteps = state_below.shape[0]
    dim = tparams[get_name(prefix,'U')].shape[0]
   
    # if we are dealing with a mini-batch
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        # initial/previous state
        if init_state is None:
            init_state = T.alloc(0., n_samples, dim)
        # initial/previous memory
        if init_memory is None:
            init_memory = T.alloc(0., n_samples, dim)
    else:
        raise Exception('Only support 3D input')
            
    w_shape = (n_samples, state_below.shape[2])
    u_shape = (n_samples, tparams[get_name(prefix, 'U')].shape[0])
    dropoutmatrix = get_dropout(shapelist = [w_shape,u_shape], 
                                dropoutrate=options['lstm_dropout'])

    # if we have no mask, we assume all the inputs are valid
    if mask == None:
        mask = T.alloc(1., state_below.shape[0], 1)

    # use the slice to calculate all the different gates
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        elif _x.ndim == 2:
            return _x[:, n*dim:(n+1)*dim]
        return _x[n*dim:(n+1)*dim]

    # one time step of the lstm
    def _step(m_, x_, h_, c_):
        if dropoutmatrix[1] is not None:
            drop_h_ = h_ *dropoutmatrix[1]
        else:
            drop_h_ = h_
        preact = T.dot(T.in_train_phase(drop_h_, h_), tparams[get_name(prefix, 'U')])
        preact += x_

        i = inner_activation(_slice(preact, 0, dim))
        f = inner_activation(_slice(preact, 1, dim))
        o = inner_activation(_slice(preact, 2, dim))
        c = activation(_slice(preact, 3, dim))

        c = f * c_ + i * c
        h = o * activation(c)

        return h, c, i, f, o, preact


    if dropoutmatrix[0] is not None:
        drop_state_below = state_below * dropoutmatrix[0]
    else:
        drop_state_below = state_below

    state_below = T.in_train_phase(drop_state_below, state_below)
    state_below = T.dot(state_below, tparams[get_name(prefix, 'W')]) + tparams[get_name(prefix, 'b')]

    rval, updates =    T.scan(  _step,
                                sequences=[mask, state_below],
                                outputs_info=[init_state, init_memory, None, None, None, None],
                                name=get_name(prefix, '_layers'),
                                n_steps=nsteps, profile=False)
    
    return rval

def LSTM(tparams, x, options, params = None, prefix='lstm', nin=None, dim=None,
              init='norm_weight', inner_init='othogo_weight',forget_bias_init='one', 
              inner_activation='sigmoid', trainable = True,
              mask=None,init_memory=None, init_state=None, activation='tanh',
              nner_activation='hard_sigmoid', belonging_Module=None,**kwargs):
    '''
    params > tparams > empty
    if params covers all the weights_keys. use params to update tparams.
    '''
    module_identifier = 'layer_' + prefix
    init_LayerInfo(options, name = module_identifier)
    if not belonging_Module:
        belonging_Module = options['belonging_Module'] if 'belonging_Module' in options else None
    else:
        belonging_Module = belonging_Module

    input_shape = x._keras_shape
    tmp_params = OrderedDict()
    tmp_params = init_lstm(prefix=prefix, nin=nin, dim=dim,init=init, inner_init=inner_init,
                          forget_bias_init=forget_bias_init,trainable = trainable, **kwargs)
    update_or_init_params(tparams, params, tmp_params=tmp_params)
    
    output = lstm_layer( tparams, x, options, prefix= prefix, mask=mask,init_memory=init_memory,
                         init_state=init_state, activation=activation,inner_activation=inner_activation,
                         **kwargs)
    update_father_module(options,belonging_Module, module_identifier)
    return output

# Conditional LSTM layer with Attention
def init_dynamic_lstm_cond(options, params, prefix='lstm_cond', nin=None, dim=None, 
                           init='norm_weight', inner_init='ortho_weight',forget_bias_init='one',
                           ctx_dim=None, proj_ctx_dim=None, trainable=True, **kwargs):
    init = initializations.get(init)
    inner_init = initializations.get(inner_init)
    forget_bias_init = initializations.get(forget_bias_init)
    
    if nin is None:
       nin = options['dim']
    if dim is None:
       dim = options['dim']
    if ctx_dim is None:
       ctx_dim = options['dim']
    if proj_ctx_dim is None:
       proj_ctx_dim = options['dim']
    # input to LSTM, similar to the above, we stack the matricies for compactness, do one
    # dot product, and use the slice function below to get the activations for each "gate"
    #print globals()['norm_weight']
    # input weights
   
    
    W = np.concatenate([ init((nin,dim),symbolic=False),
                         init((nin,dim),symbolic=False),
                         init((nin,dim),symbolic=False),
                         init((nin,dim),symbolic=False)], axis=1)
                
    params[get_name(prefix,'W')] = npwrapper(W, trainable = trainable)
    # for the previous hidden activation
    U = np.concatenate([ inner_init((dim,dim),symbolic=False),
                         inner_init((dim,dim),symbolic=False),
                         inner_init((dim,dim),symbolic=False),
                         inner_init((dim,dim),symbolic=False)], axis=1)
    params[get_name(prefix,'U')] =  npwrapper(U, trainable = trainable)
    b =   np.hstack((np.zeros(dim),
                     forget_bias_init((dim),symbolic=False) + 4,
                     np.zeros(dim),
                     np.zeros(dim)))   
    # bias to LSTM
    params[get_name(prefix,'b')] = npwrapper(b,trainable=trainable)

    # context to LSTM hidden
    Wc2h = init((ctx_dim,dim*4),symbolic=False)
    params[get_name(prefix,'Wc2h')] = npwrapper(Wc2h,trainable=trainable)

    # attention: context -> project context
    W_c2pc = init((ctx_dim,proj_ctx_dim),symbolic=False)
    params[get_name(prefix,'W_c2pc')] = npwrapper(W_c2pc,trainable=trainable)
    # attention: hidden bias
    b_c2pc = initializations.get('zero')((proj_ctx_dim,),symbolic=False).astype(T.floatX)
    params[get_name(prefix,'b_c2pc')] =  npwrapper(b_c2pc,trainable=trainable)
    
    # attention: hidden-->project_ctx
    W_h2pc = init((dim,proj_ctx_dim),symbolic=False)
    params[get_name(prefix,'W_h2pc')] =  npwrapper(W_h2pc,trainable=trainable)
    
    # optional "deep" attention,  context to proj_ctx
    if options['n_layers_att'] > 1:
        for lidx in xrange(1, options['n_layers_att']):
            params[get_name(prefix,'W_c2pc_%d'%lidx)] =  npwrapper(inner_init((proj_ctx_dim,proj_ctx_dim),symbolic=False),trainable=trainable)
            params[get_name(prefix,'b_c2pc_%d'%lidx)] =  npwrapper(np.zeros((proj_ctx_dim,)).astype(T.floatX),trainable=trainable)
 
    if options['attn_type'] == 'dynamic':
        # get_w  parameters for reading operation
       if options['addressing'] == 'softmax':
            params[get_name(prefix,'W_k_read')] =   npwrapper(inner_init((proj_ctx_dim, proj_ctx_dim),symbolic=False),trainable=trainable)
            params[get_name(prefix,'b_k_read')] =   npwrapper(initializations.get('zero')((proj_ctx_dim),symbolic=False),trainable=trainable)
            params[get_name(prefix,'W_address')] =  npwrapper(inner_init((proj_ctx_dim, proj_ctx_dim),symbolic=False),trainable=trainable)

       elif options['addressing'] == 'ntm':
            params[get_name(prefix,'W_k_read')] =  npwrapper(inner_init((proj_ctx_dim, proj_ctx_dim),symbolic=False),trainable=trainable)
            params[get_name(prefix,'b_k_read')] =  npwrapper(initializations.get('zero')((proj_ctx_dim),symbolic=False),trainable=trainable)
            params[get_name(prefix,'W_c_read')] =  npwrapper(inner_init((proj_ctx_dim, 3),symbolic=False),trainable=trainable)
            params[get_name(prefix,'b_c_read')] =  npwrapper(initializations.get('zero')((3,),symbolic=False),trainable=trainable)
            params[get_name(prefix,'W_s_read')] =  npwrapper(inner_init((proj_ctx_dim,  options['shift_range']),symbolic=False),trainable=trainable)
            params[get_name(prefix,'b_s_read')] =  npwrapper(initializations.get('zero')((options['shift_range'],),symbolic=False),trainable=trainable)
    else:
        
        # attention:
        U_att = init((proj_ctx_dim,1),symbolic=False)
        params[get_name(prefix,'U_att')] =  npwrapper(U_att,trainable=trainable)
        c_att = initializations.get('zero')((1,),symbolic=False)
        params[get_name(prefix, 'c_tt')] =  npwrapper(c_att,trainable=trainable)

    if options['selector']:
        # attention: selector
        W_sel = init((dim, 1),symbolic=False)
        params[get_name(prefix, 'W_sel')] = npwrapper(W_sel,trainable=trainable)
        b_sel = np.float32(0.).astype(T.floatX) 
        params[get_name(prefix, 'b_sel')] = npwrapper(b_sel,trainable=trainable)

    return params

def dynamic_lstm_cond_layer(tparams, state_below, options, prefix='dlstm', mask=None,
                            context=None, one_step=False,init_memory=None, init_state=None, 
                            init_alpha = None,rng=None, sampling=True, argmax=False,
                            activation='tanh', inner_activation='sigmoid',**kwargs):
    '''
    Parameters
    ----------
      tparams: contains the ordredDict of symbolic parameters.
      state_below: timestep * batchsize * input_dim
      contex : nsample * annotation * dim
      options: model configuration
    Returns
    -------
    '''
    def get_dropout(shapelist=[None], dropoutrate = 0):
        #if self.seed is None:
        if dropoutrate:
          retain_prob = 1- dropoutrate
          #retain_prob_U = 1- dropoutrate[0]

          W1 = T.binomial(shape= shapelist[0], p = retain_prob, dtype = T.floatX)/retain_prob
          #W2 = T.binomial(shape= shapelist[0], p = retain_prob, dtype = T.floatX)/retain_prob
          #W3 = T.binomial(shape= shapelist[2], p = retain_prob, dtype = T.floatX)/retain_prob
          return [W1]
        else:
          return [None]

    activation = activations.get(activation)
    inner_activation = activations.get(inner_activation)

    assert not T.isnan(context), 'Context must be provided'
    if one_step:
        assert not T.isnan(init_memory), 'previous memory must be provided'
        assert not T.isnan(init_state), 'previous state must be provided'

    
    if T.ndim(state_below) == 3: 
        n_samples = state_below.shape[1]
        if mask is None:
           mask = T.alloc(1., state_below.shape[0], 1)
    else: 
        if one_step == True:
            #means we don't have time dimension, which is the first dimension of 3', we need to pad it here to make
            # the logic cleaner
            n_samples = state_below.shape[0]
            state_below = T.expand_dims(state_below, dim=0)
            if mask is None: 
               mask = T.alloc(1., 1,state_below.shape[0])
        else:
            n_samples = tate_below.shape[1]
            state_below = T.expand_dims(state_below, dim=-1)
            if mask is None: 
               mask = T.alloc(1., 1,state_below.shape[0])
    
    nsteps = state_below.shape[0]
    # initial/previous state
    if init_state is None:
        init_state = T.alloc(0., n_samples, dim)
    # initial/previous memory
    if init_memory is None:
        init_memory = T.alloc(0., n_samples, dim)

    #w_shape   = (n_samples, state_below.shape[-1])
    #att_shape = (1,batchsize, tparams[get_name(prefix,'W_h2pc')].shape[0])
    u_shape   = (n_samples, 3*tparams[get_name(prefix, 'U')].shape[0])
    #ctx_shape = (n_samples, tparams[get_name(prefix, 'Wc')].shape[0])

    # projected x
    # state_below is timesteps*num samples by d in training 
    # this is n * d during sampling
    dropoutmatrix = get_dropout(shapelist = [u_shape], dropoutrate=options['lstm_dropout'])   
    #drop_state_below   =   state_below *dropoutmatrix[0] if dropoutmatrix[0] is not None else state_below
    #tate_below = T.in_train_phase(drop_state_below, state_below)
     
    state_below = T.dot(state_below, tparams[get_name(prefix, 'W')]) + tparams[get_name(prefix, 'b')]
    
    # infer lstm dimension
    dim = tparams[get_name(prefix, 'U')].shape[0]   
    # projected context
    pctx_ = T.dot(context, tparams[get_name(prefix,'W_c2pc')]) + tparams[get_name(prefix, 'b_c2pc')]
    if options['n_layers_att'] > 1:
        for lidx in xrange(1, options['n_layers_att']):
            pctx_ = T.dot(pctx_, tparams[get_name(prefix,'W_c2pc_%d'%lidx)])+tparams[get_name(prefix, 'b_c2pc_%d'%lidx)]
            # note to self: this used to be options['n_layers_att'] - 1, so no extra non-linearity if n_layers_att < 3
            if lidx < options['n_layers_att']:
                pctx_ = T.tanh(pctx_)
    # additional parameters for stochastic hard attention
    if options['attn_type'] == 'stochastic':
        # temperature for softmax
        temperature = options.get("temperature", 1)
        # [see (Section 4.1): Stochastic "Hard" Attention]
        semi_sampling_p = options.get("semi_sampling_p", 0.5)
        temperature_c = T.shared(np.float32(temperature), name='temperature_c')
        h_sampling_mask = T.binomial((1,), p=semi_sampling_p, n=1, dtype=T.floatX, rng= rng).sum()

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _read( w, M):
        return (w[:, :, None]*M).sum(axis=1)

    def _get_content_w( beta, k, M):
        num = beta[:, None] * cosine_distance(M, k)
        return T.softmax(num)

    def _get_location_w(g, s, C, gamma, wc, w_tm1):
        wg = g[:, None] * wc + (1-g[:, None])*w_tm1
        Cs = (C[None, :, :, :] * wg[:, None, None, :]).sum(axis=3)
        wtilda = (Cs * s[:, :, None]).sum(axis=1)
        wout = renorm(wtilda ** gamma[:, None])
        return wout

    def _get_controller_output(h, W_k, b_k, W_c, b_c, W_s, b_s):
        k = T.tanh(T.dot(h, W_k) + b_k)  # + 1e-6
        #k = theano.printing.Print('[Debug] k shape is: ', attrs=("shape",))(k)
        c = T.dot(h, W_c) + b_c
        beta = T.softplus(c[:, 0]) + 1e-4
        g = T.sigmoid(c[:, 1])
        gamma = T.softplus(c[:, 2]) + 1.0001
        s = T.softmax(T.dot(h, W_s) + b_s)
        return k, beta, g, gamma, s

    def _step(m_, x_, h_, c_, a_, as_,ct_, pctx_=None):
        """ Each variable is one time slice of the LSTM
        Only use it if you use wr_tm1, otherwise use a wrapper that does not have wr_tm1

        m_ - (mask), x_- (previous word), h_- (hidden state), c_- (lstm memory),
        a_ - (alpha distribution [eq (5)]), as_- (sample from alpha dist),
        pctx_ (projected context), dp_/dp_att_ (dropout masks)

        m_, x_ are the sequence input.
        it returns:
        rval = [h, c, alpha, alpha_sample, ctx_]
        if options['selector']:
            rval += [sel_]
        if options['attn_type'] == 'dynamic':
            rval += [wr_tm1]
        rval += [pstate_, pctx_, i, f, o, preact, alpha_pre]+pctx_list
        return rval

        """
        # attention computation
        # [described in  equations (4), (5), (6) in
        # section "3.1.2 Decoder: Long Short Term Memory Network]
        pctx_list = [] # used to store pctx_ before activation and multiplication with U.
        pstate_ = T.dot(h_, tparams[get_name(prefix,'W_h2pc')])
        pctx_ = pctx_ + pstate_[:,None,:]
        pctx_list.append(pctx_)
        pctx_ = T.tanh(pctx_)  #pctx_ is no longer pctx_list[0] anymore.


        if options['attn_type'] == 'dynamic':
            # get controller output
            #pstate_ =  h_
            if options['addressing'] == 'ntm':
                W_k_read = tparams[get_name(prefix,'W_k_read')]
                b_k_read = tparams[get_name(prefix,'b_k_read')]
                W_c_read = tparams[get_name(prefix,'W_c_read')]
                b_c_read = tparams[get_name(prefix,'b_c_read')]
                W_s_read = tparams[get_name(prefix,'W_s_read')]
                b_s_read = tparams[get_name(prefix,'b_s_read')]

                k_read, beta_read, g_read, gamma_read, s_read = _get_controller_output(
                    pstate_, W_k_read, b_k_read, W_c_read, b_c_read,
                    W_s_read, b_s_read)
                C = circulant(pctx_.shape[1], options['shift_range'])
                wc_read = _get_content_w(beta_read, k_read, pctx_)

                alpha_pre = wc_read
                alpha_shp = wc_read.shape
                
                wr_tm1 = a_
                alpha   =  _get_location_w(g_read, s_read, C, gamma_read,
                                        wc_read, wr_tm1)
                #ctx_ =  return (w[:, :, None]*M).sum(axis=1)  #_read(wr_t, M_tm1)
                ctx_ = (context * alpha[:,:,None]).sum(axis=1) # current context
                alpha_sample = alpha # you can return something else reasonable here to debug

            elif options['addressing'] == 'softmax':
                W_k_read = tparams[get_name(prefix,'W_k_read')]
                b_k_read = tparams[get_name(prefix,'b_k_read')]
                W_address = tparams[get_name(prefix,'W_address')]

                k = T.tanh(T.dot(pstate_, W_k_read
                ) + b_k_read)  # + 1e-6
                score = (T.dot(pctx_,W_address) * k[:, None, :]).sum(axis=-1) # N * location
                alpha_pre = score
                alpha_shp = alpha_pre.shape
                alpha = T.softmax(score)
                ctx_ = (context * alpha[:,:,None]).sum(axis=1) # current context
                alpha_sample = alpha # you can return something else reasonable here to debug
            elif options['addressing'] == 'cosine':
                pass
        else:
            # attention computation
            # [described in  equations (4), (5), (6) in
            # section "3.1.2 Decoder: Long Short Term Memory Network]           
            alpha = T.dot(pctx_, tparams[get_name(prefix,'U_att')])+tparams[get_name(prefix, 'c_tt')]
            alpha_pre = alpha
            alpha_shp = alpha.shape
            if options['attn_type'] == 'deterministic':
                alpha = T.softmax(alpha.reshape([alpha_shp[0],alpha_shp[1]])) # softmax
                ctx_ = (context * alpha[:,:,None]).sum(1) # current context
                alpha_sample = alpha # you can return something else reasonable here to debug
            elif options['attn_type'] == 'stochastic':
                alpha = T.softmax(temperature_c*alpha.reshape([alpha_shp[0],alpha_shp[1]])) # softmax
                if sampling:
                    alpha_sample = h_sampling_mask * T.multinomial(pvals=alpha,dtype=T.floatX, rng=rng)\
                                   + (1.-h_sampling_mask) * alpha
                else:
                    if argmax:
                        alpha_sample = T.cast(T.eq(T.arange(alpha_shp[1])[None,:],
                                                   T.argmax(alpha,axis=1,keepdims=True)), T.floatX)
                    else:
                        alpha_sample = alpha
                ctx_ = (context * alpha_sample[:,:,None]).sum(1) # current context

        if options['selector']:
            sel_ = T.sigmoid(T.dot(h_, tparams[get_name(prefix, 'W_sel')])+tparams[get_name(prefix,'b_sel')])
            sel_ = sel_.reshape([sel_.shape[0]])
            ctx_ = sel_[:,None] * ctx_


        
        preact = T.dot(h_, tparams[get_name(prefix, 'U')])
        preact += x_
        #preact += T.dot(T.in_train_phase(drop_ctx_, ctx_), tparams[get_name(prefix, 'Wc')])
        preact += T.dot(ctx_, tparams[get_name(prefix, 'Wc2h')])
        
        #applied bayesian LSTM
        cut_preact = _slice(preact, 0, 3*dim)
        drop_cut_preact   =   cut_preact *dropoutmatrix[0] if dropoutmatrix[0] is not None else cut_preact
        cut_preact = T.in_train_phase(drop_cut_preact, cut_preact)

        i = inner_activation(_slice(cut_preact, 0, dim))
        f = inner_activation(_slice(cut_preact, 1, dim))
        o = inner_activation(_slice(cut_preact, 2, dim))
        c = activation(_slice(preact, 3, dim))
        # compute the new memory/hidden state
        # if the mask is 0, just copy the previous state
        c = f * c_ + i * c
        c = m_[:,None] * c + (1. - m_)[:,None] * c_

        h = o * T.tanh(c)
        h = m_[:,None] * h + (1. - m_)[:,None] * h_
        
        
        rval = [h, c, alpha, alpha_sample, ctx_]

        if options['selector']:
            rval += [sel_]

        rval += [pstate_, pctx_, i, f, o, preact, alpha_pre]+pctx_list
        return rval
        
    
    _step0 = _step    
    #when you have an option about what you want to return in outputs_info. Wrapper _step
    #if options['attn_type'] == 'dynamic' and options['addressing'] == 'ntm':
    #    _step0 = _step
    #else:
    #    def f(m_, x_, h_, c_, a_, as_, ct_, pctx_):
    #        return _step(m_, x_, h_, c_, a_, as_, ct_, None, pctx_)
    #    _step0 = f # m_, x_, h_, c_, a_, as_,  ct_,pctx_: _step(m_, x_, h_, c_, a_, as_, ct_, pctx_)

    #if options['attn_type'] == 'dynamic' and options['addressing'] == 'ntm':
    #    if wr_tm1 == None:
    #       wr_tm1= T.alloc(0., n_samples, pctx_.shape[1])  #w_tm1

    if one_step:
        rval = _step0(mask[0], state_below[0], init_state, init_memory, init_alpha,None, None, pctx_)
        # for every rval, we remove one single axis from it. # we also need to be careful if we really changed state
        # and mask out of this function.
        return rval

    else:
        seqs = [mask, state_below]
        outputs_info = [init_state,                               # h
                        init_memory,                              # c
                        T.alloc(0., n_samples, pctx_.shape[1]),   # a_
                        T.alloc(0., n_samples, pctx_.shape[1]),   # as_
                        T.alloc(0., n_samples, context.shape[2])]  # ct_

        
        if options['selector']:
            outputs_info += [None]  
            #why do you want it to be an paramter when you dont use it???
            #outputs_info += [T.alloc(0., n_samples)] 
        #outputs_info with None don't have position in _step parameter list.
        outputs_info += [None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None] + [None] # *options['n_layers_att']
        rval, updates = T.scan( _step0,
                                sequences=seqs,
                                outputs_info=outputs_info,
                                non_sequences=[pctx_],
                                name=get_name(prefix, '_layers'),
                                n_steps=nsteps, profile=False)           
        return rval, updates
def cond_LSTM(tparams, x, options, params = None,prefix='lstm_cond', nin=None, 
              dim=None, init='norm_weight', inner_init='ortho_weight',
              forget_bias_init='one',ctx_dim=None, proj_ctx_dim=None, 
              trainable=True,  mask=None, context=None, one_step=False,
              init_memory=None, init_state=None, rng=None, sampling=True,
              init_alpha = None,argmax=False,activation='tanh', 
              inner_activation='sigmoid',belonging_Module=None,**kwargs):
    '''
    params > tparams > empty
    if params covers all the weights_keys. use params to update tparams.
    '''
    module_identifier = 'layer_' + prefix
    init_LayerInfo(options, name = module_identifier)
    if not belonging_Module:
        belonging_Module = options['belonging_Module'] if 'belonging_Module' in options else None
    else:
        belonging_Module = belonging_Module
    input_shape = x._keras_shape
    tmp_params = OrderedDict()
    tmp_params = init_lstm(prefix= prefix, nin=nin, dim=dim, init=init, 
                           inner_init=inner_init,forget_bias_init=forget_bias_init,
                           ctx_dim=ctx_dim, proj_ctx_dim=proj_ctx_dim,  
                           trainable=trainable,  **kwargs)
    update_or_init_params(tparams, params, tmp_params=tmp_params)
    
    options[module_identifier].weights_name.extend(tmp_params.keys())
    options[module_identifier].trainable_weights.extend([tparams[k] for k in tmp_params.keys()])
        
    output = lstm_layer(tparams, x, options, prefix= prefix, mask=mask,
                        context=context, one_step=one_step,init_memory=init_memory, 
                        init_state=init_state, rng=rng, sampling=sampling,
                        init_alpha = init_alpha, argmax=argmax,
                        activation=activation, inner_activation=inner_activation,**kwargs)
    
    update_father_module(options,belonging_Module, module_identifier)
    return output