from numpy import *
import numpy as np

from theano.scan_module import scan_utils
import logging
from collections import OrderedDict
from backend.scan_utils import *
_logger = logging.getLogger('theano.scan_module.scan')
_FLOATX = 'float32'

def addAttribute(x, attr=None, value=None):
    return x

def relu(x):
    return x * (x > 0)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
  
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def RandomStreams(seed = 1234):
    a = np.random
    a.seed(seed)
    return a

def normal(shape, mean=0.0, std=1.0, dtype=_FLOATX, seed=None,rng = None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = np.random.seed(seed=seed)
    return rng.normal(avg=mean, std=std, size=shape).astype(dtype)


def uniform(shape, low=0.0, high=1.0, dtype=_FLOATX, rng=None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    return rng.uniform(low=low, high=high, size = shape).astype(dtype)


def binomial(shape, p=0.0, n =1, dtype=_FLOATX, rng=None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    return rng.binomial(n=n, p=p,size= shape).astype(dtype)
    

def multinomial(shape, pvals=0.0, n =1, dtype=_FLOATX, rng=None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    return rng.binomial(size= shape,n=n, p=pvals).astype(dtype)
    


def matrix(name='x', dtype='float32', shape = None):
    x= np.zeros(shape).astype(dtype)
    return x
        
def tensor3(name='x', dtype='float32', shape = None):
    x= np.zeros(shape).astype(dtype)
    return x
def tensor4(name='x', dtype='float32', shape = None):
    x= np.zeros(shape).astype(dtype)
    return x

def switch(cond, a, b):
    if cond:
        return a
    else:
        return b

def set_subtensor(dest, source):
    dest = source
    return dest
    
def shared(value, name=None, strict=False, allow_downcast=None, **kwargs):
    return value
    
def alloc(value, *shape):
    a = zeros(tuple(shape)) + value
    return a
def unbroadcast(x, *axes):
	return x
def addbroadcast(x, *axes):
	return x

def shape_padleft(t, n_ones=1):
    pattern = [1] * n_ones + [t.shape[i] for i in xrange(t.ndim)]
    return np.reshape(t, pattern)
def shape_padright(t, n_ones=1):
    pattern =  [t.shape[i] for i in xrange(t.ndim)] + [1] * n_ones
    return np.reshape(t, pattern)
def shape_padaxis(t, axis):
    """Reshape `t` by inserting 1 at the dimension `axis`.

    Example
    -------
    >>> tensor = theano.tensor.tensor3()
    >>> theano.tensor.shape_padaxis(tensor, axis=0)
    DimShuffle{x,0,1,2}.0
    >>> theano.tensor.shape_padaxis(tensor, axis=1)
    DimShuffle{0,x,1,2}.0
    >>> theano.tensor.shape_padaxis(tensor, axis=3)
    DimShuffle{0,1,2,x}.0
    >>> theano.tensor.shape_padaxis(tensor, axis=-1)
    DimShuffle{0,1,2,x}.0

    See Also
    --------
    shape_padleft
    shape_padright
    Dimshuffle

    """

    ndim = t.ndim + 1
    if not -ndim <= axis < ndim:
        msg = 'axis {0} is out of bounds [-{1}, {1})'.format(axis, ndim)
        raise IndexError(msg)
    if axis < 0:
        axis += ndim

    pattern = [t.shape[i] for i in xrange(t.ndim)]
    pattern.insert(axis, 1)
    return np.reshape(t,pattern)

def scan(fn,
         sequences=None,
         outputs_info=None,
         non_sequences=None,
         n_steps=None,
         truncate_gradient=-1,
         go_backwards=False,
         mode=None,
         name=None,
         profile=False,
         allow_gc=None,
         strict=False):
    def wrap_into_list(x):
        """
        Wrap the input into a list if it is not already a list.

        """
        if x is None:
            return []
        elif not isinstance(x, (list, tuple)):
            return [x]
        else:
            return list(x)

    seqs = wrap_into_list(sequences)
    outs_info = wrap_into_list(outputs_info)

    # Make sure we get rid of numpy arrays or ints or anything like that
    # passed as inputs to scan
    non_seqs = wrap_into_list(non_sequences)

    # If we provided a known number of steps ( before compilation)
    # and if that number is 1 or -1, then we can skip the Scan Op,
    # and just apply the inner function once
    # To do that we check here to see the nature of n_steps
    n_fixed_steps = None

    if isinstance(n_steps, (float, int)):
        n_fixed_steps = int(n_steps)
    else:
        try:
            n_fixed_steps = int(n_steps)
        except ValueError(' n_steps must be an int. dtype provided '
                         'is %s' % n_steps.dtype):
            n_fixed_steps = None

    # Check n_steps is an int
    if (hasattr(n_steps, 'dtype') and
        str(n_steps.dtype)[:3] not in ('uin', 'int')):
        raise ValueError(' n_steps must be an int. dtype provided '
                         'is %s' % n_steps.dtype)

    # compute number of sequences and number of outputs
    n_seqs = len(seqs)
    n_outs = len(outs_info)

    return_steps = OrderedDict()
    # wrap sequences in a dictionary if they are not already dictionaries
    for i in xrange(n_seqs):
        if not isinstance(seqs[i], dict):
            seqs[i] = OrderedDict([('input', seqs[i]), ('taps', [0])])
        elif seqs[i].get('taps', None) is not None:
            seqs[i]['taps'] = wrap_into_list(seqs[i]['taps'])
        elif seqs[i].get('taps', None) is None:
            # seqs dictionary does not have the ``taps`` key
            seqs[i]['taps'] = [0]

    # wrap outputs info in a dictionary if they are not already in one
    for i in xrange(n_outs):
        if outs_info[i] is not None:
            if isinstance(outs_info[i], dict):
                # DEPRECATED :
                if outs_info[i].get('return_steps', None) is not None:
                    raise ValueError(
                            "Using `return_steps` has been deprecated. "
                            "Simply select the entries you need using a "
                            "subtensor. Scan will optimize memory "
                            "consumption, so do not worry about that.")
                # END

            if not isinstance(outs_info[i], dict):
                # by default any output has a tap value of -1
                outs_info[i] = OrderedDict([('initial', outs_info[i]), ('taps', [-1])])
            elif (outs_info[i].get('initial', None) is None and
                    outs_info[i].get('taps', None) is not None):
                # ^ no initial state but taps provided
                raise ValueError(('If you are using slices of an output '
                                  'you need to provide a initial state '
                                  'for it'), outs_info[i])
            elif (outs_info[i].get('initial', None) is not None and
                  outs_info[i].get('taps', None) is None):
                # ^ initial state but taps not provided
                if 'taps' in outs_info[i]:
                    # ^ explicitly provided a None for taps
                    _logger.warning('Output %s ( index %d) has a initial '
                            'state but taps is explicitly set to None ',
                             getattr(outs_info[i]['initial'], 'name', 'None'),
                             i)
                outs_info[i]['taps'] = [-1]
        else:
            # if a None is provided as the output info we replace it
            # with an empty OrdereDict() to simplify handling
            outs_info[i] = OrderedDict()

    ##
    # Step 2. Generate inputs and outputs of the inner functions
    # for compiling a dummy function (Iteration #1)
    ##

    # create theano inputs for the recursive function
    # note : this is a first batch of possible inputs that will
    #        be compiled in a dummy function; we used this dummy
    #        function to detect shared variables and their updates
    #        and to construct a new and complete list of inputs and
    #        outputs

    n_seqs = 0
    scan_seqs = []     # Variables passed as inputs to the scan op
    inner_seqs = []    # Variables passed as inputs to the inner function
    inner_slices = []  # Actual slices if scan is removed from the picture
    # go through sequences picking up time slices as needed
    for i, seq in enumerate(seqs):
        # Note that you can have something like no taps for
        # a sequence, though is highly unlikely in practice
        if 'taps' in seq:
            # go through the indicated slice
            mintap = np.min(seq['taps'])
            maxtap = np.max(seq['taps'])
            for k in seq['taps']:
                # create one slice of the input
                # Later on, if we decide not to use scan because we are
                # going for just one step, it makes things easier if we
                # compute the correct outputs here. This way we can use
                # the output of the lambda expression directly to replace
                # the output of scan.

                # If not we need to use copies, that will be replaced at
                # each frame by the corresponding slice
                actual_slice = seq['input'][k - mintap]
                _seq_val = seq['input'] #tensor.as_tensor_variable(seq['input'])
                _seq_val_slice = _seq_val[k - mintap]
                nw_slice = _seq_val_slice

                # Add names to slices for debugging and pretty printing ..
                # that is if the input already has a name
#                if getattr(seq['input'], 'name', None) is not None:
#                    if k > 0:
#                        nw_name = seq['input'].name + '[t+%d]' % k
#                    elif k == 0:
#                        nw_name = seq['input'].name + '[t]'
#                    else:
#                        nw_name = seq['input'].name + '[t%d]' % k
#                    nw_slice.name = nw_name

                # We cut the sequence such that seq[i] to correspond to
                # seq[i-k]. For the purposes of cutting the sequences, we
                # need to pretend tap 0 is used to avoid cutting the sequences
                # too long if the taps are all lower or all higher than 0.
                maxtap_proxy = max(maxtap, 0)
                mintap_proxy = min(mintap, 0)
                start = (k - mintap_proxy)
                if k == maxtap_proxy:
                    nw_seq = seq['input'][start:]
                else:
                    end = -(maxtap_proxy - k)
                    nw_seq = seq['input'][start:end]

                if go_backwards:
                    nw_seq = nw_seq[::-1]

                scan_seqs.append(nw_seq)
                inner_seqs.append(nw_slice)
                inner_slices.append(actual_slice)
                n_seqs += 1

    # Since we've added all sequences now we need to level them up based on
    # n_steps or their different shapes
    lengths_vec = []
    for seq in scan_seqs:
        lengths_vec.append(seq.shape[0])

    if not scan_utils.isNaN_or_Inf_or_None(n_steps):
        # ^ N_steps should also be considered
        lengths_vec.append(n_steps)

    if len(lengths_vec) == 0:
        # ^ No information about the number of steps
        raise ValueError('No information about the number of steps '
                         'provided. Either provide a value for '
                         'n_steps argument of scan or provide an input '
                         'sequence')

    # If the user has provided the number of steps, do that regardless ( and
    # raise an error if the sequences are not long enough )
    if scan_utils.isNaN_or_Inf_or_None(n_steps):
        actual_n_steps = lengths_vec[0]
        for contestant in lengths_vec[1:]:
            actual_n_steps = np.minimum(actual_n_steps, contestant)
    else:
        actual_n_steps = n_steps

    # Add names -- it helps a lot when debugging

    for (nw_seq, seq) in zip(scan_seqs, seqs):
        if getattr(seq['input'], 'name', None) is not None:
            nw_seq.name = seq['input'].name + '[%d:]' % k

    scan_seqs = [seq[:actual_n_steps] for seq in scan_seqs]
    # Conventions :
    #   mit_mot = multiple input taps, multiple output taps ( only provided
    #             by the gradient function )
    #   mit_sot = multiple input taps, single output tap (t + 0)
    #   sit_sot = single input tap, single output tap (t + 0)
    #   nit_sot = no input tap, single output tap (t + 0)

    # MIT_MOT -- not provided by the user only by the grad function
    n_mit_mot = 0
    n_mit_mot_outs = 0
    mit_mot_scan_inputs = []
    mit_mot_inner_inputs = []
    mit_mot_inner_outputs = []
    mit_mot_out_slices = []
    mit_mot_rightOrder = []

    # SIT_SOT -- provided by the user
    n_mit_sot = 0
    mit_sot_scan_inputs = []
    mit_sot_inner_inputs = []
    mit_sot_inner_slices = []
    mit_sot_inner_outputs = []
    mit_sot_return_steps = OrderedDict()
    mit_sot_tap_array = []
    mit_sot_rightOrder = []

    n_sit_sot = 0
    sit_sot_scan_inputs = []
    sit_sot_inner_inputs = []
    sit_sot_inner_slices = []
    sit_sot_inner_outputs = []
    sit_sot_return_steps = OrderedDict()
    sit_sot_rightOrder = []

    # go through outputs picking up time slices as needed
    for i, init_out in enumerate(outs_info):
        # Note that our convention dictates that if an output uses
        # just the previous time step, as a initial state we will only
        # provide a tensor of the same dimension as one time step; This
        # makes code much cleaner for those who do not use taps. Otherwise
        # they would always had to shape_padleft the initial state ..
        # which is ugly
        if init_out.get('taps', None) == [-1]:

            actual_arg = init_out['initial']
            #if not isinstance(actual_arg, tensor.Variable):
            #    actual_arg = tensor.as_tensor_variable(actual_arg)
            arg = safe_new(actual_arg)


            if getattr(init_out['initial'], 'name', None) is not None:
                arg.name = init_out['initial'].name + '[t-1]'

            # We need now to allocate space for storing the output and copy
            # the initial state over. We do this using the expand function
            # defined in scan utils
            sit_sot_scan_inputs.append(
                    expand_empty(
                    unbroadcast(
                        shape_padleft(actual_arg), 0),
                    actual_n_steps
                ))

            sit_sot_inner_slices.append(actual_arg)
            if i in return_steps:
                sit_sot_return_steps[n_sit_sot] = return_steps[i]
            sit_sot_inner_inputs.append(arg)
            sit_sot_rightOrder.append(i)
            n_sit_sot += 1

        elif init_out.get('taps', None):

            if np.any(np.array(init_out.get('taps', [])) > 0):
                # Make sure we do not have requests for future values of a
                # sequence we can not provide such values
                raise ValueError('Can not use future taps of outputs',
                                    init_out)
            # go through the taps
            mintap = abs(np.min(init_out['taps']))
            mit_sot_tap_array.append(init_out['taps'])
            idx_offset = abs(np.min(init_out['taps']))
            # Sequence
            mit_sot_scan_inputs.append(
                expand_empty(init_out['initial'][:mintap],
                                        actual_n_steps))

            if i in return_steps:
                mit_sot_return_steps[n_mit_sot] = return_steps[i]
            mit_sot_rightOrder.append(i)
            n_mit_sot += 1
            for k in init_out['taps']:
                # create a new slice
                actual_nw_slice = init_out['initial'][k + mintap]
                _init_out_var = init_out['initial']
                _init_out_var_slice = _init_out_var[k + mintap]
                nw_slice = type(_init_out_var_slice)

                # give it a name or debugging and pretty printing
                if getattr(init_out['initial'], 'name', None) is not None:
                    if k > 0:
                        nw_slice.name = (init_out['initial'].name +
                                            '[t+%d]' % k)
                    elif k == 0:
                        nw_slice.name = init_out['initial'].name + '[t]'
                    else:
                        nw_slice.name = (init_out['initial'].name +
                                            '[t%d]' % k)
                mit_sot_inner_inputs.append(nw_slice)
                mit_sot_inner_slices.append(actual_nw_slice)
        # NOTE: there is another case, in which we do not want to provide
        #      any previous value of the output to the inner function (i.e.
        #      a map); in that case we do not have to do anything ..

    # Re-order args
    max_mit_sot = np.max([-1] + mit_sot_rightOrder) + 1
    max_sit_sot = np.max([-1] + sit_sot_rightOrder) + 1
    n_elems = np.max([max_mit_sot, max_sit_sot])
    _ordered_args = [[] for x in xrange(n_elems)]
    offset = 0
    for idx in xrange(n_mit_sot):
        n_inputs = len(mit_sot_tap_array[idx])
        if n_fixed_steps in [1, -1]:
            _ordered_args[mit_sot_rightOrder[idx]] = \
                            mit_sot_inner_slices[offset:offset + n_inputs]
        else:
            _ordered_args[mit_sot_rightOrder[idx]] = \
                            mit_sot_inner_inputs[offset:offset + n_inputs]
        offset += n_inputs

    for idx in xrange(n_sit_sot):
        if n_fixed_steps in [1, -1]:
            _ordered_args[sit_sot_rightOrder[idx]] = \
                                        [sit_sot_inner_slices[idx]]
        else:
            _ordered_args[sit_sot_rightOrder[idx]] = \
                                        [sit_sot_inner_inputs[idx]]

    ordered_args = []
    for ls in _ordered_args:
        ordered_args += ls
    if n_fixed_steps in [1, -1]:
        args = (inner_slices +
                ordered_args +
                non_seqs)

    else:
        args = (inner_seqs +
                ordered_args +
                non_seqs)

    # add only the non-shared variables and non-constants to the arguments of
    # the dummy function [ a function should not get shared variables or
    # constants as input ]
    #dummy_args = [arg for arg in args
    #              if (not isinstance(arg, SharedVariable) and
    #                  not isinstance(arg, tensor.Constant))]
    dummy_args = [arg for arg in args]
    # when we apply the lambda expression we get a mixture of update rules
    # and outputs that needs to be separated
    
    condition, outputs, updates = scan_utils.get_updates_and_outputs(fn(*args))
    if condition is not None:
        as_while = True
    else:
        as_while = False

    return outputs, updates
