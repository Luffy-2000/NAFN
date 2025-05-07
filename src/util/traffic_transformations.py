import numpy as np


def permutation(x, a=1, random_state=None):
    """
    Segment the time series and reorder the segments
    Details: Define N=2+arg maxi{ai ≤ α} where ai ∈{0.15, 0.45, 0.75, 0.9}, a sample
    n∼U[2, N] and split the range [0:T-1] into n segments of random length. Compose a
    new sample x' by concatenating  x(:,t) from a random order of segments.
    
    Parameters:
    - x: Tensor of shape [batch_size, channel, n, m], where batch_size is the batch size
         and n is the sequence length, m is the number of features
    - a: parameter to control segment size, default is 1
    - random_state: optional random seed to ensure reproducibility

    Returns:
    - out: Tensor of shape [batch_size, channel, n, m] with segments reordered
    """
    first_iat = x[:, 0, 1]

    weights = [0.15, 0.45, 0.75, 0.9]
    N = 2 + np.argmax([w for w in weights if w <= a])
    x, pad = _split_on_pad(x)
    
    n = _execute_revert_random_state(
            np.random.randint, dict(low=2, high=N+1, size=None), random_state)
       
    temp = np.array_split(x, n, axis=1) # Split into n segments of random length
    temp = [t for t in temp if t.size > 0]
    
    np.random.shuffle(temp)
    
    out = np.concatenate(temp, axis=1)

    if first_iat != out[:, 0, 1]:
        # Set first IAT to 0, set swapped first iat to mean IAT
        i = np.where(out[0, :, 1] == first_iat)
        out[:, i, 1] = np.mean([iat for iat in out[0, :, 1] if iat != first_iat])
        out[:, 0, 1] = first_iat
        
    if pad is not None:
        out = np.concatenate([out, pad], axis=1)
    return out


def pkt_translating(x, a=1, random_state=None):
    """
    Move a segment to left or the right
    Details: Define N =1+arg maxi{ai ≤ α} where ai ∈ {0.15, 0.3, 0.5, 0.8} and sample
    n ∼ U[1, N]. Then, sample a direction b ∈ {left, right} and a starting point t∼U[0, T]:
    If b = left, left shift each feature values n times starting from t else right shift each
    feature values n times starting from t.(replace shifted values with the single value x(d,t))
    """
    first_iat = x[:, 0, 1]
    weights = [0.15, 0.3, 0.5, 0.8]
    N = 1 + np.argmax([w for w in weights if w <= a])
    inf, _ = _split_on_pad(x)
    _, T, *_ = inf.shape
    
    n = _execute_revert_random_state(
        np.random.randint, dict(low=1, high=N+1, size=None), random_state) # Shift amount 

    b = _execute_revert_random_state(
        np.random.randint, dict(low=0, high=2, size=None), random_state) # 0->L; 1->R
    n = -n if b==0 else n
    
    t = _execute_revert_random_state(
        np.random.randint, dict(low=0, high=T, size=None), random_state) # Starting point
    
    out = _shift(
        x, shift_value=n, 
        fill_value=[0, 0, 0.5, 0, 0, 0] if b==0 else x[0, t, :], 
        axis=1, start=t
    )
    
    if np.all(out[0, :, 0] == 0):
        # The shift created an empty biflow, restore the original input
        return x
        
    if first_iat != out[:, 0, 1]:
        # Set first IAT to 0
        out[:, 0, 1] = first_iat
        
    return out


def wrap(x, a=1, fill_value=[0, 0, 0.5, 0, 0, 0]):
    """
    Mixing interpolation, drop and no change
    Details: Compose a new sample x′ by manipulating each x(:,t) based on three options with
    probabilities P[interpolate] = P[discard] = 0.5α and P[nochange] = 1−α.
    If “nochange” then keep x(:,t); if “interpolate” then keep x(:,t) and x(:,t) = 0.5(x(:,t) + x(:,t+1));
    if “nochange” then do nothing.
    Stop when |x'| = (packet num per features) or apply tail padding (if needed).
    """
    first_iat = x[:, 0, 1]
    out = np.full_like(x, fill_value)
    
    x, _ = _split_on_pad(x)
    _, T, *_ = x.shape

    functions = [_interpolate, _drop, _no_change]
    probabilities = [0.5 * a, 0.5 * a, 1 - a]

    for t in range(T):
        func_idx = np.random.choice(len(functions), p=probabilities)
        # Apply the selected function
        out[:, t, :] = functions[func_idx](x=x, i=t, fill_value=fill_value)

    if first_iat != out[:, 0, 1]:
        # Set first IAT to 0
        out[:, 0, 1] = first_iat

    return out


def _split_on_pad(x):
    indices_pad = np.where(x[0, :, 0] == 0)[0]
    
    if indices_pad.size == 0:
        # No pad 
        bfl = x
        pad = None
    else:
        split_index = indices_pad[0]
        bfl = x[:, :split_index, :]
        pad = x[:, split_index:, :]

    return bfl, pad    


def _shift(x, shift_value, fill_value=np.nan, start=0, axis=0):
    result = np.full_like(x, fill_value)  # Initialize with fill_value
    n = x.shape[axis]

    # Copy the data before 'start' along the axis
    slices_before_start = [slice(None)] * x.ndim
    slices_before_start[axis] = slice(None, start)
    result[tuple(slices_before_start)] = x[tuple(slices_before_start)]

    if shift_value > 0:
        # Source indices: from 'start' to 'n - shift_value'
        slices_src = [slice(None)] * x.ndim
        slices_src[axis] = slice(start, n - shift_value)

        # Destination indices: from 'start + shift_value' to 'n'
        slices_dest = [slice(None)] * x.ndim
        slices_dest[axis] = slice(start + shift_value, n)

        result[tuple(slices_dest)] = x[tuple(slices_src)]
    elif shift_value < 0:
        # Source indices: from 'start - shift_value' to 'n'
        slices_src = [slice(None)] * x.ndim
        slices_src[axis] = slice(start - shift_value, n)

        # Destination indices: from 'start' to 'n + shift_value'
        slices_dest = [slice(None)] * x.ndim
        slices_dest[axis] = slice(start, n + shift_value)

        result[tuple(slices_dest)] = x[tuple(slices_src)]
    else:
        # No shift, copy the rest of the data from 'start' onwards
        slices_rest = [slice(None)] * x.ndim
        slices_rest[axis] = slice(start, None)
        result[tuple(slices_rest)] = x[tuple(slices_rest)]

    return result


def _drop(x, i, fill_value, **kwargs):
    if x.shape[1] == 1 or i==0:
        return x[:, i, :]
    return np.full((x.shape[0], 1, x.shape[2]), fill_value)


def _interpolate(x, i, **kwargs):
    if i+1 >= x.shape[1]:
        return x[:, i, :]

    interpolated = (x[:, i, :] + x[:, i+1, :]) / 2
    interpolated[:, 2] = x[:, i, 2]
    return np.expand_dims(interpolated, axis=0)


def _no_change(x, i, **kwargs):
    return x[:, i, :]


def _execute_revert_random_state(fn, fn_kwargs=None, new_random_state=None):
    """
    Execute fn(**fn_kwargs) without impacting the external random_state behavior.
    """
    old_random_state = np.random.get_state()
    np.random.seed(new_random_state)
    ret = fn(**fn_kwargs)
    np.random.set_state(old_random_state)
    return ret