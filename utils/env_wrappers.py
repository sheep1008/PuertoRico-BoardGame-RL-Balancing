import numpy as np
from gymnasium import spaces

def flatten_dict_observation(obs_dict, space):
    """
    Recursively flatten a dictionary or tuple observation space into a 1D numpy array.
    This assumes that the underlying spaces are Box, Discrete, or MultiDiscrete,
    and the values in obs_dict are numpy arrays or scalars.
    """
    flat_obs = []
    
    if isinstance(space, spaces.Dict):
        # Ensure we always iterate in a consistent order (sorted keys)
        for key in sorted(space.spaces.keys()):
            flat_obs.append(flatten_dict_observation(obs_dict[key], space[key]))
    elif isinstance(space, spaces.Tuple):
        for i in range(len(space.spaces)):
            flat_obs.append(flatten_dict_observation(obs_dict[i], space[i]))
    elif isinstance(space, spaces.Discrete):
        # Even though representation is scalar, we will flatten it as a 1D array of size 1
        flat_obs.append(np.array([obs_dict], dtype=np.float32))
    elif isinstance(space, (spaces.Box, spaces.MultiDiscrete, spaces.MultiBinary)):
        flat_obs.append(np.array(obs_dict, dtype=np.float32).flatten())
    else:
        raise NotImplementedError(f"Unsupported space type for flattening: {type(space)}")
        
    return np.concatenate(flat_obs)

def get_flattened_obs_dim(space):
    """
    Calculates the total 1D dimension of a deeply nested Dict/Tuple/Discrete/MultiDiscrete space.
    """
    if isinstance(space, spaces.Dict):
        return sum(get_flattened_obs_dim(space.spaces[k]) for k in sorted(space.spaces.keys()))
    elif isinstance(space, spaces.Tuple):
        return sum(get_flattened_obs_dim(s) for s in space.spaces)
    elif isinstance(space, spaces.Discrete):
        return 1
    elif isinstance(space, spaces.MultiDiscrete):
        return sum(np.ones_like(space.nvec).flatten())
    elif isinstance(space, spaces.MultiBinary):
        return space.n
    elif isinstance(space, spaces.Box):
        return np.prod(space.shape) if space.shape else 1
    else:
        raise NotImplementedError(f"Unsupported space type for dim calculation: {type(space)}")
