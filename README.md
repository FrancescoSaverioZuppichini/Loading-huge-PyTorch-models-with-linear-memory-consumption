# Loading huge PyTorch models with linear memory consumption

Hello There!

Today we will see how to load a Pytorch model with linear memory consumption. Loading a model takes 2x memory space. Let's see why:

First, we need a model:


```python
import torch
from torch import nn

class BoringModel(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(2, 10)
        self.stages = nn.Sequential(
             nn.Linear(10, 10),
             nn.Linear(10, 10)
        )
        self.out_proj = nn.Linear(10, 2)
        
```

Upon creation, the model takes `1x` memory, where `x` is its size


```python
model = BoringModel()
# model is now in memory
```

At some point, we are going to store our model to disk to use it later


```python
torch.save(model.state_dict(), "./checkpoint.pt")
# our models is now stored on disk
```

Later on, we may need to use our stored model.


```python
# we need to redefine the model
model = BoringModel()
# 1x memory used
state_dict = torch.load("./checkpoint.pt")
# 2x memory used -> both model and state_dict are in memory!!!
model.load_state_dict(state_dict)
# 1x memory used
```




    <All keys matched successfully>



See? We need `2x` memory to load our stored weight. This is problematic if we have a huge model, since we need two times free RAM. For example, assuming we have 16GB of ram and our model uses 10GB. To load it we need 20GB, **we need to change our strategy**.

Recently, PyTorch introduced the `meta` device. When you put a tensor to the `meta` device, only its metadata (e.g. shape) are stored, and its values are tossed away. Thus, no space is used.


```python
x = torch.tensor([1])
x
```




    tensor([1])




```python
x.to(torch.device("meta"))
```




    tensor(..., device='meta', size=(1,), dtype=torch.int64)



We can leverage this to load our model with `1x` memory consumption by:

- define our model -> `1x` memory
- place it in the `meta` device -> `1x` memory
- load our `state_dict` -> `1x` memory
- replace all empty parameters of our model with the values inside the `state_dict` -> `1x` memory

Sounds easy, but we first need to figure out how to replace all model's parameters with the original ones from a loaded `state_dict`. Let's create the `load_state_dict_with_low_memory` function.


```python
from typing import Dict

def load_state_dict_with_low_memory(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    # free up memory by placing the model in the `meta` device
    model.to(torch.device("meta"))
    # we need to associate each key in state_dict to a submodule
    # then, iteratively, re-creat all submodules' parameters with the values in `state_dict`
    pass
```


```python
load_state_dict_with_low_memory(model, {})

model.state_dict()
```




    OrderedDict([('in_proj.weight', tensor(..., device='meta', size=(10, 2))),
                 ('in_proj.bias', tensor(..., device='meta', size=(10,))),
                 ('stages.0.weight', tensor(..., device='meta', size=(10, 10))),
                 ('stages.0.bias', tensor(..., device='meta', size=(10,))),
                 ('stages.1.weight', tensor(..., device='meta', size=(10, 10))),
                 ('stages.1.bias', tensor(..., device='meta', size=(10,))),
                 ('out_proj.weight', tensor(..., device='meta', size=(2, 10))),
                 ('out_proj.bias', tensor(..., device='meta', size=(2,)))])



The model is empty now.

Now we have to figure out in which submodule of `model` each parameter from `state_dict` has to go. One way to do it is to create a dictionary with `[key_in_state_dict] -> [submodule_in_module]`. 

 So we know where we have to place the values from the loaded `state_dict`. Remember, as soon as the model is placed inside the `meta` device, all its weights are tossed away.


```python
from typing import Dict

def get_keys_to_submodule(model: nn.Module) -> Dict[str, nn.Module]:
    keys_to_submodule = {}
    # iterate all submodules
    for submodule_name, submodule in model.named_modules():
        # iterate all paramters in each submobule
        for param_name, param in submodule.named_parameters():
            # param_name is organized as <name>.<subname>.<subsubname> ...
            # the more we go deep in the model, the less "subname"s we have
            splitted_param_name = param_name.split('.')
            # if we have only one subname, then it means that we reach a "leaf" submodule, 
            # we cannot go inside it anymore. This is the actual parameter
            is_leaf_param = len(splitted_param_name) == 1
            if is_leaf_param:
                # we recreate the correct key
                key = f"{submodule_name}.{param_name}"
                # we associate this key with this submodule
                keys_to_submodule[key] = submodule
                
    return keys_to_submodule
```


```python
get_keys_to_submodule(model)
```




    {'in_proj.weight': Linear(in_features=2, out_features=10, bias=True),
     'in_proj.bias': Linear(in_features=2, out_features=10, bias=True),
     'stages.0.weight': Linear(in_features=10, out_features=10, bias=True),
     'stages.0.bias': Linear(in_features=10, out_features=10, bias=True),
     'stages.1.weight': Linear(in_features=10, out_features=10, bias=True),
     'stages.1.bias': Linear(in_features=10, out_features=10, bias=True),
     'out_proj.weight': Linear(in_features=10, out_features=2, bias=True),
     'out_proj.bias': Linear(in_features=10, out_features=2, bias=True)}



Cool, now we have a way to know which key goes with which submodule of `model`. Let's go back to our `load_state_dict_with_low_memory` function and materialize each submodules parameter using the correct value from `state_dict`


```python
def load_state_dict_with_low_memory(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    # free up memory by placing the model in the `meta` device
    model.to(torch.device("meta"))
    keys_to_submodule = get_keys_to_submodule(model)
    for key, submodule in keys_to_submodule.items():
        # get the valye from the state_dict
        val = state_dict[key]
        # we need to substitute the parameter inside submodule, 
        # remember key is composed of <name>.<subname>.<subsubname>
        # the actual submodule's parameter is stored inside the 
        # last subname. If key is `in_proj.weight`, the correct field if `weight`
        param_name = key.split('.')[-1]
        param_dtype = getattr(submodule, param_name).dtype
        val = val.to(param_dtype)
        # create a new parameter
        new_val = torch.nn.Parameter(val)
        setattr(submodule, param_name, new_val)

```


```python
model.state_dict()
```




    OrderedDict([('in_proj.weight', tensor(..., device='meta', size=(10, 2))),
                 ('in_proj.bias', tensor(..., device='meta', size=(10,))),
                 ('stages.0.weight', tensor(..., device='meta', size=(10, 10))),
                 ('stages.0.bias', tensor(..., device='meta', size=(10,))),
                 ('stages.1.weight', tensor(..., device='meta', size=(10, 10))),
                 ('stages.1.bias', tensor(..., device='meta', size=(10,))),
                 ('out_proj.weight', tensor(..., device='meta', size=(2, 10))),
                 ('out_proj.bias', tensor(..., device='meta', size=(2,)))])




```python
load_state_dict_with_low_memory(model, torch.load("checkpoint.pt"))
```


```python
model.state_dict()
```




    OrderedDict([('in_proj.weight',
                  tensor([[-0.1547, -0.0930],
                          [ 0.1150,  0.2121],
                          [-0.5649, -0.0148],
                          [-0.6554, -0.3978],
                          [ 0.3380, -0.3748],
                          [ 0.6122, -0.6004],
                          [ 0.0220, -0.6723],
                          [ 0.6127,  0.7000],
                          [-0.6631,  0.6500],
                          [-0.4773, -0.4624]])),
                 ('in_proj.bias',
                  tensor([ 0.4023, -0.3971, -0.5358, -0.2197,  0.2122, -0.3990, -0.0342, -0.2672,
                           0.3603,  0.0259])),
                 ('stages.0.weight',
                  tensor([[ 0.2900, -0.1940, -0.0990,  0.2388, -0.1067,  0.0658,  0.0420,  0.2632,
                            0.0636, -0.1373],
                          [ 0.0044,  0.2602,  0.0139,  0.2579, -0.0645, -0.2329,  0.1812,  0.0455,
                           -0.2633, -0.0102],
                          [ 0.2503,  0.1853, -0.0596,  0.1551, -0.0946,  0.0775,  0.1600, -0.0020,
                            0.1709,  0.0196],
                          [-0.0748, -0.0980,  0.0848, -0.1592, -0.1169, -0.1191,  0.2847, -0.2829,
                           -0.2709,  0.0358],
                          [ 0.1138,  0.1503,  0.1485,  0.0621, -0.0402,  0.0364, -0.2527,  0.0785,
                           -0.0985,  0.2441],
                          [ 0.0955, -0.1304,  0.0645,  0.1458,  0.1721,  0.1809,  0.0198,  0.1874,
                            0.2903, -0.2964],
                          [ 0.0918, -0.2241,  0.2559, -0.0230,  0.0306,  0.0319, -0.2530,  0.0194,
                            0.2210, -0.0114],
                          [-0.2207, -0.2347,  0.2004,  0.1407,  0.1616,  0.1039, -0.0131,  0.0682,
                           -0.2842,  0.0146],
                          [-0.2728,  0.0097, -0.2633,  0.1981,  0.0902, -0.2153,  0.2991,  0.3023,
                           -0.0356,  0.0787],
                          [-0.2030,  0.3065,  0.0496,  0.2106, -0.1146,  0.2198,  0.1767, -0.1902,
                            0.1560, -0.2211]])),
                 ('stages.0.bias',
                  tensor([ 0.3091, -0.1789, -0.1619,  0.2745, -0.2241, -0.1725, -0.2759, -0.3069,
                          -0.0204,  0.2387])),
                 ('stages.1.weight',
                  tensor([[-3.0793e-01, -9.0050e-02, -2.0628e-01,  2.1617e-01, -1.1565e-01,
                           -2.3001e-01,  1.1097e-01, -1.3036e-01, -1.4433e-01,  6.0813e-02],
                          [ 2.2130e-01, -4.8575e-02, -1.6314e-01,  1.9930e-01, -1.8808e-01,
                            3.4948e-02,  1.0408e-01, -9.5420e-03, -2.3090e-01,  1.7361e-01],
                          [ 1.6569e-01,  2.0600e-01, -2.0361e-01,  7.3987e-02,  1.5393e-01,
                           -1.1852e-01, -1.8270e-01, -1.0133e-01,  1.6203e-01,  2.3759e-01],
                          [-1.5434e-01,  2.0515e-01, -2.8056e-01, -1.3631e-01, -1.4825e-01,
                            1.0924e-01, -6.0545e-02,  1.8996e-01,  2.1768e-01, -3.0391e-01],
                          [ 9.2278e-02,  1.5420e-01, -1.9240e-01, -1.6297e-01, -2.8009e-01,
                           -2.7083e-01, -2.6585e-01, -8.4825e-03,  3.0573e-01, -9.6221e-02],
                          [ 1.7386e-01, -4.9584e-02, -9.6506e-02, -1.0148e-01, -2.3784e-01,
                            3.0834e-01,  1.2701e-01, -1.1892e-01, -2.9403e-02, -5.1145e-02],
                          [ 2.6342e-02,  5.1342e-03, -1.2207e-01,  2.4433e-01,  2.3663e-01,
                           -2.3547e-01, -1.9406e-01,  1.1746e-01, -3.0585e-01,  2.2586e-01],
                          [-4.8203e-02, -1.1129e-01, -1.4122e-01, -1.3178e-01, -7.3245e-02,
                           -2.9951e-01,  8.1352e-02,  1.4775e-01,  1.9318e-01,  2.8139e-01],
                          [-2.9153e-01, -1.7457e-01, -2.2073e-01, -1.9306e-01, -1.5470e-01,
                            1.6272e-05,  2.6527e-01, -3.1303e-01,  3.1369e-01,  1.4920e-01],
                          [ 1.0000e-01,  2.7836e-01, -2.8917e-01,  5.2028e-02, -3.4789e-03,
                           -2.5739e-01,  2.0492e-01,  2.8123e-01,  3.0439e-01,  3.3280e-03]])),
                 ('stages.1.bias',
                  tensor([ 0.2050, -0.0814, -0.1078,  0.0732,  0.1874, -0.0153,  0.0825, -0.0472,
                           0.2904, -0.0123])),
                 ('out_proj.weight',
                  tensor([[-0.0726,  0.1586,  0.3075, -0.2858, -0.1339, -0.1327,  0.0537,  0.0125,
                            0.3100,  0.1477],
                          [-0.2229,  0.2174,  0.2318, -0.3095, -0.0869,  0.0923, -0.0701, -0.1753,
                           -0.2616,  0.0118]])),
                 ('out_proj.bias', tensor([ 0.2385, -0.2242]))])



Et voila 🎉 We have successfully loaded our checkpoint inside our `model` with linear memory consumption!

I hope you enjoy :) Thanks for reading!
