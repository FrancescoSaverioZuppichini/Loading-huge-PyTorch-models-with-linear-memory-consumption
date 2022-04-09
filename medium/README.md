# Loading huge PyTorch models with linear memory consumption

Hello There!

Today we will see how to load a Pytorch model with linear memory consumption. Loading a model takes 2x memory space. Let's see why:

First, we need a model:


https://gist.github.com/7b6d0be662d1fe08afdbf824c2b11bcd

Upon creation, the model takes `1x` memory, where `x` is its size


https://gist.github.com/ef6ae646fd94bdfd41961900a101d58e

At some point, we are going to store our model to disk to use it later


https://gist.github.com/680c7d6325e09d1b323a16a305f30b2d

Later on, we may need to use our stored model.


https://gist.github.com/c5047d04ccf48afe01eb211fc1c747ec



See? We need `2x` memory to load our stored weight. This is problematic if we have a huge model, since we need two times free RAM. For example, assuming we have 16GB of ram and our model uses 10GB. To load it we need 20GB, **we need to change our strategy**.

Recently, PyTorch introduced the `meta` device. When you put a tensor to the `meta` device, only its metadata (e.g. shape) are stored, and its values are tossed away. Thus, no space is used.


https://gist.github.com/39718eb78cf9c75257ca67b3699aa39f


```
tensor([1])
```



https://gist.github.com/68e828e9613e935674febfb4baba8ba6


```
    tensor(..., device='meta', size=(1,), dtype=torch.int64)
```


We can leverage this to load our model with `1x` memory consumption by:

- define our model -> `1x` memory
- place it in the `meta` device -> `1x` memory
- load our `state_dict` -> `1x` memory
- replace all empty parameters of our model with the values inside the `state_dict` -> `1x` memory

Sounds easy, but we first need to figure out how to replace all model's parameters with the original ones from a loaded `state_dict`. Let's create the `load_state_dict_with_low_memory` function.


https://gist.github.com/82133c531a70592d8a47f51a61c390a9


https://gist.github.com/61c521719ffed468d2a6bcfe8efca2d5


```
    OrderedDict([('in_proj.weight', tensor(..., device='meta', size=(10, 2))),
                 ('in_proj.bias', tensor(..., device='meta', size=(10,))),
                 ('stages.0.weight', tensor(..., device='meta', size=(10, 10))),
                 ('stages.0.bias', tensor(..., device='meta', size=(10,))),
                 ('stages.1.weight', tensor(..., device='meta', size=(10, 10))),
                 ('stages.1.bias', tensor(..., device='meta', size=(10,))),
                 ('out_proj.weight', tensor(..., device='meta', size=(2, 10))),
                 ('out_proj.bias', tensor(..., device='meta', size=(2,)))])
```

The model is empty now.

Now we have to figure out in which submodule of `model` each parameter from `state_dict` has to go. One way to do it is to create a dictionary with `[key_in_state_dict] -> [submodule_in_module]`. 

 So we know where we have to place the values from the loaded `state_dict`. Remember, as soon as the model is placed inside the `meta` device, all its weights are tossed away.


https://gist.github.com/fc624b53f9d17bd7cd5cd95626e72554


https://gist.github.com/9a01a9fbae706bada6ae0c5ed3c90746

```
    {'in_proj.weight': Linear(in_features=2, out_features=10, bias=True),
     'in_proj.bias': Linear(in_features=2, out_features=10, bias=True),
     'stages.0.weight': Linear(in_features=10, out_features=10, bias=True),
     'stages.0.bias': Linear(in_features=10, out_features=10, bias=True),
     'stages.1.weight': Linear(in_features=10, out_features=10, bias=True),
     'stages.1.bias': Linear(in_features=10, out_features=10, bias=True),
     'out_proj.weight': Linear(in_features=10, out_features=2, bias=True),
     'out_proj.bias': Linear(in_features=10, out_features=2, bias=True)}
```


Cool, now we have a way to know which key goes with which submodule of `model`. Let's go back to our `load_state_dict_with_low_memory` function and materialize each submodules parameter using the correct value from `state_dict`


https://gist.github.com/ce1c6085404d4af0109495cf1c9cd2c9


https://gist.github.com/4442baeb029d0ea4a842fe617155801d

```
    OrderedDict([('in_proj.weight', tensor(..., device='meta', size=(10, 2))),
                 ('in_proj.bias', tensor(..., device='meta', size=(10,))),
                 ('stages.0.weight', tensor(..., device='meta', size=(10, 10))),
                 ('stages.0.bias', tensor(..., device='meta', size=(10,))),
                 ('stages.1.weight', tensor(..., device='meta', size=(10, 10))),
                 ('stages.1.bias', tensor(..., device='meta', size=(10,))),
                 ('out_proj.weight', tensor(..., device='meta', size=(2, 10))),
                 ('out_proj.bias', tensor(..., device='meta', size=(2,)))])
```

https://gist.github.com/1e5ce01e52a9ccf9df30119889eca012


https://gist.github.com/ead0e5b95219c41bff98dc4516fa67ca


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
                 ...
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
```


Et voila 🎉 We have successfully loaded our checkpoint inside our `model` with linear memory consumption!

I hope you enjoy :) Thanks for reading!
