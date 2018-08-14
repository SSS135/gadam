# Generalization of Adam, AdaMax, AMSGrad algorithms (GAdam)

Optimizer for PyTorch which could be configured as Adam, AdaMax, AMSGrad or interpolate between them. Like AMSGrad, GAdam maintains maximum value of squared gradient for each parameter, but also GAdam does decay this value over time.


When used with reinforcement learning (Atari + custom PPO implementation) it produces slightly better results than vanilla Adam. Though, I haven't done an extensive hyperparameter search.

## Pseudocode

![equation](http://quicklatex.com/cache3/8b/ql_77ad716480d29576a092a185002cb98b_l3.png)

## Hyperparameters

With `betas` = (beta1, 0) and `amsgrad_decay` = beta2 it will become AdaMax.

With `amsgrad_decay` = 0 it will become [AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ).

I've found it's better to use something in-between, like
   * `betas` = (0.9, 0.99) and `amsgrad_decay` = (0.0001)
   * `betas` = (0.9, 0.95) and `amsgrad_decay` = (0.05)
   
worked best for me, but I've seen good results with wide range of settings.

By default configured as torch.optim.Adam, except `late_weight_decay = True` as proposed in [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101)

## Usage
```python
# best betas and amsgrad_decay i've found
GAdam(model.parameters(), lr=5e-4, betas=(0.9, 0.99), amsgrad_decay=1e-4)
# works good too
GAdam(model.parameters(), lr=5e-4, betas=(0.9, 0.95), amsgrad_decay=0.05)
```

    
## Also there are some additional settings, which was not presented in torch.optim.Adam

* optimism (float, optional): Look-ahead factor proposed in [Training GANs with Optimism](https://arxiv.org/abs/1711.00141). Must be in [0, 1) range. Value of 0.5 corresponds to paper, 0 disables it, 0.9 is 5x stronger than 0.5 (default: 0)
* avg_sq_mode (str, optional): Specifies how square gradient term should be calculated. Valid values are
    * 'weight' will calculate it per-weight as in vanilla Adam (default)
    * 'output' will average it over 0 dim of each tensor,
        i.e. shape[0] average squares will be used for each tensor
    * 'tensor' will average it over each tensor
    * 'global' will take average of average over each tensor,
        i.e. only one avg sq value will be used
* l1_decay (float, optional): L1 penalty (default: 0)
* late_weight_decay (boolean, optional): Whether L1 and L2 penalty should be applied before (as proposed in [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101)) or after (vanilla Adam) normalization with gradient average squares (default: True)
