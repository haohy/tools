# Tools list

[TOC]

## Python 

### Reload module

```python
# Python 3.4+ only.
from importlib import reload  

# Condition 1: foo is a folder, ./foo/utils.py, reload utils.py
reload(foo.utils)
from foo.utils import *

# Condition 2: foo.py is a script, reload foo.py
reload(foo)
```



## Deep Learning (Pytorch)

### Parameters Counting

```python
# The model is defined before, the codes below counts the number of parameters in training model
num_parameters_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

### Model DataParallel on multi GPUs

```python
# Pytorch will only use one GPU by default. You can easily run your operations on multiple GPUs 
# by making your model run parallelly using `DataParallel`
model = nn.DataParallel(model)
```

### TensorboardX guide

```python
# import module 
from tensorboardX import SummaryWriter

# define logger
logger = SummaryWriter(dir_logs)

# write variable to tensorboard
logger.add_scalars('{}/loss'.format(log_model_name), \
                   {'loss': loss_epoch.item()}, epoch)
```

```bash
# view logs across browser
tensorboard --logdir ./ [--port 6007]
```



## Markdown

### A collapsible section with markdown

```markdown
# A collapsible section with markdown (work in github)
<details>
  <summary>Click to expand!</summary>
  
  ## Heading
  1. A numbered
  2. list
     * With some
     * Sub bullets
</details>
```

## Matplotlib

### Draw scatter figure

```python
# data:
# [['306', '3.27', 'Fake Pruning(60%)'],
#  ['422', '3.11', 'Fake Pruning(40%)']]
x = [float(i) for i in data[:,0]]
y = [float(i) for i in data[:,1]]
label = data[:,2]

fig, ax = plt.subplots(figsize=(5,5))
# different color of two part of points
ax.scatter(x[:9], y[:9], c='black', marker='o')
ax.scatter(x[9:], y[9:], c='red', marker='x')
ax.set_xlim(300, 450)
ax.set_ylim(3, 3.4)
# set interval of coordinate axis
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.set_xlabel('FLOPs(M)', fontsize=15)
ax.set_ylabel('CIFAR-10 Error(%)', fontsize=15)
# set grid's shape
plt.grid(ls='--')
for i, txt in enumerate(label):
    if i < 1:
        ax.annotate(txt, (x[i]+5,y[i]), fontsize=10)
    else:
        ax.annotate(txt, (x[i]+5,y[i]), color='r', fontsize=10)
plt.show()
plt.savefig('./result.png')
```

-   display 

    ![scatter_figure][figure_1]



[figure_1]: ./images/scatter_figure.png

