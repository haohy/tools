# Tools list

[TOC]

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