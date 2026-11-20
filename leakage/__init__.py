"""Subliminal-learning leakage experiments (teacher -> aux-only student on noise).

Modules
-------
models   : MLP / Wang-CNN / Transformer / GRU (PyTorch) with a real_logits and an aux_logit head
training : teacher training, aux-only distillation, noise generators
metrics  : classification metrics incl. balanced accuracy and the chance-adjusted leakage score
data     : smart meter loading, household-level splits, synthetic data for smoke tests
report   : seed aggregation (mean, std, 95% CI) and plots
"""
