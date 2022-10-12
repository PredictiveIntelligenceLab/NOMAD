# NOMAD
## Nonlinear Manifold Decoders for Operator Learning

[master_figure-2.pdf](https://github.com/PredictiveIntelligenceLab/NOMAD/files/9767796/master_figure-2.pdf)

This repository contains code and data accompanying the manuscript titled "Nonlinear Manifold Decoders for Operator Learning", authored by Jacob Seidman*, Georgios Kissas*, Paris Perdikaris, and George Pappas.

## Abstract

Supervised learning on function spaces is an emerging area of machine learning research with applications to the prediction of complex physical systems such as fluid flows, solid mechanics, and climate modelling.  By directly learning maps (operators) between infinite dimensional function spaces, these models are able to learn discretization invariant representations of target functions.  A common approach is to represent such target functions as linear combinations of basis elements learned from data. However, there are simple scenarios where, even though the target functions form a low dimensional submanifold, a very large number of basis elements is needed for an accurate linear representation. Here we propose a novel operator learning framework capable of learning  finite-dimensional coordinate representations for nonlinear submanifolds in function spaces.  We show this method is able to accurately learn low dimensional representations of solution manifolds to partial differential equations while outperforming linear models of larger size.  Additionally, we compare to state-of-the-art operator learning methods on a complex fluid dynamics benchmark and achieve competitive performance with a significantly smaller model size and training cost.

## Citation

    @article{seidman2022nomad,
      title={NOMAD: Nonlinear Manifold Decoders for Operator Learning},
      author={Seidman, Jacob H and Kissas, Georgios and Perdikaris, Paris and Pappas, George J},
      journal={arXiv preprint arXiv:2206.03551},
      year={2022}
    }
