# NOMAD
## Nonlinear Manifold Decoders for Operator Learning

![master_figure-2](https://user-images.githubusercontent.com/3844367/195421218-164f4be9-f258-4bed-acba-484d67cae7a3.png)

This repository contains code and data accompanying the manuscript titled "Nonlinear Manifold Decoders for Operator Learning", authored by Jacob Seidman*, Georgios Kissas*, Paris Perdikaris, and George Pappas.

## Abstract

Supervised learning on function spaces is an emerging area of machine learning research with applications to the prediction of complex physical systems such as fluid flows, solid mechanics, and climate modelling.  By directly learning maps (operators) between infinite dimensional function spaces, these models are able to learn discretization invariant representations of target functions.  A common approach is to represent such target functions as linear combinations of basis elements learned from data. However, there are simple scenarios where, even though the target functions form a low dimensional submanifold, a very large number of basis elements is needed for an accurate linear representation. Here we propose a novel operator learning framework capable of learning  finite-dimensional coordinate representations for nonlinear submanifolds in function spaces.  We show this method is able to accurately learn low dimensional representations of solution manifolds to partial differential equations while outperforming linear models of larger size.  Additionally, we compare to state-of-the-art operator learning methods on a complex fluid dynamics benchmark and achieve competitive performance with a significantly smaller model size and training cost.

## Code Repository

The repository contains three folders: one for each example presented in the manuscript, namely the Antiderivative, the Advection and the Shallow Water
folders. 

Each example folder should contain 4 subfolders,

- a) "Train_model": containing the python file that can be used for training the model
- b) "Error_Vectors": containing the error vectors resulting from multiple model runs used to generate the figures in the manuscript
- c) "Data": containing the data sets used for training and testing the model (the Antiderivative does not contain a folder like that
   because the data sets are created on-the-fly.)
- d) "Plot_results": containing the python scripts that can be used in order to reproduce the figures presented in the paper.

a) and d) can be found in this GitHub repository and contain only python script files.

All the data and codes required to reproduce the results can be downloaded from the following direct Google Drive link

https://drive.google.com/file/d/1xEzD2swxBcBR5FdHZfc9m7o0Fhe7Z3jB/view?usp=sharing

## Code usage

For training the advection model you need to run the file in the "Train_model" folder with arguments n, the size of the latent dimension, and the type of the decoder, "linear" or "nonlinear". For example if you run "python train_advection 10 nonlinear" you will train the model for the advection case using a solution manifold latent dimension of size 10 and a nonlinear decoder. If you run "python train_advection 10 linear" you will repeat the process using a linear decoder.

For the Antiderivative case, you can run the file in the "Train_model" folder in the same manner, i.e. "python train_antiderivative 10 nonlinear". For the Shallow Water Equations, the same logic applies. So, you can run "python train_SW.py 10 nonlinear" in the "Train_model" folder.

For making plots you need to run the python files in the "Plot_results" folders without any arguments, i.e. for the advection case "python analytic_solution.py". 


## Citation

    @article{seidman2022nomad,
      title={NOMAD: Nonlinear Manifold Decoders for Operator Learning},
      author={Seidman, Jacob H and Kissas, Georgios and Perdikaris, Paris and Pappas, George J},
      journal={arXiv preprint arXiv:2206.03551},
      year={2022}
    }
