from jax.flatten_util import ravel_pytree
from jax.example_libraries.stax import Dense, Gelu
from jax.example_libraries import stax, optimizers
import os
import timeit

import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy.linalg import norm
from jax import random, grad, jit
from functools import partial 
from torch.utils import data
from tqdm import trange
import itertools
import argparse

def output_construction(Ux,t_his, cx, cy, P=1000, ds=3, Nx=32, Ny=32, Nt=100):
    U_all = np.zeros((P,ds))
    Y_all = np.zeros((P,ds))
    it = np.random.randint(Nt, size=P)
    x  = np.random.randint(Nx, size=P)
    y  = np.random.randint(Ny, size=P)
    T, X, Y = np.meshgrid(t_his,cx,cy,indexing="ij")
    Y_all[:,:] = np.concatenate((T[it,x][range(P),y][:,None], X[it,x][range(P),y][:,None], Y[it,x][range(P),y][:,None]),axis=-1)
    U_all[:,:] = Ux[it,x][range(P),y]
    return U_all, Y_all

class DataGenerator(data.Dataset):
    def __init__(self, u, y, s,
                 batch_size=100, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.y = y
        self.s = s
        
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs,outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx,:,:]
        u  = self.u[idx,:,:]
        y = self.y[idx,:,:]
        inputs = (u, y)
        return inputs, s

class operator_model:
    def __init__(self,branch_layers, trunk_layers , n=None, decoder=None, ds=None):    

        seed = np.random.randint(low=0, high=100000)
        self.branch_init, self.branch_apply = self.init_NN(branch_layers, activation=Gelu)
        self.in_shape = (-1, branch_layers[0])
        self.out_shape, branch_params = self.branch_init(random.PRNGKey(seed), self.in_shape)

        seed = np.random.randint(low=0, high=100000)
        self.trunk_init, self.trunk_apply = self.init_NN(trunk_layers, activation=Gelu)
        self.in_shape = (-1, trunk_layers[0])
        self.out_shape, trunk_params = self.trunk_init(random.PRNGKey(seed), self.in_shape)

        params = (trunk_params, branch_params)
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init,self.opt_update,self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                      decay_steps=100, 
                                                                      decay_rate=0.99))
        self.opt_state = self.opt_init(params)
        # Logger
        self.itercount = itertools.count()
        self.loss_log = []

        if decoder=="nonlinear":
            self.fwd = self.NOMAD
        if decoder=="linear":
            self.fwd = self.DeepONet

        self.n  = n
        self.ds = ds


    def init_NN(self, Q, activation=Gelu):
        layers = []
        num_layers = len(Q)
        if num_layers < 2:
            net_init, net_apply = stax.serial()
        else:
            for i in range(0, num_layers-2):
                layers.append(Dense(Q[i+1]))
                layers.append(activation)
            layers.append(Dense(Q[-1]))
            net_init, net_apply = stax.serial(*layers)
        return net_init, net_apply

    @partial(jax.jit, static_argnums=0)
    def NOMAD(self, params, inputs):
        trunk_params, branch_params = params
        inputsu, inputsy = inputs
        b = self.branch_apply(branch_params, inputsu.reshape(inputsu.shape[0], 1, self.ds*inputsu.shape[1])) 
        b = jnp.tile(b, (1,inputsy.shape[1],1))
        inputs_recon = jnp.concatenate((jnp.tile(inputsy,(1,1,b.shape[-1]//inputsy.shape[-1])), b), axis=-1)
        out = self.trunk_apply(trunk_params, inputs_recon)
        return out

    @partial(jax.jit, static_argnums=0)
    def DeepONet(self, params, inputs):
        trunk_params, branch_params = params
        inputsxu, inputsy = inputs
        t = self.trunk_apply(trunk_params, inputsy).reshape(inputsy.shape[0], inputsy.shape[1], self.ds, self.n)
        b = self.branch_apply(branch_params, inputsxu.reshape(inputsxu.shape[0],1,inputsxu.shape[1]*inputsxu.shape[2]))
        b = b.reshape(b.shape[0],int(b.shape[2]/self.ds),self.ds)
        Guy = jnp.einsum("ijkl,ilk->ijk", t,b)
        return Guy
      
    @partial(jax.jit, static_argnums=0)
    def loss(self, params, batch):
        inputs, y = batch
        y_pred = self.fwd(params,inputs)
        loss = np.mean((y.flatten() - y_pred.flatten())**2)
        return loss    

    @partial(jax.jit, static_argnums=0)
    def L2error(self, params, batch):
        inputs, y = batch
        y_pred = self.fwd(params,inputs)
        return norm(y.flatten() - y_pred.flatten(), 2)/norm(y.flatten(),2)

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, batch)
        return self.opt_update(i, g, opt_state)

    def train(self, train_dataset, test_dataset, nIter = 10000):
        train_data = iter(train_dataset)
        test_data  = iter(test_dataset)

        pbar = trange(nIter)
        for it in pbar:
            train_batch = next(train_data)
            test_batch  = next(test_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, train_batch)
            
            if it % 100 == 0:
                params = self.get_params(self.opt_state)
                loss_train = self.loss(params, train_batch)
                loss_test  = self.loss(params, test_batch)
                errorTrain = self.L2error(params, train_batch)
                errorTest  = self.L2error(params, test_batch)
                self.loss_log.append(loss_train)

                pbar.set_postfix({'Training loss': loss_train, 
                                  'Testing loss' : loss_test,
                                  'Test error':    errorTest,
                                  'Train error':   errorTrain})

    @partial(jit, static_argnums=(0,))
    def predict(self, params, inputs):
        s_pred = self.fwd(params,inputs)
        return s_pred

    def count_params(self):
        params = self.get_params(self.opt_state)
        params_flat, _ = ravel_pytree(params)
        print("The number of model parameters is:",params_flat.shape[0])

def main(n, decoder):
    TRAINING_ITERATIONS = 100000
    P = 128
    m = 1024
    num_train = 1000
    num_test  = 1000
    training_batch_size = 100
    du = 3
    dy = 3
    ds = 3
    Nx = 32
    Ny = 32
    Nt = 5

    d = np.load("../Data/SW/train_SW.npz")
    u_train = d["u_train"]
    S_train = d["S_train"]
    T  = d["T"]
    CX = d["CX"]
    CY = d["CY"]

    d = np.load("../Data/SW/test_SW.npz")
    u_test = d["u_test"]
    S_test = d["S_test"]
    T  = d["T"]
    CX = d["CX"]
    CY = d["CY"]


    s_train = np.zeros((num_train,P,ds))
    y_train = np.zeros((num_train,P,dy))
    s_test = np.zeros((num_test,P,ds))
    y_test = np.zeros((num_test,P,dy))

    U_train = u_train.reshape(num_train,Nx*Ny,du)
    U_test  = u_test.reshape(num_test,Nx*Ny,du)


    for i in range(0,num_train):
        s_train[i ,:,:], y_train[i,:,:] = output_construction(S_train[i,:,:,:,:], T, CX, CY, P=P,Nt=Nt)

    for i in range(0,num_test):
        s_test[i,:,:],  y_test[i,:,:]  = output_construction(S_test[i,:,:,:,:], T, CX, CY, P=P,Nt=Nt)

    U_train = jnp.asarray(U_train)
    y_train = jnp.asarray(y_train)
    s_train = jnp.asarray(s_train)

    U_test = jnp.asarray(U_test)
    y_test = jnp.asarray(y_test)
    s_test = jnp.asarray(s_test)

    U_train = jnp.reshape(U_train,(num_train,m,du))
    y_train = jnp.reshape(y_train,(num_train,P,dy))
    s_train = jnp.reshape(s_train,(num_train,P,ds))

    U_test = jnp.reshape(U_test,(num_test,m,du))
    y_test = jnp.reshape(y_test,(num_test,P,dy))
    s_test = jnp.reshape(s_test,(num_test,P,ds))

    train_dataset = DataGenerator(U_train, y_train, s_train, training_batch_size)
    train_dataset = iter(train_dataset)

    test_dataset = DataGenerator(U_test, y_test, s_test, training_batch_size)
    test_dataset = iter(test_dataset)

    if decoder=="nonlinear":
        branch_layers = [m*du,   100, 100, 100, 100, 100, ds*n]
        trunk_layers  = [ds*n*2, 100, 100, 100, 100, 100, ds]
    elif decoder=="linear":
        branch_layers = [m*du,100, 100, 100, 100, 100, ds*n]
        trunk_layers  = [dy,  100, 100, 100, 100, 100, ds*n]

    model = operator_model(branch_layers, trunk_layers, n=n, decoder=decoder, ds=ds)
    model.count_params()

    start_time = timeit.default_timer()
    model.train(train_dataset, test_dataset, nIter=TRAINING_ITERATIONS)
    elapsed = timeit.default_timer() - start_time
    print("The training wall-clock time is seconds is equal to %f seconds"%elapsed)

    params = model.get_params(model.opt_state)


    T, X, Y = np.meshgrid(T, CX, CY,indexing="ij")
    Y_train = jnp.tile(jnp.concatenate((T.flatten()[:,None], X.flatten()[:,None], Y.flatten()[:,None]),axis=-1)[None,:,:],(num_train, 1, 1))
    Y_test  = jnp.tile(jnp.concatenate((T.flatten()[:,None], X.flatten()[:,None], Y.flatten()[:,None]),axis=-1)[None,:,:],(num_test, 1, 1))

    S_test = S_test.reshape(num_test,Nt*Nx*Ny,ds)
    s_pred_test = np.zeros_like(S_test)

    idx = np.arange(0,100)
    for i in range(0,num_test,100):
        idx = i + np.arange(0,100)
        s_pred_test[idx] = model.predict(params, (U_test[idx], Y_test[idx]))
    test_error_rho = []
    test_error_u = []
    test_error_v = []
    for i in range(0,num_train):
        test_error_rho.append(norm(S_test[i,:,0]- s_pred_test[i,:,0],2)/norm(S_test[i,:,0],2))
        test_error_u.append(norm(S_test[i,:,1]- s_pred_test[i,:,1],2)/norm(S_test[i,:,1],2))
        test_error_v.append(norm(S_test[i,:,2]- s_pred_test[i,:,2],2)/norm(S_test[i,:,2],2))
    print("The average test rho error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_rho),np.std(test_error_rho),np.min(test_error_rho),np.max(test_error_rho)))
    print("The average test u error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_u),np.std(test_error_u),np.min(test_error_u),np.max(test_error_u)))
    print("The average test v error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_v),np.std(test_error_v),np.min(test_error_v),np.max(test_error_v)))

    S_train = S_train.reshape(num_train,Nt*Nx*Ny,ds)
    s_pred_train = np.zeros_like(S_train)
    for i in range(0,num_train,100):
        idx = i + np.arange(0,100)
        s_pred_train[idx] = model.predict(params, (U_train[idx], Y_train[idx]))
    train_error_rho = []
    train_error_u = []
    train_error_v = []
    for i in range(0,num_train):
        train_error_rho.append(norm(S_train[i,:,0]- s_pred_train[i,:,0],2)/norm(S_train[i,:,0],2))
        train_error_u.append(norm(S_train[i,:,1]- s_pred_train[i,:,1],2)/norm(S_train[i,:,1],2))
        train_error_v.append(norm(S_train[i,:,2]- s_pred_train[i,:,2],2)/norm(S_train[i,:,2],2))
    print("The average train rho error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(train_error_rho),np.std(train_error_rho),np.min(train_error_rho),np.max(train_error_rho)))
    print("The average train u error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(train_error_u),np.std(train_error_u),np.min(train_error_u),np.max(train_error_u)))
    print("The average train v error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(train_error_v),np.std(train_error_v),np.min(train_error_v),np.max(train_error_v)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('n', metavar='n', type=int, nargs='+', help='Latent dimension of the solution manifold')
    parser.add_argument('decoder', metavar='decoder', type=str, nargs='+', help='Type of decoder. Choices a)"linear" b)"nonlinear"')

    args = parser.parse_args()
    n = args.n[0]
    decoder = args.decoder[0]
    main(n,decoder)
