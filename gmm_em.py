import numpy as np
import soundfile as sf
import sidekit as skk
import os
import sys
import multiprocessing
import logging
import htkmfc
import time
from scipy import sparse

def gmm_em(dataLista, nmixa, final_nitera, ds_factora, nworkers, gmmFilename):
    try:
        nmix = float(nmixa)
    except:
        nmix = 256 #default

    try:
        final_niter = float(final_nitera)
    except:
        final_niter = 10 #default

    try:
        ds_factor = float(ds_factora)
    except:
        ds_factor = 1  # default

    if not is_power2(nmix):
        print('Error nmix must be power of 2')


    # Load data
    dataList = load_data(dataLista)

    nfiles = dataList.__len__()

    print ('\n\n Initializing the GMM hyperparameters ...\n')

    gm, gv = comp_gm_gv(dataList)
    gmm = gmm_class(gm, gv)

    niter = [1,2,4,4,4,4,6,6,10,10,15];
    idd = np.log2(nmix) + 1
    niter[idd] = final_niter;
    mix =1
    while (mix <= nmix):
        if (mix >= nmix/2):
            ds_factor = 1
        print ('\nRe-estimating the GMM hyperparameters for %d components ...\n', mix)
        for iter in range(0,niter(np.log2(mix))):
            print('EM iter#: %d \t', iter)
            N = 0;
            F = 0;
            S = 0;
            L = 0;
            nframes = 0;
            start_time = time.time()
            for ix in range(0,nfiles-1):
                n, f, s, l = expectation(dataList[ix][:, 0:ds_factor:len(dataList[ix])-1], gmm);
                N = N + n;
                F = F + f;
                S = S + s;
                L = L + sum(l);
                nframes = nframes + len(l);
            elapsed_time = time.time() - start_time
            print('[llk = %.2f] \t [elaps = %.2f s]\n', L / nframes, elapsed_time);
            gmm = maximization(N, F, S);
        if (mix < nmix):
            gmm = gmm_mixup(gmm);
        mix = mix * 2;

    return gmm

class gmm_class(object):
  def __init__(self, glob_mu, glob_sigma):
     self.mu = glob_mu
     self.sigma = glob_sigma
     self.w = 1

def is_power2(num):
	'states if a number is a power of two'
	return num != 0 and ((num & (num - 1)) == 0)

def load_data(datalist):
    if isinstance(datalist, basestring):
        with open(datalist) as f:
            lines = f.read().splitlines()
            #need htk read here
            nfiles = len(lines)
            dtList = []
            for ix  in range(1,nfiles):
                signaal, sampleratea  = sf.read(lines[ix])
                aaa = skk.mfcc(signaal)
                dtList[ix] = aaa[0]
            result = dtList
    else:
        result = datalist
    return result

def comp_gm_gv(dataList):
    #compute global mean and variance
    globalmean = np.mean(dataList)
    gvariance = np.var(dataList)
    return globalmean, gvariance

def expectation(data, gmm):
    post, llk = postprob(data, gmm.mu, gmm.sigma, gmm.w)
    N = np.transpose(sum(post, 2))
    F = data * np.transpose(post)
    t = np.array(data)
    S = (t*t) * np.transpose(post)
    return N, F ,S, llk

def postprob(data, mu, sigma, w):
    post = lgmmprob(data, mu, sigma, w);
    llk  = logsumexp(post, 1);
    pp = np.array(post)
    lll = np.array(llk)
    post = np.exp(post - llk)
    return post, llk

def lgmmprob (datat ,mu,sigma,w):
    nd = np.shape(datat)
    ndim = nd[1]
    muu = np.array(mu)
    sigg = np.array(sigma)
    data = np.array(datat)
    C = sum(muu * muu / sigg) + sum(np.log(sigg));
    D = np.transpose(1. / sigma) * (data * data) - 2 * np.transpose(mu/sigma) * data + ndim * np.log(2 * np.math.pi);
    logprob = -0.5 * (np.transpose(C)  + D)
    logprob = logprob + np.log(w)
    return logprob

def logsumexp(x,dim):
    xmax = np.max(x,dim)
    y= xmax + np.log(np.sum(np.exp(x-xmax), dim));
    return y

def maximization(Nr, Fr, Sr):
    N= np.array(Nr)
    F = np.array(Fr)
    S = np. array(Sr)
    w  = N / np.sum(N);
    mu = F/N;
    sigma = (S/N) - (mu*mu);
    sigma = apply_var_floors(w, sigma, 0.1);
    gmm.w = w;
    gmm.mu= mu;
    gmm.sigma = sigma;
    return gmm

def apply_var_floors(wr, sigmar, floor_constr):
    sigma  = np.array(sigmar)
    w = np.array(wr)
    floor_const = np.array(floor_constr)
    vFloor = sigma * np.transpose(w) * floor_const;
    sigma  = np.max(sigma, vFloor)
    return sigma

def gmm_mixup(gmm):
    mu = gmm.mu; sigma = gmm.sigma; w = gmm.w
    [ndim, nmix] = np.size(sigma)
    [sig_max, arg_max] = max(sigma)
    eps = sparse(0 * mu)
    for inx in range(0,nmix-1):
        idx = arg_max + (inx - 1) * np.size(ndim, nmix);
        eps[idx] = np.sqrt(sig_max)
    mu = [mu - eps, mu + eps];
    sigma = [sigma, sigma];
    w = [w, w] * 0.5;
    gmm.w  = w;
    gmm.mu = mu;
    gmm.sigma = sigma;
    return gmm


