import numpy as np
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower

def get_Pk(cosmo_pars):
    As, ns, H0, ombh2, omch2 = cosmo_pars
    #Now get matter power spectra and sigma8 at redshift 0 and 0.8
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params(As=As, ns=ns)
    #Note non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=[0.], kmax=2.0)

    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-3, maxkh=1, npoints = 2000)
    s8 = np.array(results.get_sigma8())
    
    return kh, pk[0], s8

N_dim = 5

k_min = 0.01
k_max = 0.1
N_bins = 21

k_bins = np.logspace(np.log10(k_min), np.log10(k_max), N_bins)

from scipy.interpolate import interp1d
from scipy.integrate import quad

def integration_numerator(u, pk_interp):
    kh = np.exp(u)
    jacobian = kh
    return pk_interp(kh) * kh * kh * jacobian

def integration_denominator(u):
    kh = np.exp(u)
    jacobian = kh
    return kh * kh * jacobian

def integration_k_bincentre(u):
    kh = np.exp(u)
    jacobian = kh
    return kh * kh * kh * jacobian

def get_binned_Pk(kh, pk, k_bins):
    pk_interp = interp1d(kh, pk)
    binned_Pk_list = []
    binned_k_list = []
    for i in range(len(k_bins) - 1):
        u_lo = np.log(k_bins[i])
        u_hi = np.log(k_bins[i+1])
        I_n, _ = quad(integration_numerator, u_lo, u_hi, args=(pk_interp,))
        I_d, _ = quad(integration_denominator, u_lo, u_hi)
        I_k, _ = quad(integration_k_bincentre, u_lo, u_hi)
        
        Pk_i = I_n / I_d
        k_i  = I_k / I_d
        
        binned_Pk_list.append(Pk_i)
        binned_k_list.append(k_i)
        
    return np.array(binned_k_list), np.array(binned_Pk_list)

def compute_datavector(cosmo_pars, k_bins):
    kh, pk, _ = get_Pk(cosmo_pars)
    _, Pk_binned = get_binned_Pk(kh, pk, k_bins)
    return Pk_binned

cosmo_pars_fid = np.array([2e-9, 0.97, 70., 0.0228528, 0.1199772])

kh_fid, pk_fid, s8 = get_Pk(cosmo_pars_fid)
binned_k, binned_Pk_fid = get_binned_Pk(kh_fid, pk_fid, k_bins)

delta_k = (k_bins[1:] - k_bins[:-1])
# Volume of NGC-High-z chunk (See eqn A8 of https://arxiv.org/pdf/2009.00622.pdf)
V = 2.78 * 10**9      
Pk_cov = 4 * np.pi**2 * binned_Pk_fid**2 / binned_k**2 / delta_k / V

N_dim = 5
cosmo_prior = np.array([[1.2e-9, 2.7e-9],
                       [0.87, 1.07],
                       [55, 91],
                       [0.01, 0.04],
                       [0.002, 0.5]])

dv_obs = compute_datavector(cosmo_pars_fid, k_bins)

def ln_prior(theta):
    for i in range(N_dim):
        if (theta[i] < cosmo_prior[i,0]) or (theta[i] > cosmo_prior[i,1]):
            return -np.inf
    return 0.
        
def ln_lkl(theta):
    dv_pred = compute_datavector(theta, k_bins)
    delta_dv = (dv_pred - dv_obs)
    return -0.5 * np.sum(delta_dv**2 / Pk_cov)

def ln_prob(theta):
    return ln_prior(theta) + ln_lkl(theta)

import emcee

N_MCMC        = 5000
N_WALKERS     = 48
NDIM_SAMPLING = 5

theta0    = cosmo_pars_fid
theta_std = np.array([0.01 * 2e-9, 0.01, 1., 0.001, 0.01])

# Starting position of the emcee chain
pos0 = theta0[np.newaxis] + theta_std[np.newaxis] * np.random.normal(size=(N_WALKERS, NDIM_SAMPLING))

import os

os.environ["OMP_NUM_THREADS"] = "1"

from multiprocessing import Pool

print("Sampling...")
with Pool() as pool:
    emu_sampler = emcee.EnsembleSampler(N_WALKERS, NDIM_SAMPLING, ln_prob, pool=pool)
    emu_sampler.run_mcmc(pos0, N_MCMC, progress=True)
    
N_BURN_IN = 3000
N_THIN    = 10

samples = emu_sampler.chain[:,N_BURN_IN::N_THIN].reshape((-1,NDIM_SAMPLING))

np.save('output/full_model_output.npy')