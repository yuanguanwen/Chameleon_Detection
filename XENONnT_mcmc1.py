from __future__ import print_function
import sys
import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
from numba import njit
import time
from multiprocessing import Pool
from iminuit import Minuit
from MCMC import MyMCMC, upperLimit
import JsonTools

### load XENONnT data
if __name__ == "__main__":
    data_bin30 = np.loadtxt("XENONnT/data_bin30_wo_bkg.txt")  # Load the event data file
    energy_bin, events, err = data_bin30[:, 0], data_bin30[:, 1], data_bin30[:, 2]  # Extract energy, events reduced background, and error

# constants
pi = 3.14159265359
alpha = 1 / 137.035999084       # fine structure constant
me = 510998.950                 # e- mass [eV]
Mpl = 2e27                      # planck mass [eV]
hbar  = 6.582119569e-19         # Planck's constant  [keV s]
cs    = 2.99792458e10           # speed of light     [cm/s]
hbarc = hbar*cs                 #                    [keV*cm]

# conversion factors
s2eV = 6.582119569e-16      # Hz to eV
T2eV = 2e2                  # Tesla to eV2 conversion [eV2/T]
m2eV = 1.973269804e-7       # m-1 to eV
Dist = 1.496e13             # cm, average distance between Earth and Sun
rSolar = 6.957e8 / m2eV     # solar radius [eV-1]

NXe   = 4.58644e21          # Number of Xe atoms per g
tXE   = 1.042e6             # XENON1T target in g
year  = 3.15e7              # conversion from year to s
conv  = 10**6*year          # conversion from (g*s)^-1 to (year*ton)^-1
eX1T = 0.65                 # XENON1T exposure for 1042kg Xe for 226.9days
eXnT = 20                   # Projected exposure for XENONnT


file = "/Users/yuangw/Documents/GitHub/chameleons/2406.01691/data/"
r = np.loadtxt(file + "r.dat")          # radial distance [eV-1]
T = np.loadtxt(file + "T.dat")          # solar temperature [eV]
rho = np.loadtxt(file + "rho.dat")      # solar density [eV4]
ne = np.loadtxt(file + "ne.dat")        # electron number density [eV3]
nbar = np.loadtxt(file + "nbar.dat")    # Z2-summed number density [eV3]
wp = np.loadtxt(file + "wp.dat")        # plasma frequency [eV]
nH = np.loadtxt(file + "nH.dat")        # H number density [eV3]
nHe3 = np.loadtxt(file + "nHe3.dat")    # He3 number density [eV3]
nHe4 = np.loadtxt(file + "nHe4.dat")    # He4 number density [eV3]
z1 = np.loadtxt(file + "Z1.dat")        # gaunt factors for Z=1
z2 = np.loadtxt(file + "Z2.dat")        # gaunt factors for Z=2

# Define the interpolation functions for the solar model data vs. radius in [eV]
rho_r = interpolate.interp1d(r, rho, fill_value=(rho[0], rho[-1]), bounds_error=False)
T_r = interpolate.interp1d(r, T, fill_value=(T[0], T[-1]), bounds_error=False)
ne_r = interpolate.interp1d(r, ne, fill_value=(ne[0], ne[-1]), bounds_error=False)
wp_r = interpolate.interp1d(r, wp, fill_value=(wp[0], wp[-1]), bounds_error=False)
nbar_r = interpolate.interp1d(r, nbar, fill_value=(nbar[0], nbar[-1]), bounds_error=False)
nH_r = interpolate.interp1d(r, nH, fill_value=(nH[0], nH[-1]), bounds_error=False)
nHe3_r = interpolate.interp1d(r, nHe3, fill_value=(nHe3[0], nHe3[-1]), bounds_error=False)
nHe4_r = interpolate.interp1d(r, nHe4, fill_value=(nHe4[0], nHe4[-1]), bounds_error=False)

radius = rSolar * np.logspace(-2, 0, 100)  # [eV-1]
Tr_arr = T_r(radius)
ne_arr = ne_r(radius)
wp_arr = wp_r(radius)
nbar_arr = nbar_r(radius)
nH_arr = nH_r(radius)
nHe3_arr = nHe3_r(radius)
nHe4_arr = nHe4_r(radius)


# @njit
def mCham2(r, ge=2.0, gc=-3.0, n=1.0):  # effective chameleon mass square, Eq.3-4 in 1110.2583
    phim = (n * Mpl * 10**((4. + n) * gc - ge) / rho_r(r))**(1. / (1. + n))
    return (1. + n) * 10**ge * rho_r(r) / Mpl / phim  # [eV2]

############################# Bulk Magnetic Field #############################
@njit
def Bsolar(radius_arr, Bup=False):
    Bvals = np.zeros_like(radius_arr)
    B = [3000.0*T2eV, 50.0*T2eV, 4.0*T2eV] if not Bup else [200.0*T2eV, 4.0*T2eV, 3.0*T2eV]
    r_arr = radius_arr / rSolar  # radius dimensionless
    for i in range(len(r_arr)):
        r = r_arr[i]
        if r < 0.712:  # radiative zone
            x = r/0.712
            Bvals[i] = B[0] * (9.12) * np.power(1 + 1 / 8.12, 8.12) * x**2 * np.power(1 - x**2, 8.12)
        elif 0.712 < r < 0.752 : # tachocline
            Bvals[i] = B[1] * (1 - ((r - 0.732) / 0.02)**2.0)
        elif 0.925 < r < 0.995: # outer region
            Bvals[i] = B[2] * (1 - ((r - 0.96) / 0.035)**2.0)
        else:
            Bvals[i] = 0.0
    return Bvals  # [eV2]
   
B_arr = Bsolar(radius)  # get the magnetic field at each radius

@njit
def selectG(Tr, w):  
    Tnorm = Tr/me
    idx_T1 = np.searchsorted(z1[0,:1], Tnorm)-1
    idx_T2 = np.searchsorted(z2[0,:1], Tnorm)-1
    idx_w1 = np.searchsorted(z1[:,0]*Tr, w)-1
    idx_w2 = np.searchsorted(z2[:,0]*Tr, w)-1

    g1 = z1[idx_w1, idx_T1]
    g2 = z2[idx_w2, idx_T2]
    return g1, g2

@njit
def GammaPhoton(Tr, ner, nHr, nHe3r, nHe4r, w, g1, g2):  # absorption-production, paper 2406.01691 Eq.A20   
    p1 = 64 * pi**2 * alpha**3
    p2 = 3 * me**2 * w**3
    p3 = me * ner**2 / (2 * pi * Tr)
    p4 = 1 - np.exp(-w / Tr)
    p5 = 8 * pi * alpha**2 * ner / (3 * me**2)
    ions = nHr * g1 + g2 * (4 * nHe4r + 4 * nHe3r)  # sum of ion densities
    return p1/p2 * np.sqrt(p3) * p4 * ions + p5

@njit
def B_integrand(radius, w, ms2_arr, mg2_arr, g1_arr, g2_arr):  # [eV Bg-2] Eq.7 in paper 2406.01691
    results = np.zeros_like(radius)
    n = len(radius)
    for i in range(n):
        ms2 = ms2_arr[i]
        mg2 = mg2_arr[i]
        if w**2 <= mg2 or w**2 <= ms2:
            results[i] = 0.0
            continue  

        r = radius[i]
        Tr = Tr_arr[i]
        ner = ne_arr[i]
        B = B_arr[i]
        nHr = nH_arr[i]
        nHe3r = nHe3_arr[i]
        nHe4r = nHe4_arr[i]
        g1 = g1_arr[i]
        g2 = g2_arr[i]

        G = GammaPhoton(Tr, ner, nHr, nHe3r, nHe4r, w, g1, g2)
        factor1 = w * np.power(w*w - ms2, 1.5) * G
        factor2 = (np.power((mg2 - ms2), 2.0) + (w*w*G*G)) * (np.exp(w/Tr) - 1.0)
        results[i] = 2 * r**2.0 * B**2.0 / (pi * Mpl * Mpl) * factor1 / factor2 * 1e3 / s2eV  # factor 1e3/s2eV convert [eV/eV] to [s-1 keV-1]
    return results
    

############################# Primakoff Process #############################
@njit
def curlyI(u, v):   # Define curlyI
    return (u*u - 1) / v * np.log((u - 1) / (u + 1)) - ((u + v)**2 - 1) / v * np.log((u + v - 1) / (u + v + 1)) - 2

@njit
def curlyIapprox(u, v):   # Define curlyI approximation near u => 1
    return u*u / v - (v + 2) * np.log(v / (v + 2)) - 2

@njit
def T_integrand(radius, w, ms2_arr, mg2_arr): 
    results = np.zeros_like(radius)
    n = len(radius)
    for i in range(n):
        ms2 = ms2_arr[i]
        mg2 = mg2_arr[i]
        if w**2 <= mg2 or w**2 <= ms2:
            results[i] = 0.0
            continue  

        r = radius[i]
        Tr = Tr_arr[i]
        nbarr = nbar_arr[i]
        K2 = 8 * pi * alpha * nbarr / Tr

        kgamma = np.sqrt(np.abs(w**2 - mg2))
        kphi = np.sqrt(np.abs(w**2 - ms2))
        uArg = kgamma / (2 * kphi) + kphi / (2 * kgamma)
        vArg = K2 / (2 * kphi * kgamma)
        Iuv = curlyI(uArg, vArg) if uArg >= 1.01 else curlyIapprox(uArg, vArg)

        factor = alpha/ (8 * Mpl**2 * pi)
        bose = nbarr / (np.exp(w / Tr) - 1)
        results[i] = factor * r**2 * bose * w**2 * kphi / kgamma * Iuv * 1e3 / s2eV  # factor 1e3/s2eV convert [eV/eV] to [s-1 keV-1]
    return results

def flux_earth(w, ge, gg, gc, n):  ## the differential flux at Earth in cm^-2 s^-1 keV^-1
    ms2_arr = mCham2(radius, ge, gc, n)  # effective mass square
    mg2_arr = 4 * pi * alpha * ne_r(radius) / me  # plasma frequency square

    ## Gaunt factors for each radius point
    g1_arr = np.zeros_like(radius)
    g2_arr = np.zeros_like(radius)
    for i in range(len(radius)):
        g1_arr[i], g2_arr[i] = selectG(Tr_arr[i], w)

    ## Compute both integrands
    primakoff = T_integrand(radius, w, ms2_arr, mg2_arr)
    bulk = B_integrand(radius, w, ms2_arr, mg2_arr, g1_arr, g2_arr)
    total = np.trapz(bulk + primakoff, radius)  # integrate over radius
    return (10**gg)**2. *total /(4*pi*Dist*Dist)


#### Detection
# Photon absorption cross section for xenon in Mbarn vs energy in kev
sigma_xenon = np.loadtxt("XENONnT/xenon_cross_sec.csv")
EKeV, sigma2 = sigma_xenon[:, 0], sigma_xenon[:, 1]*4.5868e3  #4.5868e3 convert xenon Mbarn to cm^2/g
sigmae = interpolate.interp1d(EKeV, sigma2, fill_value=(sigma2[0], sigma2[-1]), bounds_error=False)

def cross_sec(w, ge, logMe): # om in keV
    dis = 0.5*NXe*(hbarc*me/1e6* w**2/(2*pi)/10**(4.*logMe))**2.
    conf = (10**ge*w/Mpl/1e3)**2*sigmae(w)*4./alpha
    return dis + conf

# XENONnT efficiency in KeV
efficiency = np.loadtxt("XENONnT/efficiency.txt") 
x1, y1 = efficiency[:, 0], efficiency[:, 1]
epsilon = interpolate.interp1d(x1, y1, fill_value=(y1[0], y1[-1]), bounds_error=False)

# Gaussian Energy resolution; sgm is from XENON paper 2006.09721 Eq.1; w in keV
@njit
def res_sgm(w_obs, w_true):
    sgm = 0.31*np.sqrt(w_true) + 0.0037*w_true # in keV
    return np.exp(-0.5*((w_true-w_obs)/sgm)**2)/(np.sqrt(2*pi)*sgm)

## skew-Gaussian Energy resolution; sgm is from XENONnT paper 2409.08778 Eq.13 and Fig.23; w in keV
@njit
def res_sgm_skew(w_obs, w_true):
    w_true = w_true * (1+ 0.02 * np.arctan(0.017 *w_true) )   ## energy bias
    omega_skew = 0.3842/np.sqrt(w_true) + 0.00477
    alpha_skew = 1.72 * np.power(w_true, -1.169)

    sgm_skew = omega_skew * np.sqrt(1 - 2*alpha_skew**2.0/(np.pi *(1.0+alpha_skew**2.0)**2.0 )) # in keV
    mu_skew = w_true + omega_skew*np.sqrt(2.0/np.pi) * alpha_skew/np.sqrt(1.0+alpha_skew**2.0) # in keV
    return np.exp(-0.5*((w_obs-mu_skew)/sgm_skew)**2)/(np.sqrt(2*pi) * sgm_skew)


omp = np.geomspace(0.1, 30.0, 200)
def Rate_cham(w_array, ge, gg, logMe, gc, n, use_skew=False):   
    result = np.zeros_like(w_array)
    for j in range(len(w_array)):
        w = w_array[j]
        phi = flux_earth(w, ge, gg, gc, n)  # flux at Earth
        cros = cross_sec(w, ge, logMe)  # cross section
        integrand = np.zeros_like(omp)
        for i in range(len(omp)):
            if use_skew:
                integrand[i] = res_sgm_skew(omp[i], w) 
            else:
                integrand[i] = res_sgm(omp[i], w)
        value = epsilon(w) * conv * np.trapz(integrand * cros * phi, omp)  # convolve and integrate
        result[j] = value  # store the result
    return result


############################### Minuit Functions and MCMC ###############################
def chi2(energy_bin, events, err, ge, gg, logMe, gc, n):   ## Calculate the chi-squared value
    event_theory = Rate_cham(energy_bin, ge, gg, logMe, gc, n)
    # chi2_value = np.sum(((event_theory - events) / err) ** 2)
    chi2_value = 0.0
    for i in range(len(energy_bin)):
        chi2_value += ((event_theory[i] - events[i]) / err[i]) ** 2
    return chi2_value


@njit
def lnprior(ge, gg, logMe, gc, n):  ## lnprior is flat in default, 
    if ge < 0.0 or ge > 2.0 or \
       gg < 0.0 or gg > 11.0 or \
       logMe < 0.0 or logMe > 25.0 or \
       gc < -8.0 or gc > 0.0 or \
       n < 0.0 or n > 6.0:
        return -np.inf
    return 0.0


if __name__ == "__main__":
    try:
        seed = int(sys.argv[1])  # Get the seed from command line argument
    except IndexError:
        seed = 4
    np.random.seed(seed+2025)  # Set the random seed for reproducibility
    name = "XENONnT_mcmc_{}".format(seed)  # Name for the output files

    # Define the initial parameters
    init_params = dict(
        ge=1.0,         # log10(beta_e), with beta_e the coupling to electrons
        gg=10.0,        # log10(beta_gamma), with beta_gamma the coupling to photons
        logMe=3.67,     # log10(M_e/keV), with M_e the electron disformal scale
        # logMg=9.0,      # log10(M_gamma/keV), with M_gamma the photon disformal scale
        gc=-3.0,        # log10(Lambda/keV), with Lambda the chameleon potential scale
        n=1.0           # n, with n the chameleon potential power-law
    )

    init_limits = dict(
        ge=(0.0, 2.0),          # Coupling to electrons limits
        gg=(0.0, 15.0),         # Coupling to photons limits
        logMe=(0.0, 25.0),      # Logarithm of electron disformal scale limits in keV
        # logMg=(0.0, 25.0),      # Logarithm of photon disforma scale limits in keV
        gc=(-8.0, 0.0),         # log10(Lambda/keV), with Lambda the chameleon potential scale
        n=(0.0, 6.0)            # Chameleon potential power-law limits
    )
  
    init_limits['gc'] = True  # Fix gc to a specific value
    init_limits['n'] = True  # Fix n to a specific value
    js = JsonTools.JsonObject()  # Initialize JSON tools for saving results

    ### Step1: minuit 
    allargs = init_params.copy()    # Copy initial parameters
    allargs.update(init_limits)     # Update with limits
    m = Minuit(chi2, **allargs)     # Create a Minuit object for minimization
    m.migrad()                      # Perform gradient minimization
    m.print_params()                # Print the parameters after minimization

    # refit
    m= Minuit(chi2, **m.fitarg)         # Recreate Minuit object
    m.set_strategy(2)                   # Set the strategy for minimization
    m.migrad()                          # Perform gradient minimization again
    m.hesse()                           # Calculate the Hessian matrix
    js['Minuit'] = {'values': dict(m.values), 'errors': dict(m.errors), 'covariance': m.covariance, 'chi2': m.fval}  # Save Minuit results to JSON


    ### Step2: MCMC
    pool = Pool(20)                      # Create a multiprocessing pool with 6 workers
    m2 = MyMCMC(chi2, lnprior=lnprior, pool=pool, fix_gc=True, fix_n=True, **m.values)  # Initialize MCMC with the chi-squared function and initial parameters
    m2.run(nstep=20000, burnin=0.1)
    m2.save_samples('chains/samples_{}.npz'.format(name))
    diags = m2.check_converge()
    js['MCMC'] = {'values': dict(m2.values_post), 'errors': dict(m2.errors_post), 'CONVERGENCE': diags}  
    js.writeToFile('chains/fitting_{}.dat'.format(name))  # Save MCMC results to JSON
