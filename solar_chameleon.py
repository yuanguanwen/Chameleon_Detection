import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from numba import njit
import time
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
# plt.style.use("style.txt")	# import plot style

# constants
pi = 3.14159265359
alpha = 1 / 137.035999084       # fine structure constant
me = 510998.950                 # electron mass in eV
Mpl = 2.435e27                  # reduced planck mass [eV]
hbar  = 6.582119569e-16         # Planck's constant  [eV s]
cs    = 2.99792458e10           # speed of light     [cm/s]
hbarc = hbar*cs                 #                    [eV*cm]

# conversion factors
eV2s = 6.582119569e-16      # 1eV to s-1
T2eV = 1.95e2               # Tesla to eV2 conversion [eV2/T]
m2eV = 1.973e-7             # m-1 to eV
cm2eV = 1.973e-5            # cm-1 to eV
Dist = 1.496e13             # cm, average distance between Earth and Sun
rSolar = 6.957e10 / cm2eV   # solar radius in eV

NXe   = 4.58644e21          # Number of Xe atoms per g
tXE   = 1.042e6             # XENON1T target in g
year  = 3.15e7              # conversion from year to s
conv  = 10**6*year          # conversion from (g*s)^-1 to (year*ton)^-1
eX1T = 0.65                 # XENON1T exposure for 1042kg Xe for 226.9days
eXnT = 20                   # Projected exposure for XENONnT
 
file = 'AGSS09/'
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

#### all of these solar parameters are eV units
radius = rSolar * np.logspace(-2, 0, 100)  # [eV-1]
Tr_arr = T_r(radius)
ne_arr = ne_r(radius)
wp_arr = wp_r(radius)
nbar_arr = nbar_r(radius)
nH_arr = nH_r(radius)
nHe3_arr = nHe3_r(radius)
nHe4_arr = nHe4_r(radius) 


# chameleon model params n, gc, ge (electron/matter coupling) are dimensionless, gc -> log10(Lambda/eV)
# assume rho dominated by electron density in eV unit
def mCham2(r, ge=2.0, gc=-2.62, n=1.0):  # effective mass square, Eq.3-4 in 1110.2583
    phim = (n * Mpl * 10**((4. + n)*gc - ge) / rho_r(r))**(1. / (1. + n))
    return (1. + n) * 10**ge * rho_r(r) / Mpl / phim  # [eV2]

############################# Bulk Magnetic Field #############################
@njit
def Bsolar(radius_arr, Bup=False):   ## radius in eV
    Bvals = np.zeros_like(radius_arr)
    B = [3000.0*T2eV, 50.0*T2eV, 4.0*T2eV] if not Bup else [200.0*T2eV, 4.0*T2eV, 3.0*T2eV]
    r_arr = radius_arr / rSolar  # radius dimensionless
    for i in range(len(r_arr)):
        r = r_arr[i]
        if r < 0.712:               # radiative zone
            x = r/0.712
            Bvals[i] = B[0] * 9.12 * np.power(1 + 1 / 8.12, 8.12) * x**2 * np.power(1 - x**2, 8.12)
        elif 0.712 < r < 0.752 :    # tachocline
            Bvals[i] = B[1] * (1 - ((r - 0.732) / 0.02)**2.0)
        elif 0.925 < r < 0.995:     # outer region
            Bvals[i] = B[2] * (1 - ((r - 0.96) / 0.035)**2.0)
        else:
            Bvals[i] = 0.0
    return Bvals                    # [eV2]
   
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
    p2 = 3 * me**2 * w**3               ## me, w in eV unit
    p3 = me * ner**2 / (2 * pi * Tr)
    p4 = 1 - np.exp(-w / Tr)
    p5 = 8 * pi * alpha**2 * ner / (3 * me**2)
    ions = nHr * g1 + g2 * (4 * nHe4r + 4 * nHe3r)  # sum of ion densities
    return p1/p2 * np.sqrt(p3) * p4 * ions + p5         # eV unit

@njit
def B_integrand(radius, w, ms2_arr, mg2_arr, g1_arr, g2_arr):  # [eV Bg-2] Eq.27 in paper 2406.01691
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
        results[i] = 2 * r**2.0 *(2.0/3.0)* B**2.0 / (pi * Mpl * Mpl) * factor1 / factor2   # in [eV] 
    return results
    

############################# Primakoff Process #############################
@njit
def curlyI(u, v):   # Define curlyI
    return (u*u - 1) / v * np.log((u - 1) / (u + 1)) - ((u + v)**2 - 1) / v * np.log((u + v - 1) / (u + v + 1)) - 2.0

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
        results[i] = factor * r**2 * bose * w**2 * kphi / kgamma * Iuv  # in [eV] 
    return results

# @njit
def flux_earth(w, ge, gg, gc, n):  ## the differential flux at Earth in cm^-2 s^-1 keV^-1
    ms2_arr = mCham2(radius, ge, gc, n)  # effective mass square, eV2
    mg2_arr = 4 * pi * alpha * ne_r(radius) / me  # plasma frequency square, eV2

    ## Gaunt factors for each radius point
    g1_arr = np.zeros_like(radius)
    g2_arr = np.zeros_like(radius)
    for i in range(len(radius)):
        g1_arr[i], g2_arr[i] = selectG(Tr_arr[i], w)

    ## Compute both integrands, factor 1e3*eV2s convert [eV eV-1] to [s-1 keV-1]
    primakoff = T_integrand(radius, w, ms2_arr, mg2_arr)   # in [eV]  
    bulk = B_integrand(radius, w, ms2_arr, mg2_arr, g1_arr, g2_arr)  # in [eV] 
    total = np.trapz(bulk + primakoff, radius) *1e3*eV2s  # integrate over radius [eV-1], unit in [s-1 keV-1]
    return (10**gg)**2. *total /(4*pi*Dist*Dist)   # cm^-2 s^-1 keV^-1


#### Detection
# Photon absorption cross section for xenon in Mbarn vs energy[keV]
sigma_xenon = np.loadtxt("XENONnT/xenon_cross_sec.csv")
EKeV, sigma2 = sigma_xenon[:, 0], sigma_xenon[:, 1]*4.5868e3  #4.5868e3 convert xenon Mbarn to cm^2/g
sigmae = interpolate.interp1d(EKeV, sigma2, fill_value=(sigma2[0], sigma2[-1]), bounds_error=False)

def cross_sec(w, ge, logMe): # w in eV, logMe -> log10(M_e/eV)
    # conf = (10**ge*w/Mpl)**2 /(2*pi*alpha) *sigmae(w/1e3)   ## cm^2/g
    dis = (hbarc*me* w**2/10**(4.*logMe))**2. /(8*pi*pi)    ## cm^2 for one particle
    return dis*NXe   ### unit in cm^2/g

# XENONnT efficiency in KeV
efficiency = np.loadtxt("XENONnT/efficiency.txt") 
x1, y1 = efficiency[:, 0], efficiency[:, 1]
epsilon = interpolate.interp1d(x1, y1, fill_value=(y1[0], y1[-1]), bounds_error=False)

# Gaussian Energy resolution; sgm is from XENON paper 2006.09721 Eq.1
@njit
def res_sgm(w_obs, w_true):  ## w_obs/true in keV unit
    sgm = 0.31*np.sqrt(w_true) + 0.0037*w_true # in keV
    return np.exp(-0.5*((w_true-w_obs)/sgm)**2)/(np.sqrt(2*pi)*sgm)

## skew-Gaussian Energy resolution; sgm is from XENONnT paper 2409.08778 Eq.13 and Fig.23; w in keV
@njit
def res_sgm_skew(w_obs, w_true):   ## w_obs/true in keV unit
    w_true = w_true * (1+ 0.02 * np.arctan(0.017 *w_true) )   ## energy bias
    omega_skew = 0.3842/np.sqrt(w_true) + 0.00477
    alpha_skew = 1.72 * np.power(w_true, -1.169)

    sgm_skew = omega_skew * np.sqrt(1 - 2*alpha_skew**2.0/(np.pi *(1.0+alpha_skew**2.0)**2.0 )) # in keV
    mu_skew = w_true + omega_skew*np.sqrt(2.0/np.pi) * alpha_skew/np.sqrt(1.0+alpha_skew**2.0) # in keV
    return np.exp(-0.5*((w_obs-mu_skew)/sgm_skew)**2)/(np.sqrt(2*pi) * sgm_skew)


omp = np.geomspace(0.01, 30.0, 200)  # in keV
def Rate_cham(wkeV_array, ge, gg, logMe, gc, n, use_skew=False):   
    result = np.zeros_like(wkeV_array)
    for j in range(len(wkeV_array)):
        wkeV = wkeV_array[j]
        weV = 1e3 * wkeV
        phi = flux_earth(weV, ge, gg, gc, n)  # flux at Earth, cm^-2 s^-1 keV^-1
        cros = cross_sec(weV, ge, logMe)  # cross section, cm^2/g
        integrand = np.zeros_like(omp)
        for i in range(len(omp)):
            if use_skew:
                integrand[i] = res_sgm_skew(omp[i], wkeV) 
            else:
                integrand[i] = res_sgm(omp[i], wkeV)
        value = epsilon(wkeV) * conv * np.trapz(integrand * cros * phi, omp)  # convolve and integrate
        result[j] = value  # store the result
    return result


w_array = np.linspace(0.1, 30, 100)  # [keV]
events_mod1 = Rate_cham(w_array, ge=1.0, gg=9.4, logMe=4.0, gc=-2.62, n=1, use_skew=False)  # Gaussian resolution


background = np.loadtxt("XENONnT/bkg_model.txt")
bkg = interpolate.interp1d(background[:, 0], background[:, 1], fill_value=(0, 0), bounds_error=False)
events = np.loadtxt("XENONnT/data_unbinned_1to30kev.txt")  # exposure of 1.16 tonne*years

bin30 = np.linspace(0, 30, 31)  # 30 bins => 31 bin edges
count30, bin30_edges = np.histogram(events, bin30) # Create histogram and get counts per bin
bin30_centers = 0.5 * (bin30_edges[:-1] + bin30_edges[1:])
err30 = np.sqrt(count30) # Statistical error bars (Poisson): sqrt(N)
err30 = np.where(count30 == 0, 1.841, np.sqrt(count30))  # Neil Gehrels(1986) recommended err=1.841 when N=0

plt.figure(figsize=(8,6))
plt.errorbar(bin30_centers, count30/1.16, yerr=err30/1.16, fmt='o', color='black', capsize=2.5, label=r'${\rm XENONnT~data}$')
plt.plot(background[:, 0], background[:, 1], color='red', linewidth=2.5, linestyle="-",  label=r"${\rm B_0}$")
plt.plot(w_array, events_mod1+bkg(w_array), label=r"${\rm B_0+Solar~chameleons}$", color='blue', linestyle="--", linewidth=2.5)

plt.xlabel(r'$ E_R\,{\rm[keV]}$', fontsize=22)
plt.ylabel(r'${\rm Events/(ton \times yr \times keV)}$', fontsize=22)
plt.xticks([0, 5, 10, 15, 20, 25, 30], fontsize=22)
plt.yticks([0, 5, 10, 15, 20, 25, 30], fontsize=22)
plt.xlim(0., 28)
plt.ylim(0, 30)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig("plots/events_fitting.pdf", dpi=120)
plt.show()