# Direct detection of solar chameleons with electron recoil data from XENONnT

This project investigates the direct detection prospects of chameleon dark energy using current ground-based dark matter experiments. In particular, we demonstrate that XENONnT can probe a solar chameleon.
Our study incorporates both the Sun’s magnetic field profile and Primakoff production in electron and ion electric fields. We find that each channel contributes significantly to the overall production rate and solar scalar flux. This framework updates and extends our previous results, providing a foundation for broader explorations of non-gravitational interactions of dark energy.


The analysis pipeline for detecting solar-induced chameleon dark energy is organized as follows:

### solarmodel.ipynb 
Implements the AGSS09 model and the magnetic field.

### solar_chameleon.py 
Computes primakoff production and magnetic conversion spectrum.

### XENONnT_mcmc.py 
Integrates the chameleon model with XENONnT data and runs Bayesian inference using the ${emcee}$ package. Posterior samples are saved in .npz format.

