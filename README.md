# Chameleon Dark Energy Detection with XENONnT

This project investigates the direct detection prospects of chameleon dark energy using current ground-based dark matter experiments. In particular, we demonstrate that XENONnT can probe a solar flux of chameleons produced in the Sun.
Our study incorporates both the Sun’s magnetic field profile and Primakoff production in electron and ion electric fields. We find that each channel contributes significantly to the overall production rate and solar scalar flux. This framework updates and extends our previous results, providing a foundation for broader explorations of non-gravitational interactions of dark energy.

## Analysis Pipeline
The workflow for detecting solar-induced chameleon dark energy is organized as follows:

### solarmodel.ipynb \
Implements the AGSS09 solar model.

### detectormodel.ipynb \
Encodes XENONnT’s efficiency, resolution, and dataset.

### chameleonmodel.ipynb \
Computes Primakoff and bulk magnetic field production channels, together with their event rates.

### mcmc.py \
Integrates the chameleon model with XENONnT data and runs Bayesian inference using the emcee package. Posterior samples are saved in .npz format.

### posterior_analysis.ipynb \
Uses GetDist to visualize posteriors and generate corner plots.
