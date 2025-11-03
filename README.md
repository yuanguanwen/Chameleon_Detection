# Direct detection of solar chameleons with electron recoil data from XENONnT
This repository contains the analysis framework developed for the study “Direct detection of solar chameleons with electron recoil data from XENONnT.”
Our work reassesses the detection prospects of chameleon dark energy using existing dark matter direct detection experiments, with a focus on the XENONnT detector.
The analysis pipeline for detecting solar chameleons is organized as follows:

### solarmodel.ipynb 
* Implements the AGSS09 solar model, including temperature, density, and composition profiles.
* Define the solar magnetic field configuration used for magnetic conversion.

### solar_chameleon.py 
* Compute the solar chameleon production rate and resulting flux at Earth. 
* Includes both Primakoff production and magnetic conversion contribution.

### XENONnT_mcmc.py 
* Integrates the solar chameleon model with XENONnT electron recoil data and Performs Bayesian inference using the ${emcee}$ package. 
* Posterior samples are saved in .npz format for further analysis or plotting.

If you use this code, please cite the associated paper:
> S. Vagnozzi, L. Visinelli,  P. Brax, A.-C. Davis & J. Sakstein, "*Direct detection of dark energy: the XENON1T excess and future prospects*", [Phys.Rev.D 104 (2021) 6, 063023](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.104.063023) [arXiv:2103.15834](https://arxiv.org/abs/2103.15834)
> T. O'Shea, A.-C. Davis, M. Giannotti, S. Vagnozzi, L. Visinelli & J. K. Vogel,  "*Solar chameleons: Novel channels*", [Phys. Rev. D 110 (2024) 6, 063027](https://doi.org/10.1103/PhysRevD.110.063027)  
[arXiv:2406.01691](https://arxiv.org/abs/2406.01691)
