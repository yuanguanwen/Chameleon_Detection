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
* Integrates the solar chameleon model with XENONnT electron recoil data and Performs Bayesian inference using the ${emcee}$ package. \\
* Posterior samples are saved in .npz format for further analysis or plotting.

If you use this code or results in your work, please cite:
@article{Vagnozzi:2021quy,
    author = "Vagnozzi, Sunny and Visinelli, Luca and Brax, Philippe and Davis, Anne-Christine and Sakstein, Jeremy",
    title = "{Direct detection of dark energy: The XENON1T excess and future prospects}",
    eprint = "2103.15834",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1103/PhysRevD.104.063023",
    journal = "Phys. Rev. D",
    volume = "104",
    number = "6",
    pages = "063023",
    year = "2021"
}

@article{OShea:2024jjw,
    author = "O'Shea, Tom{\'a}s and Davis, Anne-Christine and Giannotti, Maurizio and Vagnozzi, Sunny and Visinelli, Luca and Vogel, Julia K.",
    title = "{Solar chameleons: Novel channels}",
    eprint = "2406.01691",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1103/PhysRevD.110.063027",
    journal = "Phys. Rev. D",
    volume = "110",
    number = "6",
    pages = "063027",
    year = "2024"
}
