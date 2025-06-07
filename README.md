# SOLFD
**SOLFD** (SOLweig For Design) is a repository for modelling Mean Radiant Temperature (Tmrt) and the physiological equivalent temperature (PET) in urban settings. SOLFD builds upon [SOLWEIG](https://github.com/UMEP-dev/UMEP-processing) from [UMEP](https://umep-docs.readthedocs.io/en/latest/Introduction.html).


# Documentation
*insert read the docs link*


# Running the code
## Code

## Example
`main_jupyter.pynb` contains example chronological order all steps

## Requirements
A requirement file is provided for both pip install and setting up a conda environment. The pip install can be found at /requirements.txt, the conda environment at /conda_environment.yml. The code was developed on a Windows system, using a combination of both pip and Conda. Conda was used as for this system it simplifies installing GDAL. On the other hand, issues have occured with using the h5py package in Conda, these are resolved by using the pip install version. 

# Datasets used
The following datasets were used:

In UMEP, it is possible to run simulations for any day with the ERA5 plug-in. This not used in this code as requires more set up steps, creation of ERA5 account and did not get it to work due to upstream issues ()