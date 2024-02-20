# DEDD: Dark matter Electron scattering Direct detection from Deep learning

DEDD is a deep learning model trained on [QEdark-EFT for crystals](https://github.com/urdshals/QEdark-EFT/tree/Crystals). The model allows quick and easy evaluation of rates of electron hole pair creation in Silicon and Germanium. 

## About the code
The code requires [TensorFlow](https://www.tensorflow.org) and [NumPy](https://numpy.org).
When evaluating rates of DM induced electron hole pair creation, specify the material at the top of *inference.py*. Below the material the effective couplings can be specified. Finally, the DM mass is specified. These can either be set to scalars for evaluating a single DM model, or to 1D arrays for several simultanious evaluations. ``` f.inference ``` returns the rate of DM electron scatterings creating 1, 2, 3 and 4 electron hole pairs.
