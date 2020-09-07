# PyTorch-ScalableFHVAE

This is a port of the [Scalable Factorized Hierarchical Variational Autoencoder (ScalableFHVAE)](https://github.com/wnhsu/ScalableFHVAE) to PyTorch and Python 3.

The port is currently in-progress, so some features may not work as intended right now.

## Resources
The two main papers on the ScalableFHVAE and its predecessor, the Factorized Hierarchical Variational Autoencoder (FHVAE):
* [Unsupervised Learning of Disentangled and Interpretable Representations from Sequential Data](https://arxiv.org/abs/1709.07902)
* [Scalable Factorized Hierarchical Variational Autoencoder Training](https://arxiv.org/abs/1804.03201)

The citations for these papers:
```
@inproceedings{hsu2017learning,
  title={Unsupervised Learning of Disentangled and Interpretable Representations from Sequential Data},
  author={Hsu, Wei-Ning and Zhang, Yu and Glass, James},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017},
}
@article{hsu2018scalable,
  title={Scalable Factorized Hierarchical Variational Autoencoder Training},
  author={Hsu, Wei-Ning and Glass, James},
  journal={arXiv preprint arXiv:1804.03201},
  year={2018},
  arxiv={1804.03201},
}
```

[The original code for the FHVAE](https://github.com/wnhsu/FactorizedHierarchicalVAE).

Linked again, the code for the [ScalableFHVAE](https://github.com/wnhsu/ScalableFHVAE).


## Installation
The required Python packages can be installed using `pip install -r requirements.txt`. Running `pip install -r dev-requirements.txt` will install Black, MyPy, and [NumPy type hints](https://pypi.org/project/nptyping), which I am using as part of the porting and development process.

This project also requires [Kaldi](https://github.com/kaldi-asr/kaldi), a library for speech recognition. This will have to be compiled on your machine. The default location for the installation of Kaldi is a subdirectory named `kaldi` in the root of this project's directory (i.e. `PyTorch-ScalableFHVAE/kaldi`), but it may be installed in any directory, with the `--kaldi-root` flag on Python scripts allowing configuration of the root Kaldi directory.
