# Likelihood-Free Inference by Ratio Estimation (LFIRE)
LFIRE is a Python package that uses [ELFI (Engine for Likelihood-Free Inference)](https://github.com/elfi-dev/elfi) and [glmnet](https://github.com/civisanalytics/python-glmnet) for performing Likelihood-Free Inference.

## Installation
LFIRE requires Python 3.6 or greater and a FORTRAN compiler, for Mac users `brew install gcc` will take care of this requirement or `conda install gcc` if you are using anaconda. You can install LFIRE by typing in your terminal:
```
git clone https://github.com/elfi-dev/zoo.git
cd zoo/lfire
make install
```
After the installation you can test the LFIRE framework by typing in your terminal:
```
make test
```

## Docker container
A simple Dockerfile with Jupyter notebook support is also provided. This is a nice way to get started if you have any problems with a FORTRAN compiler. Please see [Docker documentation](https://docs.docker.com/) if you are new with Docker. You can build and run the LFIRE Docker image by typing in your terminal:
```
make docker-build
make docker-run
```
After the LFIRE Docker image is up and running, just open the page <http://localhost:8888> and you are ready to use Jupyter notebook.

## Usage
Please see the `arch_example` Jupyter notebook in the notebooks folder and [ELFI documentation](https://elfi.readthedocs.io/en/latest/) for details about ELFI.

## References
- [1] [Thomas, O., Dutta, R., Corander, J., Kaski, S. and Gutmann, M.U., 2016. Likelihood-free inference by ratio estimation. arXiv preprint arXiv:1611.10242.](https://arxiv.org/abs/1611.10242v5)
- [2] [Friedman, J., Hastie, T. and Tibshirani, R., 2010. Regularization paths for generalized linear models via coordinate descent. Journal of statistical software, 33(1), p.1.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2929880/)

## Citation
TODO
