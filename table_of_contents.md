<center><h1>Kalman and Bayesian Filters in Python</h1></center>
<p>
 <p>
Table of Contents
-----

[**Preface**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/00_Preface.ipynb)

Motivation behind writing the book. How to download and read the book. Requirements for IPython Notebook and Python. github links.


[**Chapter 1: The g-h Filter**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/01_g-h_filter.ipynb)

Intuitive introduction to the g-h filter, which is a family of filters that includes the Kalman filter. Not filler - once you understand this chapter you will understand the concepts behind the Kalman filter.


[**Chapter 2: The Discrete Bayes Filter**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/02_Discrete_Bayes.ipynb)

Introduces the Discrete Bayes Filter. From this you will learn the probabilistic reasoning that underpins the Kalman filter in an easy to digest form.


[**Chapter 3: Least Squares Filter**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/03_Least_Squares_Filters.ipynb)

Introduces the least squares filter in batch and recursive forms. I've not made a start on authoring this yet.


[**Chapter 4: Gaussian Probabilities**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/04_Gaussians.ipynb)

Introduces using Gaussians to represent beliefs in the Bayesian sense. Gaussians allow us to implement the algorithms used in the Discrete Bayes Filter to work in continuous domains.


[**Chapter 5: One Dimensional Kalman Filters**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/05_Kalman_Filters.ipynb)

Implements a Kalman filter by modifying the Discrete Bayesian Filter to use Gaussians. This is a full featured Kalman filter, albeit only useful for 1D problems.


[**Chapter 6: Multivariate Kalman Filter**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/06_Multivariate_Kalman_Filters.ipynb)

We extend the Kalman filter developed in the previous chapter to the full, generalized filter.


[**Chapter 7: Kalman Filter Math**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/07_Kalman_Filter_Math.ipynb)

We gotten about as far as we can without forming a strong mathematical foundation. This chapter is optional, especially the first time, but if you intend to write robust, numerically stable filters, or to read the literature, you will need to know this.

*This still needs a lot of work. *


[**Chapter 8: Designing Kalman Filters**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/08_Designing_Kalman_Filters.ipynb)

Building on material in Chapter 6, walks you through the design of several Kalman filters. Discusses, but does not solve issues like numerical stability.


[**Chapter 9: Nonlinear Filtering**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/09_Nonlinear_Filtering.ipynb)

Kalman filter as covered only work for linear problems. Here I introduce the problems that nonlinear systems pose to the filter, and briefly discuss the various algorithms that we will be learning in subsequent chapters which work with nonlinear systems.


[**Chapter 10: Unscented Kalman Filters**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/10_Unscented_Kalman_Filter.ipynb)

Unscented Kalman filters (UKF) are a recent development in Kalman filter theory. They allow you to filter nonlinear problems without requiring a closed form solution like the Extended Kalman filter requires.


[**Chapter 11: Extended Kalman Filters**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/11_Extended_Kalman_Filters.ipynb)

Kalman filter as covered only work for linear problems. Extended Kalman filters (EKF) are the most common approach to linearizing non-linear problems.

*Still very early going on this chapter.*


[**Chapter 12: Designing Nonlinear Kalman Filters**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/12_Designing_Nonlinear_Kalman_Filters.ipynb)

Works through some examples of the design of Kalman filters for nonlinear problems. *This is still very much a work in progress.*


[**Chapter 13: Smoothing**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/13_Smoothing.ipynb)

Kalman filters are recursive, and thus very suitable for real time filtering. However, they work well for post-processing data. We discuss some common approaches.


[**Chapter 14: Adaptive Filtering**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/14_Adaptive_Filtering.ipynb)

Kalman filters assume a single process model, but manuevering targets typically need to be described by several different process models. Adaptive filtering uses several techniques to allow the Kalman filter to adapt to the changing behavior of the target.


[**Chapter 15: H-Infinity Filters**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/15_HInfinity_Filters.ipynb)

Describes the <span class="math-tex" data-type="tex">\\(H_\infty\\)</span> filter.

*I have code that implements the filter, but no supporting text yet.*


[**Chapter 16: Ensemble Kalman Filters**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/16_Ensemble_Kalman_Filters.ipynb)

Discusses the ensemble Kalman Filter, which uses a Monte Carlo approach to deal with very large Kalman filter states in nonlinear systems.


[**Chapter XX: Numerical Stability**](not implemented)

EKF and UKF are linear approximations of nonlinear problems. Unless programmed carefully, they are not numerically stable. We discuss some common approaches to this problem.

*This chapter is not started. I'm likely to rearrange where this material goes - this is just a placeholder.*

[**Chapter XX: Particle Filters**](not implemented)

Particle filters uses a Monte Carlo technique to filter.

*This is not implemented, and I have not decided if I want to make it part of this book or not.*




[**Appendix: Installation, Python, NumPy, and filterpy**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/Appendix_A_Installation.ipynb)

Brief introduction of Python and how it is used in this book. Description of the companion
library filterpy.


[**Appendix: Symbols and Notations**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/Appendix_B_Symbols_and_Notations.ipynb)

Symbols and notations used in this book. Comparison with notations used in the literature.

*Still just a collection of notes at this point.*


### Github repository
http://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
