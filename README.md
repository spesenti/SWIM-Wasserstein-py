# SWIM-Wasserstein

### Background

We consider the problem where a modeller conducts sensitivity analysis of a model consisting of random input factors, a corresponding random output of interest, and a baseline probability measure. The modeller seeks to understand how the model (the distribution of the input factors as well as the output) changes under
a stress on the output's distribution. Specifically, for a stress on the output random variable, we derive the unique stressed distribution of the output that is closest in the **Wasserstein distance** to the baseline output's distribution and satis es the stress. We further derive the stressed model, including the stressed distribution of the inputs, which can be calculated in a numerically e cient way from a set of baseline Monte Carlo samples. 

The proposed reverse sensitivity analysis framework is model-free and allows for stresses on the output such as (a) the mean and variance, (b) any distortion risk measure including the Value-at-Risk and Expected-Shortfall, and (c) expected utility type constraints, thus making the reverse sensitivity analysis framework suitable for risk models.

### Code Organization

We derive the stressed models under two approaches: 1) distribution and 2) simulation. The distribution approach requires us to know the baseline distribution. A lognormal distribution example is presented in the `Distributional Approach` folder. The simulation approach derives the stressed model given a dataset. An in-depth spatial example is presented in the `Simulation Approach` folder.

### Introductory Example

For an introductory example of the simulation approach, see `W_Stress_example.ipynb` where a spatial dataset for modelling insurance portfolio loss is examined.

### Further Reading
For further reading, see https://privpapers.ssrn.com/sol3/papers.cfm?abstract_id=3878879.
