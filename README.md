# T6 Simulation Pipeline

In this library, we provide a simulation pipeline in order to accurately model the T6, a next-generation computational camera developed by the University of Toronto. The T6 features a dual-tap design, and allows for per-pixel encoded subexposures. The goal of this simulation model is to provide users with an efficient way to accurately prototype the camera’s functionality and results, for use in a variety of applications. In this library, we overview the steps required to capture training data, learn the camera's noise parameters, and run the simulation.

## Installation

### Quick start
Something like:
```
conda create ...
pip install -r requirements.txt
python run.py
```

### List of dependencies
@TODO

### How to run the different files in the directory, what each of them does
@TODO


## Noise modelling

### Theory
Here, we briefly overview the various learned components of our noise model. 

Consider that our clean input image is $I_{in}$ $[DN]$ (normalized to a 12-bit range), our camera's saturation level is $I_{max}$ $[DN]$, and our simulated exposure time is $t$ $[\mu s]$.

#### Gain
Let $g$ be our constant per-pixel gain parameter. We multiply $I_{in}$ by $g$ in order to account for gain variations across the sensor.

#### Fixed pattern noise (FPN)
Let $h$ be our constant per-pixel FPN parameter. We add $h$ to our final noisy result.

#### Shot noise
Shot noise normally follows a Poisson distribution with respect to the number of arrived photons. In order to learn our shot noise parameter, we instead choose to approximate this with a Gaussian distribution, which is differentiable with respect to its mean and variance. Let $\lambda_{shot}$ be our sensor-wide shot noise parameter. If our sensor is not saturated (i.e. $I_{in} \cdot g + h < I_{max}$), we add $N_{shot} \sim \mathcal{N}(\mu = 0,\sigma^2 = \lambda_{shot}^2 \cdot I_{in}) \cdot g$ to our final noisy result. Otherwise, we simply add $N_{shot} = 0$.

### Experimental captures
@TODO: discuss what experimental captures you'll need to calibrate your parameters

### Results
@TODO: show some noise modelling results (some histograms, aggregate)

## Simulating T6

### Example 1: only 1 mask
@TODO: show an example 


### Example 2: coded mask
@TODO: show an example
