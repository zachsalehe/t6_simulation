# T6 Simulation Pipeline

In this library, we provide a simulation pipeline in order to accurately model the T6, a next-generation computational camera developed by the University of Toronto. The T6 features a dual-tap design, and allows for per-pixel encoded subexposures. The goal of this simulation model is to provide users with an efficient way to accurately prototype the camera’s functionality and results, for use in a variety of applications. 

Here, we overview the steps required to capture training data, learn the camera's noise parameters, and run the simulation.

## Installation

### Quick start
```
git clone https://github.com/saleheza/t6_simulation
conda create -n t6_noise
conda activate t6_noise
pip install -r requirements.txt
python format_data.py --data_root data
python train_gan.py --data_root data
python simulation.py \
   --params data/params.mat \
   --mask masks/t6_intersect1.bmp \
   --input_imgs input/image.png \
   --output_dir output
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
Let $g$ be our constant per-pixel gain parameter. We multiply $I_{in}$ by $g$ in order to account for gain variations across the sensor in our final noisy result.

#### Fixed pattern noise (FPN)
Let $h$ be our constant per-pixel FPN parameter. We add $h$ to our final noisy result.

#### Shot noise
Shot noise normally follows a Poisson distribution with respect to the number of arrived photons. In order to learn our shot noise parameter, we instead choose to approximate this with a Gaussian distribution, which is differentiable with respect to its mean and variance. Let $\lambda_{shot}$ be our sensor-wide shot noise parameter. If our sensor is not saturated (i.e. $I_{in} \cdot g + h < I_{max}$), we add $N_{shot} \sim \mathcal{N}(\mu = 0, \sigma^2 = \lambda_{shot}^2 \cdot I_{in} \cdot g^2)$ to our final noisy result. Otherwise, we simply add $N_{shot} = 0$.

#### Read noise
Let $\lambda_{read}$ be our per-pixel read noise parameter. We add $N_{read} \sim \mathcal{N}(\mu = 0, \sigma^2 = \lambda_{read}^2)$ to our final noisy result.

#### Row noise
Let $\lambda_{row}$ be our sensor-wide row noise parameter. We add $N_{row} \sim \mathcal{N}(\mu = 0, \sigma^2 = \lambda_{row}^2)$ to our final noisy result. Note that each row shares the same Gaussian distributed random variable.

#### Temporal row noise
Let $\lambda_{row_t}$  be our sensor-wide temporal row noise parameter. We add $N_{row_t} \sim \mathcal{N}(\mu = 0, \sigma^2 = \lambda_{row_t}^2)$ to our final noisy result. Note that, in addition to each row sharing the same Gaussian distributed random variable, each image in a consecutive burst also shares the same variables.

#### Quantization noise
Let $\lambda_{quant}$ be our sensor-wide quantization noise parameter. We add $N_{quant} \sim \mathcal{U}(\frac{-\lambda_{quant}}{2}, \frac{\lambda_{quant}}{2})$ to our final noisy result.

#### Dark current
Like shot noise, dark current also follows a Poisson distribution, but instead with respect to time. For the same reasons as before, we also choose to approximate this with a Gaussian distribution. Let $\lambda_{dark}$ be our per-pixel dark current parameter. We add $N_{dark} \sim \mathcal{N}(\mu = \lambda_{dark}^2 \cdot t, \sigma^2 = \lambda_{dark}^2 \cdot t)$ to our final noisy result.

#### Final noisy result
Putting all of our noise sources together, we get out final noisy result, $I_{out}$:
$$I_{out} = I_{in} \cdot g + h + N_{shot} + N_{read} + N_{row} + N_{row_t} + N_{quant} + N_{dark}$$

### Experimental captures
<p align="center">
  <img src=docs/images/experiment.png>
</p>

The figure above demonstrates an example capture setup that can be used to gather experimental data. It is important that our scene is illuminated with natural sunlight, as most conventional lights tend to flicker, which can cause unwanted variations in the captured data. It is also important that we are photographing a plain solid surface, as to eliminate any spatial variations that this may cause in the data. To further reduce spatial variations, the lens cap can also be removed during captures.

The following steps must be done twice; once using an all white mask pattern (isolates left tap), and once using an all black mask pattern (isolates right tap). We will capture 256 photos with 1 subframe each at 50 different exposure times. The exposure times will begin from the T6's minimum exposure setting (26.21 $\mu s$), and increase uniformly until we reach a point where most of our pixels are close to (but not at) their saturation limit. This uniform jump in exposure varies depending on the scene's current illumination. Lastly, we will capture another 256 photos at a much higher exposure time, where each pixel is fully saturated, for the purpose of measuring each pixel's saturation limit. Again, the exposure time required for this will vary depending on the illumination.

```
├── data
│   ├── left_raw
│   │   ├── exp000026.21
│   │   │   ├── 0000.npy
│   │   │   ├── 0001.npy
│   │   │   ├── ....
│   │   │   ├── 0255.npy
│   │   │   ├── black_img.npy
│   │   ├── expXXXXXX.XX
│   │   ├── ...
│   │   ├── expXXXXXX.XX
│   │   ├── saturated
│   │   │   ├── 0000.npy
│   │   │   ├── 0001.npy
│   │   │   ├── ....
│   │   │   ├── 0255.npy
│   │   │   ├── black_img.npy
│   ├── right_raw
│   │   ├── ...
```

The file tree above dictates how experimental data should be stored so it can be properly accessed. Left tap and right tap captures should be saved in folders named `left_raw` and `right_raw` respectively. The content format within each of them remains the same. Captures for any of the 50 exposure times should be placed in folders named `expXXXXXX.XX`, where `XXXXXX.XX` is that captures exposure time in $\mu s$. Each image should be saved as `XXXX.npy`, where `XXXX` denotes the image number within the exposure. The black image, `black_img.npy` should also be saved within each folder, which is used for black calibration. The set of images captured at the camera's saturation limit should be saved in a folder named `saturation`, so it can be easily distinguished. The contents of this folder follow the same structure as before.

### Results
<p align="center">
  <img src=docs/images/comp.png>
</p>

<p align="center">
  <img src=docs/images/hist.png>
</p>

## Simulating T6

### Example 1: all white mask
<p align="center">
  <img src=docs/images/allwhite.png>
</p>

### Example 2: all black mask
<p align="center">
  <img src=docs/images/allblack.png>
</p>

### Example 3: merge mask
<p align="center">
  <img src=docs/images/merge.png>
</p>
