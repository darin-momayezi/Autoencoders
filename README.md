# Autoencoders
This repository documents my reserach projects as an undergraduate in Dr. Roman Grigoriev's Non-Linear Dynamics and Chaos lab at Georgia Tech.
My main project is applying Autoencoders to the problem of modeling fine structure dynamics in turbulent data. This is a well-posed problem because fine structure is both important and extremely expensive to simulate in detail. Autoencoders have proven to be effective in many areas for dimensionality reduction. Reduced dimensional modeling of fine structure in turbulent systems would reduce the computational cost of accurate models and bring important dynamics into our understanding of turbulent systems. 

The question is do Autoencoders work in this way? Well, yes! They do seem to have promise in discovering models of physical systems. To motivate this let's consider a 1-D PDE whose solution is a simple sum of cosines
$$Acos(ku) + Bcos(qu)$$
$$u=x+ct, \text{   } c=\pi$$
$$k << \frac{1}{dx}, \text{    } q >> \frac{1}{dx}.$$

There are many situations in which you may expect solutions to consist of high and low frequency components. The graphs of the high frequency component, the low frequency component, and their sum, respectively, are shown below.
<img
  src="images/high.png"
  alt="Alt text"
  title="Low Frequency"
  width="80" height="80">
<img
  src="images/low.png"
  alt="Alt text"
  title="High Frequency"
  style="display: inline-block; margin: 0 auto; max-width: 40px">
<img
  src="images/sum.png"
  alt="Alt text"
  title="Sum"
  style="display: inline-block; margin: 0 auto; max-width: 40px">
We would expect an Autoencoder to learn the first and second components with just one dimensional latent spaces and the their sum with a two dimensional latent space. 

This project started by reproducing the results produced by Graham and Linot who showed the dimensionality of the inertial manifold of the Kuramoto-Sivashinky system (https://arxiv.org/pdf/2109.00060.pdf).

I then explored a simple 1D PDE to gain more intuiton with Autoencoders.

The current work is focused on the 2-D Navier Stokes equations and finding low dimensional representations of the fine structure. Autoencoders may also be useful for quantifying the separation between high dimensional and low dimensional structure. 

Below is a simple Autoencoder applied to time averaged numerical data of the 2D Navier-Stokes equation. It is split into sixty-four spatial subdomains with their corresponding latent dimension reported by the Autoencoder. This is still a work in progress and I am applying more sophisticated techniques to increase accuracy and trust in the network, but it serves as a visual example of the kind of work I am currently doing. 

<img
  src="singlePeriodLatentDims.png"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 100px">


_______________________________________________________________
# Vorticity field

<img
  src="vorticity_images.gif"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 100px">
