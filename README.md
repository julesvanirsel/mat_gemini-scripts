# GEMINI-scripts
Auxiliary scripts for the [GEMINI ionospheric model](https://github.com/mattzett/gemini-scripts).


## Install

These scripts should be at the same directory "level" as `gemini`.
E.g.:

```
/home/username/
   gemini/
   gemini-scripts/
```
direc='./';
Simply `git clone https://github.com/mattzett/gemini-scripts`

### Requirements

either of:

* Matlab &ge; R2007b
* [GNU Octave](https://www.gnu.org/software/octave/) &ge; 4.0


## Usage

The scripts are meant to "just work" without fiddling for a particular computer.
In terms of plot formatting, we prioritize recent versions of Matlab, particularly for generating publication quality plots.

### Top-level scripts

These functions are intended to be run by the end-user or automatically for various purposes.


under `vis/`:

* `magcalc`
* `magcalc_single`
* `magcompare`
* `virtual_spacecraft`

## Notes

### GNU Octave 

GNU Octave is compatible with most Matlab scripts, except those using proprietary Mathworks toolboxes.
Octave generally makes OK plots, particularly with the Qt graphics toolkit.
You can check which Octave graphics toolkit is active by:
```matlab
graphics_toolkit
```
```
ans = qt
```
Qt is the newest and best supported Octave graphics toolkit.
GNUplot is the oldest Octave graphics system and may not be able to make all plots properly.

##Example GEMINI simulations included in this repo

Several different examples are included with the source code; although initial conditions for each must be generated by the user by running a corresponding equilibrium simulations which generates balanced initial conditions for a given date, time, etc.  These equilibrium runs generally are started with a made-up initial condition so there is a lot of initial settling before a sensible ionospheric state is achieved.  To account for this one usually needs to run for about 24 hours of simulation time to insure a set of state parameters that are a good representation of the ionosphere..  Each of these examples has its own initial and boundary conditions generation scripts which are stored in the appropriately named directories in the `initialize/` directory, along with a `config.ini` file as input to the simulation.  The generation scripts must be run in order to produce input grids and initial conditions for each simulation.

The examples are labeled:

* 2DSTEVE - an attempt to model STEVE aurora in 2D using inputs that vaguely resemble data from the nature paper.  
* ARCS - a test case that included a 3D discrete arc
* ARCS_eq - an equilibrium (eq) simulation that generates initial conditions (ICs) for the ARCS simulation described below.
* Aether - a set of example simulations of a cusp and discrete aurora.  There are two sets of files for setting the boundary conditions for the cusp and discrete arc cases, respectively.  
* isinglass - example from Guy's 2017 AGU talk showing our attempt to model the ionosphere during the isinglass launch using 2D PFISR vvels and Guy's inversions as input.  
* RISR_eq - eq simulation for the GDI and KHI examples described below (location:  Resolute Bay ISR)
* isinglass_clayton - pre isinglass event used in Rob Clayton's paper
* isinlgass_clayton_flight - Rob's isinglass example
* isinglass_eq - an equilibrium simulation generating ICs for the ISINGLASS simulation
* nepal20152D_highres - A high resolution simulation in 2D for the 2015 Nepal earthquake; use by Paul Inchin in his paper
* nepal20152D_eq - eq simulation for the 2D nepal earthquake simulation
* tohoku20113D_eq - eq simulation for 2D Tohoku earthquake simulations
* tohoku20112D_highres - simulation of 2011 Tohoku earthquake ionospheric effects (require MAGIC input data)
* tohoku20113D_eq - eq simulation for 3D Tohoku earthquake simulation
* tohoku20113D_medres - A medium resolution simulation in 3D of the 2011 Tohoku earthquake.
* tohoku20113D_medres_control - a background (control) simulation for the medium resolution tohoku example.  
* tohoku20113D_highres - High resolution tohoku simulation
* tohoku20113D_highres_control - A control (no perturbation) run for tohoku (needed to detrend TEC)
* tohoku20113D_highres_restart - an exmaple showing how a simulation can be restarted (under construction)
* tohoku20113D_highres_var - the tohoku simulation on a variable spacing grid
* GDI_periodic_medres_fileinput - a simulation of gradient-drift instability illustrating the use of a periodic mesh
* GDI_periodic_highres_fileinput - a highres GDI example
* GDI_periodic_highres_fileinput_large - a highres GDI example on a larger domain.  This one takes about a week to run on 256 cores.
* KHI_periodic_highres_fileinput - a simulation of Kelvin-Helmholtz instability illustration periodic meshes and use of polarization current solver

A fair bit of testing has been done on these, but there could still be problems so contact a developer if you are having issues with the examples.