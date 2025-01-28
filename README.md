# Designer-Artifact-User
Multi-agent active inference with PyMDP

# Designer-Artifact-User: Installation Instructions

This guide provides step-by-step instructions to install and set up the **Designer-Artifact-User** application on an Ubuntu 24.04 Linux server.

---
## Purpose of dau-active-inference package

The purpose of the dau-active-inference package is to allow a user to run an active inference based simulation for three interacting agents.

## dau-active-inference package dependencies

- dau-active-inference depends on:
   - numpy
   - matplotlib
   - pymdp
   - inferactively-pymdp
      - pytorch

## Installation Instructions

Follow these steps in sequence:

1. Install python tools and dependencies:
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-virtualenv git

2. Clone and change directory to Designer-Artifact-User directory:
   ```bash
   git clone https://github.com/LearnableLoopAI/Designer-Artifact-User && cd Designer-Artifact-User

3. Create python virtualenv (.venv) and activate from inside Designer-Artifact-User directory:
   ```bash
   virtualenv .venv && source .venv/bin/activate

4. Install dau-active-inference package and dependencies:
   ```bash
   pip install --no-cache-dir dau-active-inference

5. Run dau-active-inference package with --help flag to see switch options:
   ```bash
   dau-active-inference --duration 20

6. Uninstall dau-active-inference package and clean .venv environment like so:
   ```bash
   pip uninstall dau-active-inference && pip freeze | xargs pip uninstall -y

## How to run simulations with the package from PyPI

7. To reach PyPI repository, go to: https://pypi.org/project/dau-active-inference/
   ```bash
   dau-active-inference --duration 20

### duration argument

The duration argument specifies the number of time steps or iterations to perform. The default value of the duration argument is 20. For example, if you need a simulation run over 50 steps, provide:
dau-active-inference --duration 50

Currently, provision is only made for this single argument. In the future, the system might be expanded so that more arguments could be supplied.

If you are more technically-minded and understand the basic operation of the pymdp package, you are welcome to modify the dau_activate_inference.py file and experiment locally.

## How to run simulations with the notebook file Designer-Artifact-User.ipynb

If you are more technically-minded and understand the basic operation of the pymdp package, you are welcome to modify the Designer-Artifact-User.ipynb notebook file.