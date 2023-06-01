# 2023 Summer School for Astrostatistics in Crete

Notebooks from the teaching sessions and the workshops of the
[2023 Summer School for Astrostatistics in Crete](
https://astro.physics.uoc.gr/Conferences/Astrostatistics_School_Crete_2023).
Our Slack Workspace: [2023 Summer School Slack Workspace](https://astrostatisti-vpx2288.slack.com).

## Getting Started

The school material is a collection of Jupyter notebooks, organized in one
folder per subject. The data are also provided in the repository. You can view
and use the material by either:
* Installing the package (see "Installing" below)
* Manually downloading the material and use it (see "Downloading without installation" below)

### Installing (suggested)

We suggest to install the package in a new, clean virtual environment to avoid conflicts with existing software, and
to ensure you are using package versions that were found compatible with the school notebooks. Here we give instructions
for a `conda` environment:

```
conda create --channel conda-forge --name astrostat23 python=3.9.16
conda activate astrostat23
conda install nb_conda=2.2.1
git clone https://github.com/astrostatistics-in-crete/2023_summer_school.git
cd 2023_summer_school
pip install .
```

The `nb_conda` package is automatically registering all conda environments to
the Jupyter kernels. Alternatively, you can run `python -m ipykernel install --user --name=astrostat23` and a new
kernel will appear next time you run Jupyter.

### Downloading without installation (not suggested)

You can download the material in two ways:
* Download the material using the Code -> Download ZIP button.
* Running `git clone https://github.com/astrostatistics-in-crete/2023_summer_school.git` (provided `git` is installed in your system)

Successfully viewing and using the notebooks depends on the Python version, and 
the ability to install all dependencies. It is advisable to create a new clean
Python 3.9.16 environment and work from there. To avoid complications with
dependencies we suggest to install the package (see "Installing above").

## Authors

* **Andrews, Jeff** - University of Florida (USA)

* **Bonfini, Paolo** - Alma Sistemi (Italy)

* **Kovlakas, Kostantinos** - Institute of Space Sciences (Spain)

* **Maravelias, Grigoris** - National Observatory of Athens & FORTH (Greece)

## References

All the material provided here (notebooks and scripts) is licenced
under the GNU GPLv3.

The notebooks have adopted publicly available material from several sources
that are properly credited. All the references to published papers, data, and
software tools are properly addressed within each notebook.

## Acknowledging the school

If the material you learned through this summer school directly and
significantly contributed to your work, we invite you to include the
following acknowledgement in your manuscript:

> We wish to thank the "Summer School for Astrostatistics in Crete" for providing training on the statistical methods adopted in this work.
