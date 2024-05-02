<div align="center">
<h1>qsarKit</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions](https://img.shields.io/badge/contributions-welcome-blue.svg)](https://pysd.readthedocs.io/en/latest/development/development_index.html)
[![Py version](https://img.shields.io/badge/python-3.10-blue)](https://pypi.python.org/pypi/pysd/)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Ftahiri-lab%2FQSAR&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub release](https://img.shields.io/badge/release-v1.0-blue)](https://github.com/tahiri-lab/QSAR/releases)

</div>

<h2 align="center">⚛️ QSAR Predictive Modeling for Evaluating Contaminant Transfer</h2>

<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the project</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
     <li>
       <a href="#use-cases">Use cases</a>
    </li>
    <li>
      <a href="#tutorials">Tutorials</a>
    </li>
    <li>
      <a href="#documentation">Documentation</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>

<a id="about-the-project"></a>

# 📝 About the project

`qsarKit` is a Python package that offers robust predictive modeling using QSAR for evaluating the transfer of
environmental contaminants in breast milk. Developed by the dedicated team led by
Professor [Nadia Tahiri](https://tahirinadia.github.io/) at the University of Sherbrooke in Quebec, Canada. This
open-source integrates multiple predictive models, provides synthetic data generation via GANs, and is tailored for
researchers and health professionals.

<a id="installation"></a>

# ⚒️ Installation

[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) is used to handle the environment
dependencies.

Once ```miniconda``` is installed, the environment can be created and activated with the following commands:

```bash
conda env create -f environment.yaml
conda activate qsar_env
```

If you encounter any issues activating the environment, try sourcing the Conda script first and then retry activation:

```bash
source ~/miniconda3/bin/activate qsar_env
```

or if you installed Anaconda instead of Miniconda:
```bash

source ~/anaconda3/bin/activate qsar_env
```

⚠️ We currently only support Python 3.10 due to some dependencies that are not yet compatible with Python 3.11+. We will
update the package as soon as the dependencies are updated.

<a id="use-cases"></a>

# 🚀 Use cases

The `qsarKit` package can be encapsulated in other applications or used as a standalone package.
You can refer to the tutorials on how to use the package functionalities, or use the package as a standalone application.
To perform a quick test, you can run the package with only one model by executing the following command:

```bash
python main.py --config ridge.yaml --output results/
```

For a more generic way of running the package as a standalone application, you can execute the following command by
specifying the ```<config_file>``` (path to the `YAML` configuration file) and ```<output_dir>``` (path to the output
directory).

```bash
python main.py --config <config_file> --output <output_dir>
```

Both arguments are optional. If not provided, the default values are ```config/compare_all_models.yaml```
and ```results/```, respectively.

We can also generate synthetic data using GANs by including the ```gan``` flag in the configuration file.
You can explore examples of the different options provided by the package in the ```config/``` folders.
And you can refer to the ```gan``` tutorial.

<a id="tutorials"></a>

# 📚 Tutorials

We provide several tutorials to help you get started with the package. You can find them in the ```tutorials/``` folder.
You can explore the ```tutorials/models/```, ```tutorials/gan/```, and ```tutorials/preprocessing/``` folders to learn
more about the different functionalities of the package.

<a id="documentation"></a>

# 📖 Documentation

You can also refer to the [documentation](https://tahiri-lab.github.io/QSAR/) for more details.

We generated the documentation using [Sphinx](https://www.sphinx-doc.org/en/master/). To generate the documentation
locally, you can run the following command:

Linux/Mac:
```bash
cd docs/
make html
```

Windows:
```bash
cd docs/
.\make.bat html
```

The documentation will be generated in the ```docs/build/html/``` folder. You can open the ```index.html``` file in your
browser to view the documentation.

<a id="contact"></a>

# 📧 Contact

Please email us at: <Nadia.Tahiri@USherbrooke.ca> for any questions or feedback.

[Go to Top](#about-the-project)