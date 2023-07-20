# OpenFL-XAI

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
<!--[![Citation](https://img.shields.io/badge/cite-citation-brightgreen)](#)-->

OpenFL-XAI is an extension to the open-source [Intel® OpenFL][openfl] framework for providing user-friendly support to Federated Learning (FL) of Fuzzy Rule-Based Systems (FRBS) as explainable-by-design models.
As an extension to OpenFL, OpenFL-XAI enables several data owners, possibly dislocated across different sites, to collaboratively train an eXplainable Artificial Intelligence (XAI) model while preserving the privacy of their raw data. An overview of the extensions to OpenFL that characterize OpenFL-XAI is reported in **openfl_changelog.txt**. <br/>
By supporting FL of highly interpretable FRBSs, OpenFL-XAI addresses two key requirements towards trustworthiness of AI systems, namely privacy preservation and transparency.

The current version of the framework includes the implementation of FL of Takagi-Sugeno-Kang Fuzzy Rule-Based Systems for solving regression problems (please refer to [Bárcena et al., 2022][Barcena2022] for more details). However, the framework is composed by a set of general classes allowing developers to design additional rule-based systems and new aggregation schemes.

This work has been developed by the Artificial Intelligence R&D Group at the Department of Information Engineering, University of Pisa, as part of the activities carried out within [Hexa-X Project, the European Union's Flagship 6G Project][hexa]. OpenFL-XAI has supported research, development, and demonstration activities concerning the FL of XAI models, which has been recently awarded as [key innovation][inno] by the EU Innovation Radar. This work has been also partially funded by the PNRR - M4C2 - Investimento 1.3, Partenariato Esteso PE00000013 - ``FAIR - Future Artificial Intelligence Research" - Spoke 1 ``Human-centered AI"

<p align="center">
	<img src="./images/hexa-x_logo_color_large.png" alt="tree aggregator cert" style="height: 80px">
	&emsp;&emsp;
	<img src="./images/logo-DII.png" alt="tree aggregator cert" style="height: 80px">
	&emsp;&emsp;
	<img src="./images/logo_fair.png" alt="tree aggregator cert" style="height: 80px">
</p>

## Table of Contents

- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Illustrative Example](#illustrative-example)
- [Setup and run a new Federated Learning process](#setup-and-run-a-new-federated-learning-process)
- [License](#license)
- [Citations](#citations)

# Repository Structure

```bash
├── openfl-xai_workspaces                   # folder containing two OpenFL workspaces, namely:
│   ├── xai_frbs_generic                    	# Openfl-XAI workspace template containing all the customized classes to enable FL of XAI models.
│   ├── xai_tsk_frbs                        	# workspace of a first order TSK-FRBS, based on the Openfl-XAI workspace. This workspace is used in the Illustrative Example.
├── certificates                            # folder containing certificates used by Aggregator and Collaborators to prove their identity.
├── data                                    # folder containing the private data of the Collaborators, to be used for local model training.
├── configuration.json                      # json file in which the name of the XAI model to be used is specified.
├── docker-compose.yml                      # file used to generate Docker images to deploy Openfl-XAI components in Docker containers.
├── Dockerfile.openfl_xai                   # file used to generate Docker images to deploy Openfl-XAI components in Docker containers.
├── Dockerfile.xai_aggregator               # file used to generate Docker images to deploy Openfl-XAI components in Docker containers.
├── Dockerfile.xai_collaborator             # file used to generate Docker images to deploy Openfl-XAI components in Docker containers.
├── global_models                           # folder for storing aggregated model.
├── logs                                    # folder for storing containers logs.
├── terminal_interface.py                   # Python module implementing a command line interface for executing the Illustrative Example.
├── OpenFL_changelog.txt                    # changes made in OpenFL-XAI as extension of the OpenFL base components.
├── use_case_requirements.txt               # dependencies needed to execute the Illustrative Example. These requirements are installed in the Docker images.
├── Illustrative_Example.md                 # documentation: guide to execute an illustrative example for FL of first-order TSK-FRBS.
├── Customization_Guide.md                  # documentation: guide to customize FL process with your own models and settings.
└── images                                  # utility folder for documentation images.
```




# Prerequisites

OpenFL-XAI requires:

- [Python 3](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/engine/install/)

In addition, the following Python packages are required to leverage the generated models and perform inference outside the Docker environment.

- NumPy >= 1.24.3
- SimpFul >= 2.11.0
- Scikit-learn >= 1.2.2

# Illustrative Example
Check the [illustrative example][IllustrativeExample] for a step-by-step guide to the usage of OpenFL-XAI for learning a first-order TSK-FRBS in a federated fashion. 

# Setup and run a new Federated Learning process
Check the [customization guide][CustomizationGuide] for a step-by-step guide to the customization of the FL process with your own models and settings.

# License
This project is licensed under [Apache License Version 2.0][License]. By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

# Contributors
- Mattia Daole (mattia.daole@phd.unipi.it)
- Alessio Schiavo (alessio.schiavo@phd.unipi.it)
- José Luis Corcuera Bárcena ([scholar](https://scholar.google.it/citations?user=dasDbcAAAAAJ)) (joseluis.corcuera@phd.unipi.it)
- Pietro Ducange ([scholar](https://scholar.google.it/citations?user=HCgZqXEAAAAJ)) (pietro.ducange@unipi.it)
- Francesco Marcelloni ([scholar](https://scholar.google.it/citations?user=_EkQr2QAAAAJ)) (francesco.marcelloni@unipi.it)
- Alessandro Renda ([scholar](https://scholar.google.it/citations?user=13nYgdUAAAAJ)) (alessandro.renda@unipi.it)

# Citations
1. M. Daole, A. Schiavo, P. Ducange, A. Renda, J. L. Corcuera Bárcena, F. Marcelloni, "OpenFL-XAI: Federated Learning of Explainable Artificial Intelligence Models in Python" (under review)

```
@article{openfl-xai_citation,
	author={Daole, Mattia and Schiavo, Alessio and Corcuera B{\'a}rcena, Jos{\'e} Luis and Ducange, Pietro and Marcelloni, Francesco and Renda, Alessandro},
	title={OpenFL-XAI: Federated Learning of Explainable Artificial Intelligence Models in Python},
	journal={Under review},
	url={},
	year={},
	doi={},
	publisher={}
}
```

2.  J. L. Corcuera Bárcena, P. Ducange, A. Ercolani, F. Marcelloni, A. Renda, "An Approach to Federated Learning of Explainable Fuzzy Regression Models", in: 2022 IEEE International Conference  in Fuzzy Systems (FUZZ-IEEE), IEEE, 2022, pp. 1–8. doi:10.1109/FUZZ-IEEE55066.2022.9882881.

```
@INPROCEEDINGS{CorcueraBarcena-FedTSK,   
  author={Corcuera B{\'a}rcena, Jos{\'e} Luis and Ducange, Pietro and Ercolani, Alessio and Marcelloni, Francesco and Renda, Alessandro},   
  booktitle={2022 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE)},   
  title={An Approach to Federated Learning of Explainable Fuzzy Regression Models},   
  year={2022},   
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/FUZZ-IEEE55066.2022.9882881}}
```

3. P. Foley et al., "OpenFL: the open federated learning library", Physics in Medicine & Biology (2022). doi:10.1088/1361-6560/ac97d9.
```
@article{openfl_citation,
	author={Foley, Patrick and Sheller, Micah J and Edwards, Brandon and Pati, Sarthak and Riviera, Walter and Sharma, Mansi and Moorthy, Prakash Narayana and Wang, Shi-han and Martin, Jason and Mirhaji, Parsa and Shah, Prashant and Bakas, Spyridon},
	title={OpenFL: the open federated learning library},
	journal={Physics in Medicine \& Biology},
	url={http://iopscience.iop.org/article/10.1088/1361-6560/ac97d9},
	year={2022},
	doi={10.1088/1361-6560/ac97d9},
	publisher={IOP Publishing}
}
```
Authors would like to thank Intel® and, in particular, Ing. Dario Sabella for providing hardware support and for the fruitful discussions.

[inno]: https://www.innoradar.eu/innovation/45988
[CustomizationGuide]: Customization_Guide.md
[IllustrativeExample]: Illustrative_Example.md
[License]: LICENSE
[Barcena2022]: https://ieeexplore.ieee.org/document/9882881
[docker-engine-setup]: https://docs.docker.com/engine/install/ubuntu/#install-docker-engine 
[hexa]: https://hexa-x.eu/
[docker-docs]:https://docs.docker.com/get-started/
[docker-file]:https://docs.docker.com/engine/reference/builder/
[docker-compose]: https://docs.docker.com/compose/
[keel]:http://www.keel.es/
[openfl]: https://github.com/securefederatedai/openfl
[openfl-docs]: https://openfl.readthedocs.io/en/latest/index.html
