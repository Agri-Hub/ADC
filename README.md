# Agriculture Monitoring Data Cube (ADC)
A Data Cube of Big Satellite Image Time-Series for Agriculture Monitoring

[![PyPI - Python](https://img.shields.io/pypi/pyversions/iconsdk?logo=pypi)](https://pypi.org/project/iconsd)
![alt text](https://img.shields.io/badge/database-postgis-orange)
## Table of Contents
- [Introduction](#Introduction)
- [Features](#Features)
- [Who uses ADC](#Who_uses_ADC)
- [How it Works](#How_it_Works)
- [Usage](#Usage)
- [Configuration](#Configuration)

## Introduction
The Agriculture monitoring Data Cube (ADC) is an automated, modular, end-to-end framework for discovering, pre-processing and indexing optical and Synthetic Aperture Radar (SAR) images into a multidimensional cube. ADC is based on the Open Data Cube framework, but it also comes with a set of powerful functionalities on top of it enhancing the validation process during CAP monitoring and extracting knowledge from the SITS analyses and the machine learning tasks .Thus, ADC can be used for building scalable country-specific knowledge bases that can efficiently answer complex and multi-faceted geospatial queries. 

## Features
A few things that you can do with ADC:
- Generation of analysis-ready feature spaces of big satellite data to feed downstream machine learning tasks
- Support of Satellite Image Time-Series (SITS) analysis via services pertinent to the monitoring of the CAP
- Spatial Buffering
- Smart Multidimensional Queries

![alt text](https://i.ibb.co/KXBV7dP/ts-preprocessing-2.png)

## Who uses ADC
- Paying Agencies
- Data Scientists

## How it Works
The ADC can be setup and deployed in any environment. Our solution is hosted in the Creo-DIAS cloud plaform allowing the direct access to satellite data. Currently, ADC is configured to monitor at national scale two countries, Lithuania and Cyprus. 
- Dedicated ADC back-end processes are automatically discover new acquired Sentinel-1 and Sentinel-2 products for these two countries, pre-process these images and generate Analysis-Ready-Data (ARD)
- The ARD are then indexed into the ADC
- Users have the potential to make usage of the on-the-top tools that have been implemented and part of them is presented to this git. 

## Usage
- [Example 1: Feature Generation](https://github.com/Agri-Hub/ADC/blob/main/Examples/01_ADC_Feature_Space_Generation.ipynb)
- [Example 2: Photointerpretation via Satellite Time Series Analysis](https://github.com/Agri-Hub/ADC/blob/main/Examples/02_ADC_Photo_Interpretation.ipynb)
- [Example 3: Smart Multidimensional Queries](https://github.com/Agri-Hub/ADC/blob/main/Examples/03_Smart_Multidimensional_Queries.ipynb)

```sh
data = get_data_for_validation(parcel=alerts[id_to_check],start_date='2019-01-01',end_date='2019-12-31',index='ndvi')
data_resampled = data.resample(time='15D').interpolate('linear')
```

## Configuration
It is recommended to follow the instructions related to the installation of the Open Data Cube and the construction of the yaml files as it has been presented in our previous work: https://github.com/Agri-Hub/datacap



## License
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)



