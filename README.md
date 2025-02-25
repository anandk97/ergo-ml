# ergo-ml

Repository consisting of machine learning approaches for automated ergonomic assessment of hand-intensive manufacturing. This project leverages preprocessed force and motion data to analyze, predict, and visualize ergonomic scores, supporting research and development in ergonomic risk assessment.

---

## Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [Contributors and Contact Information](#contributors-and-contact-information)
- [Acknowledgements](#acknowledgements)

---

## Overview

The ergo-ml repository is organized into three main parts:
- **Annotation:** Methods to annotate ergonomic scores from the raw or preprocessed data.
- **Prediction:** Machine learning models and associated scripts for predicting ergonomic scores.
- **Visualization:** Tools and notebooks to visualize both the data and the results of the analysis.

In addition, the repository includes notebooks for exploratory data analysis (EDA) and hypothesis testing to better understand the data characteristics and validate assumptions.

---

## File Structure

Below is an overview of the individual files and folders included in the repository:

ergo-ml/ ├── BACH Score/
│ └── [Files related to the BACH scoring method]
│ - This folder contains scripts, notebooks, and data processing tools for computing the BACH Score, an alternative ergonomic scoring approach. │ ├── Ergonomic Score Annotation/
│ └── [Annotation modules and documentation]
│ - Contains code and documentation to support the annotation process. This module focuses on the labeling or scoring of ergonomic risk based on operator motion and force data. │ ├── Ergonomic Score Prediction/
│ └── [Prediction models and experiments]
│ - Houses machine learning models, training scripts, and experiment logs that predict ergonomic scores. The code here demonstrates various regression and classification approaches to assess ergonomic risk. │ ├── Ergonomic Score Visualization/
│ └── [Visualization scripts and notebooks]
│ - Provides scripts and notebooks to visualize both raw data and model outputs, aiding in the interpretation of ergonomic scores through plots, charts, and interactive dashboards. │ ├── EDA.ipynb
│ - A Jupyter Notebook for Exploratory Data Analysis (EDA) on the collected force and motion data. It includes data cleaning, statistical summaries, and initial visualizations to understand key trends. │ ├── Hypothesis_Testing.ipynb
│ - A Jupyter Notebook dedicated to performing hypothesis tests on the dataset. It explores statistical relationships and validates assumptions regarding ergonomic risk factors. │ └── README.md
- This file. It provides an overview of the repository structure, instructions for setup, usage guidelines, and other relevant information.


*Note:* The descriptions provided in the folder listings are indicative. For more detailed information, please refer to the documentation within each folder or the code comments.

## Contributors and Contact Information
The main contributors to this repository are Anand Krishnan (anandkrishnan6561@gmail.com) and Xingjian Yang (yxj1995@uw.edu)

## Acknowledgements
This work was supported by the Joint Center for Aerospace Technology Innovation (JCATI) and
the National Science Foundation AI Institute in Dynamic Systems (grant number 2112085).