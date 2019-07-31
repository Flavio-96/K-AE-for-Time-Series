# Time Series K-Means Clustering with Autoencoder Feature Extraction
This is a class project for Data Science and Machine Learning course of Università di Salerno Computer Science Master degree.
In this repo there are two Jupyter Nootebook in which are experimented two clustering approaches for time series.
One relies on an autoencorder to extract TS features an then cluster them with K-Means.
The other uses [TSLearn](https://tslearn.readthedocs.io/en/latest/index.html) for [DTW](https://en.wikipedia.org/wiki/Dynamic_time_warping)  based K-Means.

## Installation
### Prerequisites
Python 3 is required. If you want to run TSLearn notebook, also [C++ Build Tools](https://visualstudio.microsoft.com/it/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15) are needed.
### Steps
1. Download the repo;
2. Go in the repo source folder
3. (Optional) Install a virtual envirorment
4. Run <code>pip install -r requirements.txt</code>
5. Run <code>jupyter notebook</code> to open Jupyter
6. Notebooks are located in Run <code>jupyter/</code> folder

## Used Datasets

Time series are **not multivariate**
- fordA
- fordB
- ECG5000
- ECG200
- phalangesOutlinesCorrect
- TwoPatterns
- ChorelineConcetration
- refrigerationDevice
- TwoLeadECG

## Feature selection in unsupervised learning
[scikit-feature](https://github.com/jundongl/scikit-feature)
