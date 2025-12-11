# [NYC Restaurant Inspection Results (DOHMH)](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j)
## By Jolie Ng, Rameez Rauf and Yuthi Madireddy

## Goal:
The goal of this project is to build and compare multiple machine learning models to predict New York City restaurant health inspection outcomes. Our team applied penalized linear models, ensemble methods, support vector machines, and neural networks to understand which approaches work best and what features drive predictive performance.

## Section 1:
To run the notebooks in this repository, you will need:

**Core Tools**
- Python 3.10+
- Jupyter Notebook / JupyterLab

**Required Python Packages**
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pyarrow
- imbalanced-learn
- torch

**Operating System**
- Compatible with macOS, Windows, or Linux


## Section 2:
ds_4022_final_project/

- README.md
- final_report.pdf
- Notebooks/
	- 01_descriptive_analysis.ipynb
	- 02_penalized_linear_model.ipynb
	- 03_ensemble_model.ipynb
	- 04_neural_network_model.ipynb
	- 05_svm_model.ipynb
 	- 06_model_comparison.ipynb
- Data/
	- train_NYC_inspection.parquet
	- test_NYC_inspection.parquet
 	- raw/
 		- data_split.py

- Outputs/
	- EDA/
 		- grade_distribution.png
 		- nyc_numerical_histogram.png
	- figures/
 		- final_test_confusion_matrix.png
 		- tinal_test_confusion_matrix.png
 		- linear_svc_confusion_matrix.png
 		- logreg_balanced_confusion.png
