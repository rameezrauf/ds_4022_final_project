# NYC Restaurant Inspection Results (DOHMH) 
## By Jolie Ng, Rameez Rauf and Yuthi Madireddy

(Data Found at )[https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j]

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
Project-Folder/

- README.md
- final_report.pdf
- Notebooks/
	- 01_descriptive_analysis.ipynb
	- 02_penalized_linear_model.ipynb
	- 03_random_forest_model.ipynb
	- 04_neural_network_model.ipynb
	- 05_svm_model.ipynb
 	- 06_model_comparison.ipynb
- Data/
	- train_NYC_inspection.parquet
	- test_NYC_inspection.parquet
 	- (any additional cleaned or processed data set)

- Outputs/
	- figures
