# Machine Learning Engineer Nanodegree
# Capstone Project
## Project: Analysis and Classification of Crimes in Chicago

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

### Code

Code is provided in the `Chicago_crime.ipynb` notebook file. The `data.py`, `model.py`, and `vis.py` should be imported.

### Run

In a terminal or command window, navigate to the top-level project directory `finding_donors/` (that contains this README) and run one of the following commands:

```bash
ipython notebook finding_donors.ipynb
```  
or
```bash
jupyter notebook finding_donors.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Data

The crime data and shapefiles are imported from the Chicago Data Portal website  [UCI](https://data.cityofchicago.org/). Only crime data in 2015 and 2016 are used, with 510,070 data points and 21 columns.

**Target Variable**
- `Primary Type`: type of crime based on the IUCR code (the Illinois Uniform Crime Reporting code).
