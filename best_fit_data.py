import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

class Fitter:

    def __init__(self):
        self.__xvals = []
        self.__yvals = []
        self.__fig, self.__ax = plt.subplots()
    
    
    def set_initial_data(self, x, y):
        if len(x) == len(y):
            self.__xvals = np.array(x)
            self.__yvals = np.array(y)
        else:
            print("Data set length not consistent")


    def __f(x, a, b, c=0, d=0, e=0, f=0):
            return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5  
    

    def generate_fit_plot(self, xrange, max_poly=1, include_sine=False):
        param_len = max_poly + 1
        self.__ax.scatter(self.__xvals, self.__yvals, color="tab:grey", label="sample")
        self.__ax.grid(True)

        lineform = "y = {} + ({})x"
        for i in range(1, max_poly+1):
            popt, pcov, infoDic, msg, _ = curve_fit(Fitter.__f, self.__xvals, self.__yvals, p0=np.ones(i+1), full_output=True)
            self.__ax.plot(xrange, Fitter.__f(xrange, *popt), label=f"fit line (x^{i})")
            if i >= 2:
                lineform += " + ({})x^(" + str(i) + ")"
            print("Model line with parameters: " + lineform.format(*popt))
            print("Condition number of covariance matrix: " + str(np.linalg.cond(pcov)))
            print("Uncertainty of fit for parameters: ", np.diag(pcov))
            print("\n")

        self.__ax.legend()
        plt.show()


# Plug in the name of the file located in the sample_data folder
# Also specify the column that corresponds to the x-axis data and the y-axis data

filename = "trees.csv"
xdata_col_index = 1
ydata_col_index = 3

data_path = "/Users/ramoneagard0/MyProjects/python-tutorials/fit_data/sample_data/" + filename

dataframe = pd.read_csv(data_path, dtype=np.float64)
print(dataframe.head())
print(dataframe.info())
print()

y = dataframe.values[:, ydata_col_index]
x = dataframe.values[:, xdata_col_index]
z = dataframe.values[:, 2]

fitter = Fitter()

fitter.set_initial_data(x, y)
fitter.generate_fit_plot(np.linspace(min(x)-1., max(x)+1.), max_poly=4)