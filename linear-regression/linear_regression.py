'''linear_regression.py
Subclass of Analysis that performs linear regression on data
Olivia Doherty
CS251 Data Analysis Visualization
Spring 2024
'''
import numpy as np
import scipy.linalg
import data
import matplotlib.pyplot as plt

from analysis import Analysis

class LinearRegression(Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable (true values) being predicted by linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean SEE. float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression by using Scipy to solve the least squares problem y = Ac
        for the vector c of regression fit coefficients. Don't forget to add the coefficient column
        for the intercept!
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var

        #ind and dep variables
        int_data = self.data.select_data(ind_vars)
        dep_data = self.data.select_data([dep_var])

        #create matrix a
        self.A = np.hstack((np.ones((len(int_data), 1)), int_data))

        #assin y
        self.y = dep_data

        #regression
        c, residuals, rank, s = np.linalg.lstsq(self.A, self.y, rcond=None)

        #slope & intercept
        self.intercept = float(c[0])
        self.slope = c[1:]

        #R^2 value
        y_pred = self.A @ c
        self.R2 = 1 - np.sum((self.y - y_pred) ** 2) / np.sum((self.y - np.mean(self.y)) ** 2)

        #residuals
        self.residuals = self.y - y_pred

        #MSE
        self.mse = np.mean(self.residuals ** 2)

    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''

        if X is None:
           X = self.A
        else:
            #add column of ones if necesasry
            X = np.hstack((np.ones((X.shape[0], 1)), X))

            if self.p > 1:
                X = self.make_polynomial_matrix(X, self.p)

            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        if X.shape[1] != self.slope.shape[0] + 1:  # +1 for intercept
            raise ValueError(f"Expected X to have {self.slope.shape[0] + 1} features, got {X.shape[1]}.")

        # Prediction
        y_pred = np.dot(X, np.vstack((self.intercept, self.slope)))
        return y_pred.reshape(-1, 1)
  

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        y_true = self.y
        mean_y = np.mean(y_true)

        #total sum of squares
        ss_total = np.sum((y_true - mean_y) ** 2)

        #residual sum of squares
        ss_res = np.sum((y_true - y_pred) ** 2)

        #R^2 stat
        R2 = 1 - (ss_res / ss_total)
        return R2

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''

        #compute residuals
        self.residuals = self.y - y_pred
        return self.residuals

    def compute_mse(self):
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error

        Hint: Make use of self.compute_residuals
        '''

        #compute MSE
        self.compute_residuals(self.predict())
        mse = np.mean(self.residuals ** 2)
        return mse

    def scatter(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        
        #regression line
        x, y = Analysis.scatter(self, ind_var, dep_var, (title))
        x_min = min(x)
        x_max = max(x)
        x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)

        #support for polynomials
        if self.p > 1:
            x_poly = self.make_polynomial_matrix(x_line, self.p)
            r_poly = np.dot(x_poly, self.slope) + self.intercept
            plt.plot(x_line, r_poly, color='red')
        else:
            r_line = x_line *self.slope[0] + self.intercept
            plt.plot(x_line, r_line, color = 'red')

        #scatter plot
        plt.scatter(x, y)
        plt.title(f'{title} (R^2 = {self.R2:.2f})')
        plt.show()

    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''

        num_vars = len(data_vars)  # define num_vars based on the length of data_vars
        fig, axes = plt.subplots(num_vars, num_vars, figsize=fig_sz)
        
        for i, var_y in enumerate(data_vars):
            for j, var_x in enumerate(data_vars):
                if i == j and hists_on_diag:
                    # code for histograms on the diagonal
                    continue
                
                # use Data class method to get column data
                x_data = self.data.select_data([var_x])
                y_data = self.data.select_data([var_y])

                # scatter plot for variable pairs
                axes[i, j].scatter(x_data, y_data, alpha=0.5)
                
                # calculate and plot regression line
                self.linear_regression([var_x], var_y)  # fit model
                x_vals = np.array(axes[i, j].get_xlim())
                y_vals = self.intercept + self.slope[0] * x_vals
                axes[i, j].plot(x_vals, y_vals, color="red")  # regression line
                
                # set title with R^2 value
                axes[i, j].set_title(f'R^2: {self.R2:.2f}')
                
                # adjust axis labels visibility
                if j > 0:
                    axes[i, j].set_yticklabels([])
                if i < num_vars - 1:
                    axes[i, j].set_xticklabels([])
                if i == num_vars - 1:
                    axes[i, j].set_xlabel(var_x)
                if j == 0:
                    axes[i, j].set_ylabel(var_y)

        plt.tight_layout()
        plt.show()

    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        should take care of that.
        '''
        
        if not isinstance(p, int):
            raise ValueError("Degree 'p' must be an integer")
        
        powers = np.arange(1, p + 1)
        A_reshape = A.reshape(-1, 1)
        poly_matrix = np.hstack([A**i for i in range(1, p+1)])
        return poly_matrix

    def poly_regression(self, ind_var, dep_var, p):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and an added column of 1s for the intercept.

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''

        #regression variables
        self.p = p
        self.A = self.data.select_data([ind_var])
        self.y = self.data.select_data([dep_var])

        A_poly = self.make_polynomial_matrix(self.A, p)
        Ahat = np.hstack((np.ones((A_poly.shape[0], 1)), A_poly))

        c, residuals, rank, s = scipy.linalg.lstsq(Ahat, self.y)

        self.intercept = c[0]
        self.slope = c[1:]
        self.mse = np.mean(residuals**2)

        y_pred = Ahat @ c
        self.R2 = 1 - np.sum((self.y - y_pred) ** 2) / np.sum((self.y - np.mean(self.y)) ** 2)
        
    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope
        

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor. 
        '''
        self.ind_vars = ind_vars
        self.dep_vars = dep_var
        self.slope = slope
        self.intecept = intercept
        self.p = p
        self.poly_matrix = self.make_polynomial_matrix(self.data.get_all_data(), self.p)
