import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
import time
from pprint import pprint


class stepwiseregression:

    def __init__(self, direction:str, criterion:str, threshold:float=None, trace:bool=True):

        self.direction = direction
        self.threshold = threshold
        self.criterion = criterion
        self.trace = trace

    def fit(self, X, Y ):

        self.X = X
        self.Y = Y
        self.__init_model()
        self.step_history = []
        while True:

            removed_param = None
            inserted_param = None
            if self.direction in ["forward", "mixed"]:
                inserted_param = self.__forward_step()
                if inserted_param is not None:
                    step_info = {'Added': inserted_param,
                                 'Removed': '',
                                 'R2': round(self.model.rsquared, 3),
                                 'R2_adj': round(self.model.rsquared_adj, 3),
                                 'AIC': round(self.model.aic, 3),
                                 'BIC': round(self.model.bic, 3),
                                 'F-stat': round(self.model.fvalue, 3)}

                    self.step_history.append(step_info)
            if self.direction in ["backward", "mixed"]:
                removed_param = self.__backward_step()
                if removed_param is not None:
                    step_info = {'Added': '',
                                 'Removed': removed_param,
                                 'R2': round(self.model.rsquared, 3),
                                 'R2_adj': round(self.model.rsquared_adj, 3),
                                 'AIC': round(self.model.aic, 3),
                                 'BIC': round(self.model.bic, 3),
                                 'F-stat': round(self.model.fvalue, 3)}

                    self.step_history.append(step_info)

            # model could not be further imporoved by removing or inserting parameter
            if removed_param is None and inserted_param is None:
                break

        # detect multicollinearty
        VIF_values = []
        for i, __ in enumerate(self.model_parameters):
            tmp_VIF_value = VIF(self.X[self.model_parameters].values, i)
            VIF_values.append(tmp_VIF_value)

        while any([val > 10 for val in VIF_values]):
            idx_max_VIF = np.argmax(VIF_values)
            param_max_vif = self.model_parameters[idx_max_VIF]
            self.model_parameters.remove(param_max_vif)

            VIF_values = []
            for i, __ in enumerate(self.model_parameters):
                tmp_VIF_value = VIF(self.X[self.model_parameters].values, i)
                VIF_values.append(tmp_VIF_value)

        updated_model = self.__fit(self.X[self.model_parameters], self.Y)
        self.model = updated_model

        self.step_history = pd.DataFrame(self.step_history)
        if self.trace:
            pprint(self.step_history)
            pprint(self.model_parameters)

    def predict(self, X):

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(data=X, columns=self.all_params)

        if "Intercept" not in X:
            X["Intercept"] = 1

        y_pred = self.model.predict(X[self.model_parameters])
        return y_pred.values

    def __init_model(self):

        # add an intercept to the data
        self.__add_intercept()
        # intialize model with only an intercept
        if self.direction == 'forward' or self.direction == 'mixed':
            self.model_parameters = ['Intercept']
            self.model = sm.OLS(self.Y, self.X["Intercept"]).fit()

        # initialize model with all parameters
        else:
            self.model_parameters = ['Intercept'] + self.all_params
            self.model = sm.OLS(self.Y, self.X).fit()

        if self.criterion == "BIC":
            self.current_best_criterion = self.model.bic
        elif self.criterion == "AIC":
            self.current_best_criterion = self.model.aic

    def __add_intercept(self):

        if isinstance(self.X, pd.DataFrame):
            tmp_x = self.X.copy()
            tmp_x['Intercept'] = 1
            self.X = tmp_x
        else:
            intercept  = np.array([1]*len(self.Y))[:, None]
            self.X = np.hstack(intercept, self.X)

    def __forward_step(self):

        crit_values = {}
        models = {}
        for param in self.all_params:
            if param in self.model_parameters:
                continue

            tmp_X_pred = self.X[self.model_parameters + [param]]
            model = self.__fit(tmp_X_pred, self.Y)
            crit_val = self.__get_criterion_value(model)

            crit_values[param] = crit_val
            models[param] = model

        best_param = self.__eval_criterion_values(crit_values)

        if best_param is not None:
            updated_model = models[best_param]
            self.model = updated_model
            self.model_parameters.insert(len(self.model_parameters), best_param)

        return best_param

    def __backward_step(self):

        crit_values = {}
        models = {}

        if self.criterion in ["AIC", "BIC"]:
            for param in self.all_params:

                if not (param in self.model_parameters):
                    continue
                tmp_params = self.model_parameters.copy()
                tmp_params.remove(param)
                tmp_X_pred = self.X[tmp_params]
                model = self.__fit(tmp_X_pred, self.Y)
                crit_val = self.__get_criterion_value(model)

                crit_values[param] = crit_val
                models[param] = model

            removed_param = self.__eval_criterion_values(crit_values)
            if removed_param is not None:
                updated_model = models[removed_param]
                self.model = updated_model
                self.model_parameters.remove(removed_param)

        elif self.criterion == "pval":
            coef_pval = self.model.pvalues.values[1:]
            insign_coef = coef_pval > self.threshold
            print([round(pval, 3) for pval in coef_pval])
            print(insign_coef)
            if any(insign_coef):
                endo_vars = self.model_parameters[1:]
                print(endo_vars)
                removed_param = endo_vars[np.argmax(coef_pval)]
                print(removed_param)
                self.model_parameters.remove(removed_param)
                updated_model = self.__fit(self.X[self.model_parameters], self.Y)
                self.model = updated_model

            else:
                removed_param = None


        return removed_param

    def __fit(self, X, Y):

        return sm.OLS(Y, X).fit()

    def __get_criterion_value(self, model):

        AIC = model.aic
        BIC = model.bic
        pval = model.pvalues.values[-1]

        if self.criterion == 'AIC':
            return AIC
        elif self.criterion == 'BIC':
            return BIC
        elif self.criterion == 'pval':
            return pval

    def __eval_criterion_values(self, values):

        best_param = None
        crit_values = list(values.values())
        evaluated_params = list(values.keys())
        if self.criterion in ["AIC", "BIC"]:
            model_improvement_list = [val < self.current_best_criterion for val in crit_values]

            if any(model_improvement_list):
                idx_best = int(np.argmin(crit_values))
                best_criterion = np.min(crit_values)
                self.current_best_criterion = best_criterion

                best_param = evaluated_params[idx_best]

        elif self.criterion == "pval":
            sign_params = [val < self.threshold for val in crit_values]
            if any(sign_params):
                idx_best = int(np.argmin(crit_values))
                best_param = evaluated_params[idx_best]

        return best_param

    @property
    def X(self):
        return self._X


    @X.setter
    def X(self, X):

        # check the instance of endogenous data
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError(f"X is {type(X)} but should be of type {np.ndarray} or {pd.DataFrame}")

        # check that at least two endogenous variables are used and that therre are more than two observations
        n_row, n_col = X.shape
        if n_row <= 2:
            raise ValueError(f"X has only {n_row} observations, X should have at least 3 observations")

        if n_col == 1:
            raise ValueError(f"X has only {n_col} variables, X should have at least 2 columns")

        # convert X to a dataframe for the ease of use
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=list(range(X.shape[1])))

        self.all_params = X.columns.tolist()
        if "Intercept" in self.all_params:
            self.all_params.remove('Intercept')

        self._X = X

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, Y):

        # check the instance of exogenous data
        if not isinstance(Y, (np.ndarray, pd.DataFrame)):
            raise ValueError(f"Y is {type(Y)} but should be of type {np.ndarray} or {pd.DataFrame}")

        # get shape of the exogenous variable
        n_row, n_col= Y.shape
        if n_row != self.X.shape[0]:
            raise ValueError(f"Y has not the same number of observations ({n_row}) as X")

        if n_col != 1:
            raise ValueError(f"Shape of Y is {(n_row, n_col)} but should be {(self.X.shape[0], 1)}")

        self._Y = Y

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, criterion):
        if criterion not in ["AIC", "BIC", "pval"]:
            raise ValueError(f"The criterion should be AIC, BIC, pval")
        if criterion == "pval":
            if self.threshold is None:
                raise ValueError(f"When the pval criterion is chosen, a threshold value should be provided")

        self._criterion = criterion
