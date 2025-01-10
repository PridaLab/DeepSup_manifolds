# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:00:27 2022

@author: Usuario
"""
from sklearn import linear_model #For Wiener Filter and Wiener Cascade
from sklearn.svm import SVR #For support vector regression (SVR)
from sklearn.svm import SVC #For support vector classification (SVM)
import xgboost as xgboost #For xgboost

import numpy as np


DECODERS = dict()
def register(func):
    '''Register a function as a decoder'''
    DECODERS[func.__name__] = func
    return func

##################### WIENER FILTER ##########################
#Copied from https://github.com/KordingLab/Neural_Decoding (19/10/2021)
@register
class WienerFilterRegression(object):

    """
    Class for the Wiener Filter Decoder

    There are no parameters to set.

    This simply leverages the scikit-learn linear regression.
    """

    def __init__(self):
        return


    def fit(self,X_flat_train,y_train):

        """
        Train Wiener Filter Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        self.model=linear_model.LinearRegression() #Initialize linear regression model
        self.model.fit(X_flat_train, y_train) #Train the model


    def predict(self,X_flat_test):

        """
        Predict outcomes using trained Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted=self.model.predict(X_flat_test) #Make predictions
        return y_test_predicted
    
    
#################### WIENER CASCADE ##########################
#Copied from https://github.com/KordingLab/Neural_Decoding (19/10/2021)
@register
class WienerCascadeRegression(object):
    """
    Class for the Wiener Cascade Decoder

    Parameters
    ----------
    degree: integer, optional, default 3
        The degree of the polynomial used for the static nonlinearity
    """

    def __init__(self,degree=3):
         self.degree=degree


    def fit(self,X_flat_train,y_train):

        """
        Train Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        num_outputs=y_train.shape[1] #Number of outputs
        models=[] #Initialize list of models (there will be a separate model for each output)
        for i in range(num_outputs): #Loop through outputs
            #Fit linear portion of model
            regr = linear_model.LinearRegression() #Call the linear portion of the model "regr"
            regr.fit(X_flat_train, y_train[:,i]) #Fit linear
            y_train_predicted_linear=regr.predict(X_flat_train) # Get outputs of linear portion of model
            #Fit nonlinear portion of model
            p=np.polyfit(y_train_predicted_linear,y_train[:,i],self.degree)
            #Add model for this output (both linear and nonlinear parts) to the list "models"
            models.append([regr,p])
        self.model=models


    def predict(self,X_flat_test):

        """
        Predict outcomes using trained Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        num_outputs=len(self.model) #Number of outputs being predicted. Recall from the "fit" function that self.model is a list of models
        y_test_predicted=np.empty([X_flat_test.shape[0],num_outputs]) #Initialize matrix that contains predicted outputs
        for i in range(num_outputs): #Loop through outputs
            [regr,p]=self.model[i] #Get the linear (regr) and nonlinear (p) portions of the trained model
            #Predictions on test set
            y_test_predicted_linear=regr.predict(X_flat_test) #Get predictions on the linear portion of the model
            y_test_predicted[:,i]=np.polyval(p,y_test_predicted_linear) #Run the linear predictions through the nonlinearity to get the final predictions
        return y_test_predicted

##################### EXTREME GRADIENT BOOSTING (XGBOOST) ##########################
#Adapted from https://github.com/KordingLab/Neural_Decoding: added regularization and 
#fixed some deprecated parameters
@register
class XGBoostRegression(object):

    """
    Class for the XGBoost Decoder

    Parameters
    ----------
    max_depth: integer, optional, default=3
        the maximum depth of the trees

    num_round: integer, optional, default=300
        the number of trees that are fit

    eta: float, optional, default=0.3
        the learning rate

    gpu: integer, optional, default=-1
        if the gpu version of xgboost is installed, this can be used to select which gpu to use
        for negative values (default), the gpu is not used
    """

    def __init__(self,max_depth=3,num_round=300,eta=0.3,gpu=-1):
        self.max_depth=max_depth
        self.num_round=num_round
        self.eta=eta
        self.gpu=gpu

    def fit(self,X_flat_train,y_train):

        """
        Train XGBoost Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """
        num_outputs=y_train.shape[1] #Number of outputs
        #Set parameters for XGBoost
        param = {'objective': "reg:squarederror", #for linear output
            'eval_metric': "logloss", #loglikelihood loss
            'max_depth': self.max_depth, #this is the only parameter we have set, it's one of the way or regularizing
            'eta': self.eta,
            'seed': 2925, #for reproducibility
            }
        if self.gpu<0:
            param['nthread'] = -1 #with -1 it will use all available threads
        else:
            param['gpu_id']=self.gpu
            param['updater']='grow_gpu'

        models=[] #Initialize list of models (there will be a separate model for each output)
        for y_idx in range(num_outputs): #Loop through outputs
            '''
            dtrain = xgboost.DMatrix(X_flat_train, label=y_train[:,y_idx]) #Put in correct format for XGB
            bst = xgboost.train(param, dtrain, self.num_round) #Train model
            models.append(bst) #Add fit model to list of models
            '''
            #breakpoint()
            val_div = int(0.9*X_flat_train.shape[0])
            dtrain = xgboost.DMatrix(X_flat_train[:val_div,:], label=y_train[:val_div,y_idx]) #Put in correct format for XGB
            dval = [(xgboost.DMatrix(X_flat_train[val_div:,:], label=y_train[val_div:,y_idx]), 'eval')] #Put in correct format for XGB
            bst = xgboost.train(param, dtrain, self.num_round, evals=dval, early_stopping_rounds=20,verbose_eval=False) #Train model
            models.append(bst) #Add fit model to list of models
        self.model=models


    def predict(self,X_flat_test):

        """
        Predict outcomes using trained XGBoost Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        dtest = xgboost.DMatrix(X_flat_test) #Put in XGB format
        num_outputs=len(self.model) #Number of outputs
        y_test_predicted=np.empty([X_flat_test.shape[0],num_outputs]) #Initialize matrix of predicted outputs
        for y_idx in range(num_outputs): #Loop through outputs
            bst=self.model[y_idx] #Get fit model for this output
            y_test_predicted[:,y_idx] = bst.predict(dtest) #Make prediction
        return y_test_predicted

##################### SUPPORT VECTOR REGRESSION ##########################
#Copied from https://github.com/KordingLab/Neural_Decoding (19/10/2021)
@register
class SVRegression(object):
    
    """
    Class for the Support Vector Regression (SVR) Decoder
    This simply leverages the scikit-learn SVR

    Parameters
    ----------
    C: float, default=3.0
        Penalty parameter of the error term

    max_iter: integer, default=-1
        the maximum number of iteraations to run (to save time)
        max_iter=-1 means no limit
        Typically in the 1000s takes a short amount of time on a laptop
    """

    def __init__(self,max_iter=-1,C=3.0):
        self.max_iter=max_iter
        self.C=C
        return


    def fit(self,X_flat_train,y_train):

        """
        Train SVR Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        num_outputs=y_train.shape[1] #Number of outputs
        models=[] #Initialize list of models (there will be a separate model for each output)
        for y_idx in range(num_outputs): #Loop through outputs
            model=SVR(C=self.C, max_iter=self.max_iter) #Initialize SVR model
            model.fit(X_flat_train, y_train[:,y_idx]) #Train the model
            models.append(model) #Add fit model to list of models
        self.model=models


    def predict(self,X_flat_test):

        """
        Predict outcomes using trained SVR Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        num_outputs=len(self.model) #Number of outputs
        y_test_predicted=np.empty([X_flat_test.shape[0],num_outputs]) #Initialize matrix of predicted outputs
        for y_idx in range(num_outputs): #Loop through outputs
            model=self.model[y_idx] #Get fit model for that output
            y_test_predicted[:,y_idx]=model.predict(X_flat_test) #Make predictions
        return y_test_predicted


##################### SUPPORT VECTOR CLASSIFICATION ##########################
#Copied from https://github.com/KordingLab/Neural_Decoding (19/10/2021)
@register
class SVClassification(object):
    
    """
    Class for the Support Vector Classification (SVC) Decoder
    This simply leverages the scikit-learn SVC

    Parameters
    ----------
    C: float, default=3.0
        Penalty parameter of the error term

    max_iter: integer, default=-1
        the maximum number of iteraations to run (to save time)
        max_iter=-1 means no limit
        Typically in the 1000s takes a short amount of time on a laptop
    """

    def __init__(self,max_iter=-1,C=3.0):
        self.max_iter=max_iter
        self.C=C
        return


    def fit(self,X_flat_train,y_train):

        """
        Train SVR Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        num_outputs=y_train.shape[1] #Number of outputs
        models=[] #Initialize list of models (there will be a separate model for each output)
        for y_idx in range(num_outputs): #Loop through outputs
            model=SVC(C=self.C, max_iter=self.max_iter) #Initialize SVR model
            model.fit(X_flat_train, y_train[:,y_idx]) #Train the model
            models.append(model) #Add fit model to list of models
        self.model=models


    def predict(self,X_flat_test):

        """
        Predict outcomes using trained SVR Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        num_outputs=len(self.model) #Number of outputs
        y_test_predicted=np.empty([X_flat_test.shape[0],num_outputs]) #Initialize matrix of predicted outputs
        for y_idx in range(num_outputs): #Loop through outputs
            model=self.model[y_idx] #Get fit model for that output
            y_test_predicted[:,y_idx]=model.predict(X_flat_test) #Make predictions
        return y_test_predicted

##################### ALLIASES ##########################
@register
def wf(*args, **kwargs):
    return WienerFilterRegression(*args, **kwargs)

@register
def wc(*args, **kwargs):
    return WienerCascadeRegression(*args, **kwargs)

@register
def xgb(*args, **kwargs):
    return XGBoostRegression(*args, **kwargs) 

@register
def svr(*args, **kwargs):
    return SVRegression(*args, **kwargs)

@register
def svc(*args, **kwargs):
    return SVClassification(*args, **kwargs)