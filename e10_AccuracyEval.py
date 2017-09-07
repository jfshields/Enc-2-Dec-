import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #R2 metric and TF

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn import cross_validation as sklearn_cv
from sklearn import gaussian_process
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.datasets import fetch_mldata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def calc_ActualArrearsStart(MX_pred_test, ABScalc_out_test):
    #  Returns pred_ActualArrearsStart based on pred_ABS and loan features as below
    #  ABScalc_out_tXXXX 2: ScoreStart
    #  ABScalc_out_tXXXX 3: ActualArrearsStart
    #  ABScalc_out_tXXXX 4: PaymentDueUsed
    #  ABScalc_out_tXXXX 5: DeltaDaysUsed
    MX_pred_ActualArrearsStart= np.multiply(np.divide(MX_pred_test, ABScalc_out_test[:, -12:, 5]), ABScalc_out_test[:, -12:, 4])* 30
    actu_ActualArrearsStart = ABScalc_out_test[:, -12:, 2]
    return MX_pred_ActualArrearsStart, actu_ActualArrearsStart

def calc_ArrearsAccuracy(MX_pred_ActualArrearsStart, actu_ActualArrearsStart):
    diff= np.abs(actu_ActualArrearsStart - MX_pred_ActualArrearsStart) / MX_pred_ActualArrearsStart
    return diff

def calc_Coverage(MX_pred_test, actu_ABSStart):
    n_samples= MX_pred_test.shape[0]* MX_pred_test.shape[1]
    n_b1 = float((np.abs(MX_pred_test - actu_ABSStart)< 0.23).sum())/ n_samples
    n_b2 = float((np.abs(MX_pred_test - actu_ABSStart)< 0.47).sum())/ n_samples
    n_b3 = float((np.abs(MX_pred_test - actu_ABSStart)< 0.70).sum())/ n_samples
    n_b4 = float((np.abs(MX_pred_test - actu_ABSStart)< 0.93).sum())/ n_samples
    return n_b1, n_b2, n_b3, n_b4
    
def eval_ModelAll(pred_Model_test, true_Actua_test, ABScalc_out_test):
    true_Actua_test_r = np.reshape(true_Actua_test, (-1, 12))
    pred_Model_test_r = np.reshape(pred_Model_test, (-1, 12))

    print('MSE', mean_squared_error(true_Actua_test_r, pred_Model_test_r))
    print('MAE', mean_absolute_error(true_Actua_test_r, pred_Model_test_r))
    print('R^2', r2_score(true_Actua_test_r, pred_Model_test_r))

    MX_pred_ActualArrearsStart, actu_ActualArrearsStart = calc_ActualArrearsStart(pred_Model_test_r, ABScalc_out_test)
    diff = calc_ArrearsAccuracy(MX_pred_ActualArrearsStart, actu_ActualArrearsStart)
    print('%_diff', np.mean(np.ma.masked_invalid(diff)))

    print('All  1W: %s, 2W: %s, 3W: %s, 4W: %s' % (calc_Coverage(true_Actua_test_r.reshape(-1, 12), pred_Model_test_r.reshape(-1, 12))))
    print('1st 1W: %s, 2W: %s, 3W: %s, 4W: %s' % (calc_Coverage(true_Actua_test_r[:, 1].reshape(-1, 1), pred_Model_test_r[:, 1].reshape(-1, 1))))
    print('4th 1W: %s, 2W: %s, 3W: %s, 4W: %s' % (calc_Coverage(true_Actua_test_r[:, 4].reshape(-1, 1), pred_Model_test_r[:, 4].reshape(-1, 1))))
    print('8th 1W: %s, 2W: %s, 3W: %s, 4W: %s' % (calc_Coverage(true_Actua_test_r[:, 8].reshape(-1, 1), pred_Model_test_r[:, 8].reshape(-1, 1))))
    print('12th 1W: %s, 2W: %s, 3W: %s, 4W: %s' % (calc_Coverage(true_Actua_test_r[:, -1].reshape(-1, 1), pred_Model_test_r[:, -1].reshape(-1, 1))))

def eval_ModelAll2(pred_Model_test, true_Actua_test, ABScalc_out_test):
    true_Actua_test_r = np.reshape(true_Actua_test, (-1, 12))
    pred_Model_test_r = np.reshape(pred_Model_test, (-1, 12))
    
    MAEall= mean_absolute_error(true_Actua_test_r, pred_Model_test_r)
    MSEall= mean_squared_error(true_Actua_test_r, pred_Model_test_r)
    all_b1, all_b2, all_b3, all_b4= calc_Coverage(true_Actua_test_r.reshape(-1, 12), pred_Model_test_r.reshape(-1, 12))
    
    MAEn1= mean_absolute_error(true_Actua_test_r[:, 0], pred_Model_test_r[:, 0])
    MSEn1= mean_squared_error(true_Actua_test_r[:, 0], pred_Model_test_r[:, 0])
    n1_b1, n1_b2, n1_b3, n1_b4= calc_Coverage(true_Actua_test_r[:, 0].reshape(-1, 1), pred_Model_test_r[:, 0].reshape(-1, 1))
    
    MAEn4= mean_absolute_error(true_Actua_test_r[:, 3], pred_Model_test_r[:, 3])
    MSEn4= mean_squared_error(true_Actua_test_r[:, 3], pred_Model_test_r[:, 3])
    n4_b1, n4_b2, n4_b3, n4_b4= calc_Coverage(true_Actua_test_r[:, 3].reshape(-1, 1), pred_Model_test_r[:, 3].reshape(-1, 1))

    MAEn8= mean_absolute_error(true_Actua_test_r[:, 7], pred_Model_test_r[:, 7])
    MSEn8= mean_squared_error(true_Actua_test_r[:, 7], pred_Model_test_r[:, 7])
    n8_b1, n8_b2, n8_b3, n8_b4= calc_Coverage(true_Actua_test_r[:, 7].reshape(-1, 1), pred_Model_test_r[:, 7].reshape(-1, 1))

    MAEn12= mean_absolute_error(true_Actua_test_r[:, 11], pred_Model_test_r[:, 11])
    MSEn12= mean_squared_error(true_Actua_test_r[:, 11], pred_Model_test_r[:, 11])
    n12_b1, n12_b2, n12_b3, n12_b4= calc_Coverage(true_Actua_test_r[:, 11].reshape(-1, 1), pred_Model_test_r[:, 11].reshape(-1, 1))

    print(MSEall, MAEall, all_b1, all_b2, all_b3, all_b4, 0
         , MSEn1, MAEn1, n1_b1, n1_b2, n1_b3, n1_b4, 0
         , MSEn4, MAEn4, n4_b1, n4_b2, n4_b3, n4_b4, 0
         , MSEn8, MAEn8, n8_b1, n8_b2, n8_b3, n8_b4, 0
         , MSEn12, MAEn12, n12_b1, n12_b2, n12_b3, n12_b4)
