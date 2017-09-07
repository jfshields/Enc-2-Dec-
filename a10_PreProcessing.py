import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #R2 metric and TF
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn import cross_validation as sklearn_cv
from sklearn import gaussian_process
from sklearn import preprocessing
from sklearn.utils import shuffle
from scipy import interp, arange, exp

def data_import():
    df_sub1 = pd.read_csv('/home/#/Weekly_Chain_2017-06-29_1.csv')
    df_sub2 = pd.read_csv('/home/#/Weekly_Chain_2017-06-29_2.csv')
    df_sub3 = pd.read_csv('/home/#/Weekly_Chain_2017-06-29_3.csv')
    df_sub4 = pd.read_csv('/home/#/Weekly_Chain_2017-06-29_4.csv')
    print('Length 1: %s, 2: %s, 3: %s, 4: %s' % (len(df_sub1), len(df_sub2), len(df_sub3), len(df_sub4)))
    
    df_feat = pd.read_csv('/home/#/LoanCust_feat2.csv')

    df_all= df_sub1.append(df_sub2, ignore_index=True)
    df_all= df_all.append(df_sub3, ignore_index=True)
    df_all= df_all.append(df_sub4, ignore_index=True)
    print('Length all:', len(df_all))
    
    df_all_feat= pd.merge(df_all, df_feat, left_on='LoanId', right_on='LSSLoanID', how='left')
    return df_all_feat

def data_year_fil(df_in, YYYY= 2015):
    df_in['DueDateDT'] = pd.to_datetime(df_in['DueDate'], format='%Y-%m-%d')
    df_subYYYY = df_in[(df_in['DueDate'] >= (str(YYYY)+ '-01-01')) & (df_in['DueDate'] < (str(YYYY+ 1)+ '-01-01'))]
    return df_subYYYY

def data_subweekly_g36(df_weekly):
    #2 Add entry count on LoanId
    df_entry_count= pd.DataFrame()
    df_entry_count['entry_count']= df_weekly['LoanId'].value_counts()
    df_entry_count['LoanId'] = df_entry_count.index
    df_weekly= pd.merge(df_weekly, df_entry_count, on='LoanId', how='left')
    df_weekly_g36W= df_weekly.loc[df_weekly['entry_count'] >= 36]
    return df_weekly_g36W

def data_dummies(df_all):
    # Get dummies
    df_all['Calc_Wave']= np.where(df_all['Wave']<= 4, df_all['Wave'], 5)
    df_all['DueDateYY']= df_all['DueDate'].str[2: 4]
    df_all['DueDateMM']= df_all['DueDate'].str[5: 7]
    df_all['DueDateD1']= df_all['DueDate'].str[8: 9]

    df_all= pd.concat([df_all, pd.get_dummies(df_all['Calc_Wave'])], axis= 1)
    df_all= pd.concat([df_all, pd.get_dummies(df_all['LoanApplicationType'])], axis= 1)
    df_all= pd.concat([df_all, pd.get_dummies(df_all['LoanType'])], axis= 1)
    df_all= pd.concat([df_all, pd.get_dummies(df_all['LoanPricingLevel'])], axis= 1)
    df_all= pd.concat([df_all, pd.get_dummies(df_all['MobileAppUser'])], axis= 1)
    df_all= pd.concat([df_all, pd.get_dummies(df_all['MaritalStatus_1'])], axis= 1)
    df_all= pd.concat([df_all, pd.get_dummies(df_all['TypeOfBankAccount_1'])], axis= 1)
    df_all= pd.concat([df_all, pd.get_dummies(df_all['NameOfBank_1'])], axis= 1)

    df_all= pd.concat([df_all, pd.get_dummies(df_all['DueDateD1'])], axis= 1)
    df_all= pd.concat([df_all, pd.get_dummies(df_all['DueDateMM'])], axis= 1)
    df_all= pd.concat([df_all, pd.get_dummies(df_all['DueDateYY'])], axis= 1)
    return df_all

def scale_data(x_nstd):
    scaler= preprocessing.StandardScaler().fit(x_nstd)
    x_std= scaler.transform(x_nstd)
    return x_std, scaler

def data_in_out_all(df_in, k=24, j=12, standardise= 'Yes', shuffle_= 'Yes'):
    y = []
    t = []
    l = []
    x = []
    ABScalc = []
    er1= []
    er2= []
    
    print('Loading')

    for i, row in df_in.iterrows():
        l_temp0 = df_in[i: i+ k+ j]['JourneyId'].values
        t_temp0 = df_in[i: i+ k+ j]['JourneyRowNum'].values

        if (len(set(l_temp0)) == 1) and (1 not in t_temp0[1:]):
            y_temp = df_in[(i+ k): (i+ k+ j)]['ScoreStart'].values
            if y_temp.shape[0] == j:
                l.append(l_temp0[0])
                y.append(y_temp)
                x_temp = df_in[i: i+ k][['ScoreStart'
                                       , 'RowNum'
                                       , 'ReverseRowNum'
                                       , 'TotalDueToDate'
                                       , 'TotalPaidStart'
                                       , 'LatePayment'
                                       , 'DueDayPayment'
                                       , 'TotalPaidEnd'
                                       , 'LoanAmount_y'
                                       , 'UsingCAT'
                                       , 'DailyDeclineCount'
                                       , 'DailySuccessCount'
                                       , 'AppRegister'
                                       , 'AppUsage'
                                       , 'CustomerCall'
                                       , 'StoreCall'
                                       , 'StartStatusID'
                                       , 'CAT_Indicator'
                                       , 'LoanTerm'

                                        ,'RefinanceCounter'
                                        ,'C_DateOfBirthMM'
                                        ,'C_CustomerStartDateMM'
                                        ,'NumberOfDependents_1'
                                        ,'ResidentInUKMonths_1'
                                        ,'HasBankAccount_1'

                                        , 0, 1, 2, 3, 4, 5, 'NFC Application',
                                        'TU Application', 'Wave0 Application', 'New', 'New from Closed',
                                        'Top up', 'Bronze', 'Gold', 'Silver', 'Unknown', 'Write-Off',
                                        'Mobile App User', 'n/a', 'Divorced', 'Married', 'Other', 'Single',
                                        'C/A', 'CA', 'CURRENT A/C', 'Current account', 'Visa debit',
                                        'current', 'current acc', 'BARCLAYS BANK PLC', 'Barclays',
                                        'CO-OPERATIVE BANK', 'HSBC Bank Plc', 'Halifax', 'LLOYDS BANK PLC',
                                        'LLOYDS TSB BANK PLC', 'METRO BANK', 'NAT WEST BANK PLC',
                                        'NATIONWIDE BLDG SCTY', 'ROYAL BANK OF SCOT', 'Santander',
                                        'TSB BANK PLC', '0', '1', '2', '3', '01', '02', '03', '04', '05',
                                        '06', '07', '08', '09', '10', '11', '12', '11', '12', '13', '14',
                                        '15', '16', '17']].values
                x.append(x_temp)

                t_temp = df_in[i: i+ k+ j]['JourneyRowNum'].values
                t.append(t_temp)

                ABScalc_temp = df_in[i: i+ k+ j][['JourneyId'
                        , 'JourneyRowNum'
                        , 'ScoreStart'
                        , 'ActualArrearsStart'
                        , 'PaymentDueUsed'
                        , 'DeltaDaysUsed']].values
                ABScalc.append(ABScalc_temp)

            else:
                er1.append(x_temp)
        else:
            er2.append(x_temp)

    x_out= np.asarray(x)
    y_out= np.asarray(y)
    t_out= np.asarray(t)
    loadID_out = np.asarray(l)
    ABScalc_out = np.asarray(ABScalc)

    if shuffle_== 'Yes':
        x_out, y_out, t_out, loadID_out, ABScalc_out = shuffle(x_out, y_out, t_out, loadID_out, ABScalc_out, random_state=0)

    dic_scaler= {}
    if standardise == 'Yes':
        x_out= np.nan_to_num(x_out)
        for i in range(25):
            _, dic_scaler[i]= scale_data(x_out[:, :, i].reshape(-1, 1))
            x_out[:, :, i]= dic_scaler[i].transform(x_out[:, :, i])
    else:
        pass

    print('Data shape:', x_out.shape, y_out.shape, t_out.shape, loadID_out.shape, ABScalc_out.shape)
    return x_out, y_out.reshape(-1, j, 1), t_out, loadID_out, ABScalc_out, dic_scaler
