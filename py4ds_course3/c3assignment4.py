import pandas as pd
import numpy as np


def blight_model():

    #remember to get rid of readonly b4 your submission
    train = pd.read_csv('readonly/train.csv', encoding = 'ISO-8859-1')
    test = pd.read_csv('readonly/test.csv')
    location = pd.read_csv('readonly/latlons.csv')
    address = pd.read_csv('readonly/addresses.csv')

    train = train[~train['compliance'].isnull()]

    address_loc = pd.merge(address, location, on='address')
    train = pd.merge(train, address_loc, on='ticket_id')
    test = pd.merge(test, address_loc, on='ticket_id')

    list(train.columns)
    #drop the string list as much, as well as floats we want not

    train.drop(['agency_name', 'inspector_name', 'violator_name', 'violation_street_number', 'violation_street_name',
                'violation_zip_code', 'mailing_address_str_number', 'mailing_address_str_name', 'city', 'state', 'zip_code',
                'non_us_str_code', 'country', 'ticket_issued_date', 'hearing_date', 'violation_code', 'violation_description', 'disposition',
                'balance_due', 'payment_date', 'payment_status', 'collection_status', 'grafitti_status', 'compliance_detail',
                'address', 'lat', 'lon'], axis=1, inplace=True)
    #error with something extra for test frame
    test.drop(['agency_name', 'inspector_name', 'violator_name', 'violation_street_number', 'violation_street_name',
                'violation_zip_code', 'mailing_address_str_number', 'mailing_address_str_name', 'city', 'state', 'zip_code',
                'non_us_str_code', 'country', 'ticket_issued_date', 'hearing_date', 'violation_code', 'violation_description', 'disposition',
                'grafitti_status', 'address', 'lat', 'lon'], axis=1, inplace=True)

    #(list(train.columns), list(test.columns)), I just check that payment amount is the one i havent cleared with

    #Checked logistic regression for predict with auc score 0.75329... approved!
    #here is the check code

    #y = train['compliance']
    #train.drop(['payment_amount', 'compliance'], axis=1, inplace=True)
    #X = train
    #X = X.fillna('')

    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    #from sklearn.linear_model import LogisticRegression
    #lr = LogisticRegression().fit(X_train, y_train)
    #y_predict = lr.predict_proba(X_test)[:,1]

    #from sklearn.metrics import roc_curve, auc
    #fpr_lr, tpr_lr, _ = roc_curve(y_test, y_predict)
    #roc_auc_grd = auc(fpr_lr, tpr_lr)

    #return (roc_auc_grd)

    y_train = train['compliance']
    train.drop(['payment_amount', 'compliance'], axis=1, inplace=True)
    X_train = train
    X_train = X_train.fillna('')
    X_test = test
    X_test = X_test.fillna('')

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression().fit(X_train, y_train)
    y_predict = lr.predict_proba(X_test)[:,1]

    test['compliance'] = y_predict
    test.set_index('ticket_id', inplace=True)
    y = test['compliance']

    return (y)

blight_model()
