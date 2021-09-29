import sys
import joblib
import xgboost
import warnings
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def Model_XGBoost(X_regressor,Y_target,
                  colsample_bytree,gamma,learning_rate,depth,min_child,
                  n_estimators,reg_alpha,reg_lambda,subsample,seed):

    X_train, X_test, y_train, y_test = train_test_split(X_regressor,Y_target, test_size=0.2, random_state=0)
    
    model_XGB = xgboost.XGBRegressor(colsample_bytree=colsample_bytree,
                 gamma=gamma,                 
                 learning_rate=learning_rate,
                 max_depth=depth,
                 min_child_weight=min_child,
                 n_estimators=n_estimators,                                                                    
                 reg_alpha=reg_alpha,
                 reg_lambda=reg_lambda,
                 subsample=subsample,
                 seed=seed)
    
    model_XGB.fit(X_train, y_train)
    prediction = model_XGB.predict(X_test)
    
    df = pd.DataFrame()
    df = X_test.copy()
    df['price'] = y_test
    df['prediction'] = prediction
    
    if Y_target.name=='Precio/m2':
        prediction = df['prediction']*df['m2']
        y_validate = df['price']*df['m2']
    else:
        prediction = df['prediction']
        y_validate = df['price']
    
    MAE_score = mean_absolute_error(prediction,y_validate)
    MSE_score = np.sqrt(mean_squared_error(prediction,y_validate))
    print('En la colonia',name_colonia,'la diferencia es de',MAE_score,'+/-',MSE_score)
    
    absolute_variation = np.floor((abs(prediction/y_validate-1))*100)

    bins = [-0.1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,10000]
    S = pd.cut(absolute_variation, bins)
    Y_exe = S.value_counts().reindex(S.cat.categories)
    fig = plt.figure(figsize = (14, 7))
    plt.plot(range(0,11), Y_exe, 'ro-') 
    plt.xticks(range(0,12),bins) 
    plt.ylabel('Frecuencia')
    plt.xlabel('Rango en diferencia Porcentual')
    plt.show()
    
    return model_XGB
    
colonias_in = [365,363,305,452,148,46,392,87,440,444,436,20,319,339,196,438,15,454,204]
result={}
columns_order = ['m2', 'Rec', 'Est', 'Ba�']

for i in colonias_in:
    X_colonia = X[X['Colonia2']==i][{'m2', 'Rec', 'Est', 'Ba�'}]
    X_colonia = X_colonia.reindex(columns = columns_order)
    Y_colonia = Y['Precio/m2'][Y['Colonia2']==i]
    name_colonia = str((x["Colonia"][x['Colonia2']==i].iloc[1]))
    
    result[i]=Model_XGBoost(X_colonia,Y_colonia,0.4,0,0.07,6,1.5,10000,0.75,0.45,0.6,42)
    result
    
joblib.dump(result, 'modelo_entrenado_xgboost.pkl')
