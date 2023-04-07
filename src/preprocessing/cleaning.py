import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler

def remove_outliers(data):
  '''
  Gets dataframe as input (with only categorical and numerical columns)
  removes outliers, returns the updated dataframe.
  Used IQR method to detect the outlieres
  '''
  data_Categorical = data.select_dtypes(include=['object'])
  Categorical_Cols = data_Categorical.columns.tolist()
  data_Numerical = data.drop(Categorical_Cols,axis=1)
  Numerical_Cols = data_Numerical.columns.tolist()

  Q1 = data[Numerical_Cols].quantile(0.25)
  Q3 = data[Numerical_Cols].quantile(0.75)
  IQR = Q3 - Q1
  
  
  for col in Numerical_Cols:
    data.loc[ ( (data[col] < (Q1[col] - 1.5 * IQR[col])) | (data[col] > (Q3[col] + 1.5 * IQR[col])) ),col] = np.nan
    
  return data


def imputate_missing_values(data, **kwargs):
    scaler = kwargs.get(scaler, MinMaxScaler())
    ### kwargs.get('val2',"default value")
    if method == 'IterativeImputer':
        X_minmax =  scaler.fit_transform(data)
        imputer = IterativeImputer(n_nearest_features = kwargs.get(n_nearest_features, '4'),
                                   initial_strategy = kwargs.get(initial_strategy, 'median'), 
                                   max_iter = kwargs.get(max_iter, 100),
                                   random_state= kwargs.get(random_state, random.randint(1,100)))
        
        # fit on the dataset
        imputer.fit(X_minmax)
        
        # transform the dataset
        X_trans_scaled = imputer.transform(X_minmax)
        X_trans = min_max_scaler.inverse_transform(X_trans_scaled)
        
        return pd.DataFrame(X_trans, columns = data.columns.tolist())
      
      
    if method == 'mean':
        return data.fillna(data.mean())
      
    if method == 'median':
        return data.fillna(data.median())
