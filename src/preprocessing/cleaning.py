
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
