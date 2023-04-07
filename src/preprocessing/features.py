
from sklearn.feature_extraction.text import CountVectorizer

def get_correlated_cols(local_df, **kwargs):

  upper_limit = kwargs.get(upper_limit,1.0)
  lower_limit = kwargs.get(lower_limit,0.75)
  
  corr = local_df.corr()
  # Create correlation matrix
  corr_matrix = corr.abs()

  # Select upper triangle of correlation matrix
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

  # Find features with correlation greater than 0.95
  to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

  # Print correlated cols
  s = upper.unstack()
  so = s.sort_values(kind='quicksort')

  print('Correlated columns are: ' )
  print(so[(so>lower_limit)&(so<upper_limit)])
  
  return so[(so>lower_limit)&(so<upper_limit)]

  
def PrepareForMl(local_df):
  ''' function to make dummy variables.
  gets a dataframe as input
  returns a dataframe with dummy variables (ready for ML training) 
  '''
    local_df = local_df.replace('',np.nan)
    local_Temp  = local_df.select_dtypes(include=['object'])
    Categorical_cols = local_Temp.columns.tolist()

    vectorizer = CountVectorizer()
    Data_local_Temp = local_df
    for col in Categorical_cols:
        local_Temp.loc[:,col] = local_Temp[col].replace(np.nan,'')
        X = vectorizer.fit_transform(local_Temp[col])
        colname_vect = vectorizer.get_feature_names()
        colname_vect = [x.upper() for x in colname_vect]
        colname = [str(col)+'_' +s for s in colname_vect]
        Data_local_Temp = pd.concat( [ Data_local_Temp , pd.DataFrame( X.toarray() , columns = colname ) ] , axis = 1 )
        Data_local_Temp = Data_local_Temp.drop([col] , axis = 1) 
    return Data_local_Temp
