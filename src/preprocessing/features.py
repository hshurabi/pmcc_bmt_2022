
from sklearn.feature_extraction.text import CountVectorizer

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
