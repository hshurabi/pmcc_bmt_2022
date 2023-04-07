
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

  
def prepare_for_ml(local_df):
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

  
  
  def print_info_cat(local_df):
    #Print Informaton of categorical features 
    local_df = local_df.replace('',np.nan)
    Temp  = local_df.select_dtypes(include=['object'])
    local_df = local_df.replace(np.nan,'NULL')
    Categorical_cols = Temp.columns.tolist()
    for col in Categorical_cols:
        df = pd.value_counts(local_df[col].str.cat(sep=' ').split(),
                             dropna =False).rename_axis(col).reset_index(name='Count')
        try:
            missing_no = float(df.loc[df[col]=='NULL','Count'])
        except:
            missing_no = 0
        new=pd.DataFrame({col:'Total','Count':  [local_df.shape[0] - missing_no] } )
        df = pd.concat([new,df],axis=0,ignore_index = True)
        df['Percentage'] = df['Count']/local_df.shape[0]
        df.loc[df[col]=='Total', 'No Missing'] = missing_no
        df = df.drop(df[df[col]=='NULL'].index)
        df.loc[0,'Missing%'] = missing_no/local_df.shape[0]*100
        print(df)
        
        
   def print_info_num(local_df):
    #Print Informaton of numerical features 
    datatolook = local_df.replace('',np.nan)
    Categorical_Cols = local_df.select_dtypes(include=['object']).columns.tolist()
    local_df = local_df.drop(Categorical_Cols,axis=1)
    Numerical_Cols = local_df.columns.tolist()

    for col in Numerical_Cols:
        m = pd.DataFrame(local_df[col].describe()).T
        m['No Missing'] = local_df.shape[0] - m['count']
        m['Missing %'] = (local_df.shape[0] - m['count'])/local_df.shape[0]        
        m = m[['count','50%','min','max','No Missing','Missing %']]
        print(m)
