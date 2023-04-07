
def scatter_plot(data, outcome_col):
  '''
  Scatter plot of numerical columns with respect to the outcome.
  '''
  # Get numerical and categorical columns
  
  data_Categorical = data.select_dtypes(include=['object'])
  Categorical_Cols = data_Categorical.columns.tolist()
  data_Numerical = data.drop(Categorical_Cols,axis=1)
  Numerical_Cols = data_Numerical.columns.tolist()

  # Make figure object
  
  fig,ax = plt.subplots(len(Numerical_Cols),1, figsize=(6,len(Numerical_Cols)*2))
  
  
  # Generate scatter plots for numerical columns
  
  for i,col in enumerate(Numerical_Cols):
      plt.sca(ax[i])
      sns.scatterplot(x = col,y = outcome_col, data = data)
      plt.ylabel(col)
  plt.show()
  
  return ax
      
def dist_plot(data,outcome_col):
  
  # Get numerical and categorical columns
  
  data_Categorical = data.select_dtypes(include=['object'])
  Categorical_Cols = data_Categorical.columns.tolist()
  data_Numerical = data.drop(Categorical_Cols,axis=1)
  Numerical_Cols = data_Numerical.columns.tolist()

  
  # Make figure object
  
  fig,ax = plt.subplots(len(Numerical_Cols),1, figsize=(6,len(Numerical_Cols)*2))
  
  for i,col in enumerate(Numerical_Cols):
      plt.sca(ax[i])
      sns.displot(x = col,hue = outcome_col,data = data, kind='kde')
      
  return ax
