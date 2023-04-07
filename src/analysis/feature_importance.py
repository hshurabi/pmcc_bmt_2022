from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import sys, pandas


def select_k_best_feature_importance(X,y,colnames):
    if X.shape[0]!=len(y) or X.shape[1]!=len(colnames):
        print('X and y must be in same length. Column names must be the same length as X columns.')
        sys.exit()
    featureScores = pd.DataFrame()
    featureScores_AllData = pd.DataFrame()
    
    #apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k='all')
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfpvalues = pd.DataFrame(fit.pvalues_)
    dfcolumns = pd.DataFrame(colnames)
    
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores,dfpvalues],axis=1)
    featureScores.columns = ['covariate','Chi2_Score','P-Value']  #naming the dataframe columns
    featureScores_AllData = pd.concat([featureScores_AllData,
                                       featureScores.sort_values('Chi2_Score', ascending = False).reset_index(drop=True)],
                                       axis=1)
    return featureScores_AllData
  
def feature_addition_model(data, y,
                           model = RandomForestClassifier(random_state = 1, n_estimators = 1000, criterion= 'entropy'),
                           featureScores_df, cv = 10, n_jobs = -1, plot = True):
  # Run RF iteratively
  for idx,col in enumerate(featureScores_df['covariate']):
      if idx<=1:
          continue
      mask = [data.columns.get_loc(x)] for x in list(featureScores_df['covariate'][:idx])]
      X_important_train = data[:,mask].reshape(len(data),idx)
      Cv_scores = cross_validate(model, 
                                 X_important_train, 
                                 y, 
                                 cv = 10, 
                                 scoring=('roc_auc','accuracy','recall','precision'),
                                 n_jobs=n_jobs) 
      model.fit(X_important_train, y)
      featureScores_df.at[idx,'ROC_AUC'] =  np.mean(Cv_scores['test_roc_auc'])
  if plot:
      x = featureScores_df['ROC_AUC'].plot()
      x.set_xlabel('Feature Index')
      x.set_ylabel('AUC score')
      x.get_figure().savefig('FeatureAddition.png')
  return featureScores_df
  
  
  
def rf_feature_importance(X,y,colnames, **kwargs):
  if X.shape[0]!=len(y) or X.shape[1]!=len(colnames):
      print('X and y must be in same length. Column names must be the same length as X columns.')
      sys.exit()
  n_estimators = kwargs.get(n_estimators, 1000)
  random_state = kwargs.get(random_state, 1)
  n_jobs = kwargs.get(n_jobs, -1)
  clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
  # Train the classifier
  clf.fit(X,y)

  # Create a selector object that will use the random forest classifier to identify
  # features that have an importance of more than 0.15
  dfscores = pd.DataFrame(clf.feature_importances_)
  dfcolumns = pd.DataFrame(colnames)
  #concat two dataframes for better visualization 
  featureScores = pd.concat([dfcolumns,dfscores],axis=1)
  featureScores.columns = ['covariate','RF_Score']  #naming the dataframe columns
  featureScores = featureScores.sort_values('RF_Score', ascending = False,ignore_index=True)
  return featureScores
