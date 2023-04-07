from lifelines import AalenJohansenFitter
from lifelines import NelsonAalenFitter
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
import pandas as pd

def plot_cumulative_function(data,event_col,survival_time_col, **kwargs):
  naf_m = NelsonAalenFitter()
  naf_f = NelsonAalenFitter()
  
  label_e = kwargs.get(labels[0],'Relapse')
  label_ne = kwargs.get(labels[1],'Non-Relapse')
                       
  naf_e.fit(data['SurvTime'][data[event_col]==1],
            event_observed = data[survival_time_col][data[event_col]==1], 
            label="Relapse")
  
  naf_ne.fit(data['SurvTime'][data[event_col]==0],
            event_observed = data[survival_time_col][data[event_col]==0],
            label="Non-Relapse")

  naf_e.plot_cumulative_hazard()
  naf_ne.plot_cumulative_hazard()
  plt.title("Cumulative hazard function of %s vs. %s." %(label_e,label_ne))
  
  plt.show()
  
  return naf_m, naf_f


def plot_kmf(durations, event_observed, **kwargs):
  xlabel = kwargs.get(xlabel, 'Years after treatment') 
  ylabel = kwargs.get(ylabel, 'Probability of survival')
  outfile =  kwargs.get(outfile, 'kmf_plot.png')
  
  kmf = KaplanMeierFitter()
  
  kmf.fit(durations = durations, event_observed = event_observed)
  
  Survival_fun = kmf.survival_function_
  
  plt.ylim(0,1.0)
  plt.plot(kmf.survival_function_.index/365,
           kmf.survival_function_.iloc[:,0],
           color='k',
           label = 'Estimated survival function')
  
  plt.plot(kmf.survival_function_.index/365,
           kmf.confidence_interval_.iloc[:,0],
           color='grey',
           linestyle='--',
           label = 'Confidence interval') #exponential Greenwood 
  
  plt.plot(kmf.survival_function_.index/365,
           kmf.confidence_interval_.iloc[:,1],
           color='grey',
           linestyle='--') #exponential Greenwood 
  
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend(loc='upper right')
  plt.savefig(outfile)
  print('The median survival time is:', kmf.median_survival_time_)
  
  return kmf




def plot_kmf_multivariate(durations, event_observed, factor, **kwargs):
  '''
  factor as series
  '''
  min_instances_for_km =  kwargs.get(min_instances_for_km, 100)
  title = kwargs.get(title, 'Multivariate survival curves analysis')
  xlabel = kwargs.get(xlabel, 'Years after treatment') 
  ylabel = kwargs.get(ylabel, 'Probability of survival')
  outfile =  kwargs.get(outfile, 'kmf_plot.png')
  
  
  if not isinstance(factor, pd.Series):
    raise TypeError
    
  # Select unique categiories (levels)
  levels = factor.unique()

  ax = plt.subplot(111)
  # plt.style.use('seaborn')
  kmf = KaplanMeierFitter()
  for level in levels:
      if factor.value_counts()[level]<min_instances_for_km:
          continue

      flag = factor == level

      kmf.fit(durations[flag], 
              event_observed=event_observed[flag], 
              label = level)
      
      kmf.plot(ax=ax, ci_show=False)

  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.savefig(outfile)
  
  return ax


def fit_coxPH(data, durations, event, **kwargs):
  
  # Get arguments
  penalizer = kwargs.get(penalizer,0.1)
  mask = np.random.rand(len(data)) < 0.8
  train_test_mask = kwargs.get(train_test_mask, mask)
  step_size = kwargs.get(step_size, 0.1)
  scoring_method = kwargs.get(scoring_method, 'concordance_index')
  file_name = kwargs.get(file_name, 'Cox_Results')
  
  # writer object
  writer = pd.ExcelWriter(file_name + '.xlsx')
  
  # Concatenate data and survival and event columns
  cph_data = pd.concat([data, durations.rename('survival_time'), event.rename('event')], axis=1)
  
  # create cox PH fitter object
  cph = CoxPHFitter(penalizer = penalizer)
  
  # Separate train and test datasets
  train = cph_data[train_test_mask]
  test = cph_data[~train_test_mask]
  
  # fit
  cph.fit(train, 'survival_time' ,event_col = 'event', step_size = step_size)
  cph.fit(cph_data,'survival_time' ,event_col = 'event', step_size = step_size)
  
  print('************ C-statistic for train set: ', cph.score(train, scoring_method= scoring_method))
  print('************ C-statistic for test set: ', cph.score(test, scoring_method= scoring_method))


  cph.print_summary()

  my_dict = {'duration_col':cph.duration_col,
  'event_col': cph.event_col,
  'penalizer': cph.penalizer ,
  'l1_ratio': cph.l1_ratio,
  'event_observed': cph.event_observed.sum(),
  'log_likelihood': cph.log_likelihood_,
  'concordance_index': cph.concordance_index_,
  'partial AIC': cph.AIC_partial_,
  'log_likelihood ratio test statistic': cph.log_likelihood_ratio_test().test_statistic,
  'log_likelihood ratio test p-value': cph.log_likelihood_ratio_test().p_value}

  Result_P1 = pd.DataFrame.from_dict(my_dict, orient='index', columns=['value'])

  Result_P2 = cph.summary


  Result_P1.to_excel(writer,'Sheet 1')
  Result_P2.to_excel(writer, 'Sheet 1', startrow = len(Result_P1) + 2 )
  writer.save()

  plt.figure(figsize=(6,12))
  cph.plot()
  plt.savefig(file_name + '.png')
  
  return cph



