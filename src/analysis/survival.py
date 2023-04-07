from lifelines import AalenJohansenFitter
from lifelines import NelsonAalenFitter
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter

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
  x_label = kwargs.get(x_label, 'Years after treatment') 
  y_label = kwargs.get(y_label, 'Probability of survival')
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
  
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend(loc='upper right')
  plt.savefig(outfile)
  print('The median survival time is:', kmf.median_survival_time_)
  
  return kmf
