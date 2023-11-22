import pandas as pd
import scipy.stats as stats
import re
import statsmodels.api as sm
from statsmodels.formula.api import ols

def get_num_columns(database: pd.DataFrame) -> pd.DataFrame:
  _nums = database.select_dtypes(include = 'number').columns.to_list()

  return database[_nums].copy()

def get_obj_columns(database: pd.DataFrame) -> pd.DataFrame:
  _objs = database.select_dtypes(include = 'object').columns.to_list()

  return database[ _objs].copy()

def check_correlation_p_value(column1: pd.Series, column2: pd.Series,
                              coef_threshold = 0.5, p_threshold = 0.05,
                              include = None) -> list:


  includes = [None, 'all']
  assert include in includes, "include just be None or 'all'"

  _coef, _p_value = stats.pearsonr(column1, column2)

  if _p_value > p_threshold:
    return False
  if _coef >= coef_threshold or _coef <= -1.0*coef_threshold:
    if include == 'all':
      return [True, _coef, _p_value]
    return True

  return False

def check_affection(database: pd.DataFrame, target_column: str, coef_threshold = 0.1, p_threshold = 0.05, include = None) -> list:



  includes = [None, 'all']
  assert include in includes, "include just be None or 'all'"
  database = get_num_columns(database)

  _affection = []

  for i in database.columns:
    if i == target_column:
      continue
    _affect = check_correlation_p_value(database[i], database[target_column], coef_threshold = coef_threshold, p_threshold = p_threshold, include=include)
    if _affect and include == 'all':
      _affection.append([i, _affect[1], _affect[2]])
    elif _affect:
      _affection.append(i)


  return _affection

def get_strong_week_affection(database: pd.DataFrame, target_column: str, coef_threshold = [0.3, 0.5, 0.8], p_threshold = 0.05) -> dict:
  _strong_affection = []
  _week_affection = []
  _median_affection = []
  database = get_num_columns(database)

  _affection = check_affection(database= database, target_column= target_column, coef_threshold=coef_threshold[0], p_threshold=p_threshold, include = 'all')

  for column in _affection:
    X = {'name': column[0],
         'coef': column[1],
         'p_value': column[2]}
    if X['coef'] <= -1*coef_threshold[2] or  X['coef'] >= 1*coef_threshold[2]:
      _strong_affection.append(X)
    elif X['coef'] >= -1*coef_threshold[1] and X['coef'] <= coef_threshold[1]:
      _week_affection.append(X)
    else:
      _median_affection.append(X)



  return {'strong_affection': _strong_affection,
          'week_affection': _week_affection,
          'median_affection': _median_affection}



def get_category_influence(df:pd.DataFrame, target_column, get_top = None, plimit = 1e-04, ascending =True):
  _df = get_obj_columns(df)
  _category = _df.columns
  _category = [re.sub('\W', '', i) for i in _category]
  _df.columns = _category
  _df[target_column] = df[target_column]


  model = ols(target_column + ' ~ ' + ' + '.join(_category), data = _df).fit()
  # thực hiện kiểm định ANOVA
  anova_table = sm.stats.anova_lm(model)

  if ascending:
    anova_table = anova_table[anova_table['PR(>F)'] < plimit]
  else:
    anova_table = anova_table[anova_table['PR(>F)'] >= plimit]


  try:
    get_top = int(get_top)

    anova_table = anova_table.sort_values(by = 'PR(>F)', ascending= ascending)[:get_top]

    return anova_table.index.to_list(), anova_table['PR(>F)'].to_list()
  except:
    anova_table = anova_table.sort_values(by = 'PR(>F)', ascending= ascending)
    return anova_table.index.to_list(), anova_table['PR(>F)'].to_list()
