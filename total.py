from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer

def monomial_metrics(_input: pd.Series, _output: pd.Series,
                   model = LinearRegression(), ) -> dict:
    if len(_input.shape) == 1:
        _name = _input.name
        _input = np.reshape(_input.values, (-1, 1))
    else: 
        _name = ', '.join(_input.columns)

    model.fit(_input, _output)
    _pred = model.predict(_input)

    mse = mean_squared_error(_output, _pred)
    r2 = r2_score(_output, _pred)
    
    return {'variable': _name,
            'MSE': mse,
            'R^2': r2}

def polinomial_metrics(_input: pd.Series, _output: pd.Series,
                      degree = 1, model = None, csr_matrix = False) -> dict:

    if model == None:
        model = Pipeline([
            ('scale', StandardScaler()),
            ('polynomial', PolynomialFeatures(degree=degree)),
            ('model', LinearRegression())
        ])
        
    if csr_matrix == True:
        _name = 'all'
        
    elif len(_input.shape) == 1:
        _name = _input.name
        _input = np.reshape(_input.values, (-1, 1))
    else: 
        _name = ', '.join(_input.columns)
        
    

    model.fit(_input, _output)
    _pred = model.predict(_input)

    mse = mean_squared_error(_output, _pred)
    r2 = r2_score(_output, _pred)
    
    return {'variable': _name,
            'degree': degree,
            'MSE': mse,
            'R^2': r2}

def preprocesing_dataX(num_df: pd.Series, obj_df:pd.Series, model = None):
    
    X = pd.DataFrame()
    ## obj_df
    for i, obj in enumerate(obj_df.columns):
        label = LabelEncoder()
        encode = label.fit_transform(obj_df[obj])
        X[obj] = encode
        
    ## num_df
    scale = StandardScaler()
    X[num_df.columns] = scale.fit_transform(num_df)
    
    ## column transformer
    ct = ColumnTransformer([('town', OneHotEncoder(), [0, 1, 2])], remainder = 'passthrough')
    X = ct.fit_transform(X)
    
    return X

def get_value_influence(df6: pd.DataFrame, target_column: str, compared_column: list, max_target = 30000, include = None) -> list:
  df6_grp = df6.groupby(['make', 'bodystyle', 'enginetype'],
                           as_index = False).mean()
  df6_pivot = df6_grp.pivot(index = ['make', 'enginetype'], columns = ['bodystyle'])

  if include == 'max':
    result = [df6_pivot.max(axis=1).idxmax() + (df6_pivot.max().idxmax()[1],)]
  elif include == None:
    df_new = df6_pivot[df6_pivot > max_target]
    df_new = df_new.stack()
    result = df_new.index.to_list()

  return result

def get_outlier(df: pd.Series):
  # Calculate Q1, Q3 and IQR
  Q1 = df.quantile(0.25)
  Q3 = df.quantile(0.75)
  IQR = Q3 - Q1

  df_outliers = df[(df < Q1 - 1.5 * IQR) | (df > Q3 + 1.5 * IQR)].values.tolist()
  outlier_percent = 100.0*(len(df_outliers)/len(df))

  return(df_outliers, outlier_percent)


        


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer

def monomial_metrics(_input: pd.Series, _output: pd.Series,
                   model = LinearRegression(), ) -> dict:
    if len(_input.shape) == 1:
        _name = _input.name
        _input = np.reshape(_input.values, (-1, 1))
    else: 
        _name = ', '.join(_input.columns)

    model.fit(_input, _output)
    _pred = model.predict(_input)

    mse = mean_squared_error(_output, _pred)
    r2 = r2_score(_output, _pred)
    
    return {'variable': _name,
            'MSE': mse,
            'R^2': r2}

def polinomial_metrics(_input: pd.Series, _output: pd.Series,
                      degree = 1, model = None, csr_matrix = False) -> dict:

    if model == None:
        model = Pipeline([
            ('scale', StandardScaler()),
            ('polynomial', PolynomialFeatures(degree=degree)),
            ('model', LinearRegression())
        ])
        
    if csr_matrix == True:
        _name = 'all'
        
    elif len(_input.shape) == 1:
        _name = _input.name
        _input = np.reshape(_input.values, (-1, 1))
    else: 
        _name = ', '.join(_input.columns)
        
    

    model.fit(_input, _output)
    _pred = model.predict(_input)

    mse = mean_squared_error(_output, _pred)
    r2 = r2_score(_output, _pred)
    
    return {'variable': _name,
            'degree': degree,
            'MSE': mse,
            'R^2': r2}

def preprocesing_dataX(num_df: pd.Series, obj_df:pd.Series, model = None):
    
    X = pd.DataFrame()
    ## obj_df
    for i, obj in enumerate(obj_df.columns):
        label = LabelEncoder()
        encode = label.fit_transform(obj_df[obj])
        X[obj] = encode
        
    ## num_df
    scale = StandardScaler()
    X[num_df.columns] = scale.fit_transform(num_df)
    
    ## column transformer
    ct = ColumnTransformer([('town', OneHotEncoder(), [0, 1, 2])], remainder = 'passthrough')
    X = ct.fit_transform(X)
    
    return X
        
    
