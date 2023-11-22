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
        
    
