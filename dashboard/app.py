import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import model_selection

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

housing = pd.read_csv('../Ames_HousePrice.csv', index_col=0)
#Remove Outliers
housing = housing[np.logical_and(housing.SalePrice >= 40000, housing.SalePrice <= 750000)]

# #Remove Bad Classes
housing = housing[housing.Neighborhood != 'Landmrk']
housing.SaleType = housing.SaleType.astype('string')
housing. SaleType = housing.SaleType.str.strip()
housing = housing[housing.SaleType == 'WD']
housing = housing[housing.SaleCondition == 'Normal']


#Replace NAs
housing = housing.fillna(0)
#Log Transforms
housing['LogSalePrice'] = np.log(housing.SalePrice)
# housing['LogGrLivArea'] = np.log(housing.GrLivArea)

#Area Calculations
housing['PorchTotSF'] = housing.OpenPorchSF + housing.EnclosedPorch + housing['3SsnPorch'] + housing.ScreenPorch

#Binary HasBLANK Categories
housing['HasGarage'] = np.where(housing.GarageCars > 0, 1, 0)
housing['HasPool'] = np.where(housing.PoolArea > 0, 1, 0)
housing['HasPorch'] = np.where(housing.PorchTotSF > 0, 1, 0)
housing['HasDeck'] = np.where(housing.WoodDeckSF > 0, 1, 0)
housing['HasFinBsmt'] = np.where(housing.BsmtFinSF1 > 0, 1, 0)
housing['HasFireplace'] = np.where(housing.Fireplaces > 0, 1, 0)
housing['HasFence'] = np.where(housing.Fence.notna(), 1, 0)
housing.Neighborhood = housing.Neighborhood.replace({'MeadowV':1,'BrDale':2, 'IDOTRR':3, 'BrkSide':4, 'OldTown':5, 'Edwards':6, 'SWISU':7, 'Landmrk':8, 'Sawyer':9,\
                           'NPkVill':10, 'Blueste':11, 'NAmes':12, 'Mitchel':13, 'SawyerW':14, 'Gilbert':15, 'NWAmes':16, 'Greens':17, 'Blmngtn':18,\
                           'CollgCr':19, 'Crawfor':20, 'ClearCr':21, 'Somerst':22, 'Timber':23, 'Veenker':24, 'GrnHill':25, 'StoneBr':26,'NridgHt':27, 'NoRidge':28})

#Binary Quality/Cond Categories
housing['GarageFinish_Fin']= np.where(housing.GarageFinish == 'Unf', 0, 1)

keep = ['LogSalePrice', 'GrLivArea', 'HasGarage', 'HasPool','HasFireplace','OverallQual','OverallCond','Neighborhood','GarageCars']
housing = housing[keep]



y = housing['LogSalePrice']
x = housing.drop('LogSalePrice', axis=1)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=.2, random_state=0)

lm = LinearRegression()
lm.fit(x_train, y_train)

app.layout = html.Div(
    children=[
    html.H1(children='Ames Housing Tool'),

    html.Div(children='''
        House hunter toolkit to fit budget.
    '''),

    dbc.Label("Budget"),
    dcc.Input(
        id='budget',
        placeholder='Insert budget',
        type='number',
        value=100000,
        debounce=True,
        min=12789, max=755000
    ),

    dbc.Label("Gross Living Area"),
    dcc.Input(
        id='GrLivArea',
        placeholder='Insert living area(ft^2)',
        type='number',
        value=1000,
        debounce=True,
        min=334, max=4676
    ),

    dbc.Label("Number of cars in garage"),
    dcc.Input(
        id='Garage_Cars',
        placeholder='Number of cars in garage',
        type='number',
        value=1,
        debounce=True,
        min=0, max=5
    ),

    dbc.Label("Overall condition"),
    dcc.Input(
        id='overall_cond',
        placeholder='Desired overall condition (1-10)',
        type='number',
        value=5,
        debounce=True,
        min=0, max=10
    ),

    dbc.Label("Overall quality"),
    dcc.Input(
        id='overall_qual',
        placeholder='Desired overall quality (1-10)',
        type='number',
        value=5,
        debounce=True,
        min=0, max=10
    ),

    dcc.Checklist(
        id='boolean_features',
        options=[
            {'label': 'Garage', 'value': 'HasGarage'},
            {'label': 'Pool', 'value': 'HasPool'},
            {'label': 'Fire Place', 'value': 'HasFireplace'}
        ],
        value=[],
        labelStyle={'display': 'inline-block'}
    ),

    dbc.Label("Neighborhood"),
    dcc.Dropdown(
        id='neighborhood',
        options=[
            {'label': 'Meadow', 'value': 1},
            {'label': 'BrDale', 'value': 2},
            {'label': 'IDOTRR', 'value': 3},
            {'label': 'BrkSide', 'value': 4},
            {'label': 'OldTown', 'value': 5},
            {'label': 'Edwards', 'value': 6},
            {'label': 'SWISU', 'value': 7},
            {'label': 'Landmrk', 'value': 8},
            {'label': 'Sawyer', 'value': 9},
            {'label': 'NPkVill', 'value': 10},
            {'label': 'Blueste', 'value': 11},
            {'label': 'NAmes', 'value': 12},
            {'label': 'Mitchel', 'value': 13},
            {'label': 'SawyerW', 'value': 14},
            {'label': 'Gilbert', 'value': 15},
            {'label': 'NWAmes', 'value': 16},
            {'label': 'Greens', 'value': 17},
            {'label': 'Blmngtn', 'value': 18},
            {'label': 'CollgCr', 'value': 19},
            {'label': 'Crawfor', 'value': 20},
            {'label': 'ClearCr', 'value': 21},
            {'label': 'Somerst', 'value': 22},
            {'label': 'Timber', 'value': 23},
            {'label': 'Veenker', 'value': 24},
            {'label': 'GrnHill', 'value': 25},
            {'label': 'StoneBr', 'value': 26},
            {'label': 'NridgHt', 'value': 27},
            {'label': 'NoRidge', 'value': 28}
        ],

        value=12
    ),


    html.Div(id='predicted_price')
])


@app.callback(
    Output(component_id='predicted_price', component_property='children'),
    [Input(component_id='budget', component_property='value'),
    Input(component_id='GrLivArea', component_property='value'),
    Input(component_id='Garage_Cars', component_property='value'),
    Input(component_id='overall_cond', component_property='value'),
    Input(component_id='overall_qual', component_property='value'),
    Input(component_id='boolean_features', component_property='value'),
    Input(component_id='neighborhood', component_property='value')
    ]
)
def update_output_div(budget_value,GrLivArea_value, Garage_Cars_value, overall_cond_value,overall_qual_value,boolean_features_value,neighborhood_value):
    if budget_value is None:
        Budget = 100000
    else:
        Budget = budget_value
    LivArea = GrLivArea_value
    print(LivArea)
    GarageCars =Garage_Cars_value
    OverallQual = overall_qual_value
    OverallCond = overall_cond_value
    Neighborhood = neighborhood_value
    HasGarage = 0
    HasPool = 0
    HasFireplace = 0
    if "HasGarage" in boolean_features_value:
        HasGarage=1
    if "HasPool" in boolean_features_value:
        HasPool=1
    if "HasFireplace" in boolean_features_value:
        HasFireplace=1
    

    buyer_data = [[np.log(Budget), LivArea, HasGarage, HasPool, HasFireplace, OverallQual, OverallCond,Neighborhood,GarageCars]]
    buyer = pd.DataFrame(data = buyer_data, columns = keep)

    budget = buyer['LogSalePrice']
    buyer_x = buyer.drop('LogSalePrice', axis=1)
    predicted_price = np.exp(lm.predict(buyer_x)[0])

    
    return 'Predicted Price: ${:.2f}'.format(predicted_price)


if __name__ == '__main__':
    app.run_server(debug=True)