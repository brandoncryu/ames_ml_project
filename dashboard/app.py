import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX])

##### Initial data handling ########

### Feature Cleaning ###
housing = pd.read_csv('./Ames_HousePrice.csv', index_col=0)
# Remove Outliers
housing = housing[np.logical_and(housing.SalePrice >= 40000, housing.SalePrice <= 750000)]
housing = housing[housing.GrLivArea < 4000]
housing = housing[housing.BedroomAbvGr > 0]
housing.loc[housing['GarageCars'] == 5, 'GarageCars'] = 4

# Remove Bad Classes
housing = housing[housing.Neighborhood != 'Landmrk']

housing.MSZoning = housing.MSZoning.astype('string')
housing.MSZoning = housing.MSZoning.str.strip()
housing = housing[housing.MSZoning.isin(["FV", "RH", "RL", "RM"])]

housing = housing[housing.Functional.isin(["Typ", "Min1", "Min2"])]

housing.SaleType = housing.SaleType.astype('string')
housing.SaleType = housing.SaleType.str.strip()
housing = housing[housing.SaleType == 'WD']

housing = housing[housing.SaleCondition == 'Normal']

# Replace NAs
housing = housing.fillna(0)

### Feature Engineering ###
# Area Calculations
housing['PorchTotSF'] = housing.OpenPorchSF + housing.EnclosedPorch + housing['3SsnPorch'] + housing.ScreenPorch
housing['BsmtSF'] = housing.BsmtFinSF1 + housing.BsmtFinSF2
# housing.loc[housing['BsmtSF'] == 0, 'BsmtSF'] = np.exp(1)

# Log Transforms
housing['LogSalePrice'] = np.log(housing.SalePrice)
housing['LogLotArea'] = np.log(housing.LotArea)
housing['LogGrLivArea'] = np.log(housing.GrLivArea)
housing['LogBsmtSF'] = np.log(housing.BsmtSF)

# Categorical to Ordinal
housing.Neighborhood = housing.Neighborhood.replace({'MeadowV':1,'BrDale':2, 'IDOTRR':3, 'BrkSide':4, 'OldTown':5, 'Edwards':6, 'SWISU':7, 'Landmrk':8, 'Sawyer':9,\
                           'NPkVill':10, 'Blueste':11, 'NAmes':12, 'Mitchel':13, 'SawyerW':14, 'Gilbert':15, 'NWAmes':16, 'Greens':17, 'Blmngtn':18,\
                           'CollgCr':19, 'Crawfor':20, 'ClearCr':21, 'Somerst':22, 'Timber':23, 'Veenker':24, 'GrnHill':25, 'StoneBr':26,'NridgHt':27, 'NoRidge':28})
housing.BldgType = housing.BldgType.replace({'2fmCon':1,'Twnhs':2, 'Duplex':3, '1Fam':4, 'TwnhsE':5})
housing.HouseStyle = housing.HouseStyle.replace({'1.5Unf':1,'1.5Fin':2, 'SFoyer':3, 'SLvl':4, '1Story':5, '2.5Unf':6, '2Story':7, '2.5Fin':8})
housing.MoSold = housing.MoSold.replace({1:11, 9:10, 8:9, 6:8, 7:7, 11:6, 12:5, 2:4, 3:3, 10:2, 5:1, 4:0})

# Renumber Numerical
housing['NumBath'] = housing.FullBath + 0.5*housing.HalfBath + 0.5*housing.BsmtFullBath

# Binary HasBLANK Categories
housing['BeenRemod'] = np.where(housing.YearBuilt != housing.YearRemodAdd, 1, 0)
housing['HasFinBsmt'] = np.where(housing.BsmtFinSF1 > 0, 1, 0)
housing['HasFinGarage'] = np.where(housing.GarageFinish == "Fin", 1, 0)
housing['HasPool'] = np.where(housing.PoolArea > 0, 1, 0)
housing['HasFireplace'] = np.where(housing.Fireplaces > 0, 1, 0)
housing['HasPorch'] = np.where(housing.PorchTotSF > 0, 1, 0)
housing['HasDeck'] = np.where(housing.WoodDeckSF > 0, 1, 0)


# Binary Quality/Cond Categories
housing['AttachedGarage'] = np.where(housing.GarageType == "Attchd", 1, 0)
housing['GreatElectric'] = np.where(housing.Electrical == "SBrkr", 1, 0)
housing['GreatHeat'] = np.where(housing.HeatingQC == "Ex", 1, 0)
housing['CentralAir'] = np.where(housing.CentralAir == "Y", 1, 0)

# Feature Selection
model_cols = [ 'LogGrLivArea', 'LogLotArea', 'OverallQual', 'OverallCond',
       'Neighborhood', 'BldgType', 'NumBath', 'GarageCars', 'HasFinBsmt',
       'HasFinGarage', 'HasFireplace', 'HasPorch', 'HasDeck', 'AttachedGarage',
       'GreatElectric', 'GreatHeat', 'CentralAir']


x = housing[model_cols]
y = housing.LogSalePrice

##### Fitting lasso linear model with training data #####

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=.2, random_state=0)

clf = linear_model.Lasso(alpha=9.326033468832199e-05)
clf.fit(x_train,y_train)

##### UI components #####
app.layout = html.Div(
    children=[
        html.Br(),
        dbc.Row(
            dbc.Col(
                html.H1(children='Ames Housing Tool'),
                width={"size": 6, "offset": 1}
            )
        ),

        dbc.Row(
            dbc.Col(
                html.H5(children='House hunter toolkit to fit their budget.'),
                width={"size": 6, "offset": 1}
            )
        ),

        html.Br(),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Budget"),
                        daq.NumericInput(
                            id='budget',
                            value=100000,
                            min=12789, max=755000,
                            size=200
                        ),
                        html.Br(),

                        dbc.Label("Gross Living Area (ft^2)"),
                        daq.NumericInput(
                            id='GrLivArea',
                            value=1000,
                            min=334, max=4676,
                            size=200
                        ),
                        html.Br(),

                        dbc.Label("Lot Area (ft^2)"),
                        daq.NumericInput(
                            id='LotArea',
                            value=1000,
                            min=0, max=215000,
                            size=200
                        ),
                        html.Br(),

                        dbc.Label("Overall quality"),
                        daq.NumericInput(
                            id='overall_qual',
                            value=5,
                            min=0, max=10,
                            size=200    
                        ),
                        html.Br(),

                        dbc.Label("Overall condition"),
                        daq.NumericInput(
                            id='overall_cond',
                            value=5,
                            min=0, max=10,
                            size=200
                        ),
                        html.Br(),

                        dbc.Label("Neighborhood"),
                        dcc.Dropdown(
                            id='neighborhood',
                            clearable=False,
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
                        html.Br(),

                        dbc.Label("Building Type"),
                        dcc.Dropdown(
                            id='BldgType',
                            clearable=False,
                            options=[
                                {'label': '2 Family Condo', 'value': 1},
                                {'label': 'Town House (Middle)', 'value': 2},
                                {'label': 'Duplex', 'value': 3},
                                {'label': '1 Family', 'value': 4},
                                {'label': 'Townhouse (End)', 'value': 5},
                            ],
                            value=4
                        ),
                        html.Br(),

                        dbc.Label("Number of Bathrooms"),
                        daq.NumericInput(
                            id='NumBath',
                            value=5,
                            min=0, max=7,
                            size=200
                        ),
                        html.Br(),

                        dbc.Label("Number of cars in garage"),
                        daq.NumericInput(
                            id='Garage_Cars',
                            value=1,
                            min=0, max=5,
                            size=200
                        ),
                        html.Br(),

                        dbc.Label("Select features you would like in your home"),
                        dcc.Checklist(
                            id='boolean_features',
                            options=[
                                {'label': 'Finished Basement', 'value': 'HasFinBsmt'},
                                {'label': 'Finished Garage', 'value': 'HasFinGarage'},
                                {'label': 'Fire Place', 'value': 'HasFireplace'},
                                {'label': 'Porch', 'value': 'HasPorch'},
                                {'label': 'Deck', 'value': 'HasDeck'},
                                {'label': 'Attached Garage', 'value': 'AttachedGarage'},
                                {'label': 'Excellent Elctricity', 'value': 'GreatElectric'},
                                {'label': 'Excellent Heating', 'value': 'GreatHeat'},
                                {'label': 'Central Air', 'value': 'CentralAir'},
                            ],
                            value=[],
                            labelStyle={'display': 'inline-block'}
                        ),
                        html.Br(),

                        
                    ],
                    width={"size": 2, "offset": 1}
                ),
                dbc.Col(
                    [
                        html.H4(id='predicted_price'),
                        html.Br(),
                        html.H4(id='over_under_budget'),
                        html.Br(),
                        html.H4('Recommendations:'),
                        html.Ul(
                            id='recommendations'
                        )
                    ]

                )
            ]
        )
])

##### Output functions #####

@app.callback(
    Output(component_id='predicted_price', component_property='children'),
    Output(component_id='over_under_budget', component_property='children'),
    Output(component_id='recommendations', component_property='children'),
    [Input(component_id='budget', component_property='value'),
    Input(component_id='GrLivArea', component_property='value'),
    Input(component_id='LotArea', component_property='value'),
    Input(component_id='overall_qual', component_property='value'),
    Input(component_id='overall_cond', component_property='value'),
    Input(component_id='neighborhood', component_property='value'),
    Input(component_id='BldgType', component_property='value'),
    Input(component_id='NumBath', component_property='value'),
    Input(component_id='Garage_Cars', component_property='value'),
    Input(component_id='boolean_features', component_property='value')
    ]
)
def update_recommendations(budget_value,GrLivArea_value, LotArea_value,OverallQual_value, OverallCond_value,Neighborhood_value,BldgType_value,NumBath_value, GarageCars_value,boolean_features_value):

    Neighborhood_dict = {1:'MeadowV',2:'BrDale', 3:'IDOTRR', 4:'BrkSide', 5:'OldTown', 6:'Edwards',7:'SWISU', 8:'Landmrk',9: 'Sawyer',\
                           10:'NPkVill', 11:'Blueste', 12:'NAmes', 13:'Mitchel', 14:'SawyerW', 15:'Gilbert', 16:'NWAmes', 17:'Greens', 18:'Blmngtn',\
                           19:'CollgCr', 20:'Crawfor', 21:'ClearCr',22: 'Somerst', 23:'Timber',24: 'Veenker', 25:'GrnHill',26: 'StoneBr',27:'NridgHt', 28:'NoRidge'}
    BldgType_dict = {1:'2fmCon',2:'Twnhs', 3:'Duplex', 4:'1Fam', 5:'TwnhsE'}

    HasFinBsmt=0
    HasFinGarage=0
    HasFireplace=0
    HasPorch=0
    HasDeck=0
    AttachedGarage=0
    GreatElectric=0
    GreatHeat=0
    CentralAir=0
    if "HasFinBsmt" in boolean_features_value:
        HasFinBsmt=1
    if "HasFinGarage" in boolean_features_value:
        HasFinGarage=1
    if "HasFireplace" in boolean_features_value:
        HasFireplace=1
    if "HasPorch" in boolean_features_value:
        HasPorch=1
    if "HasDeck" in boolean_features_value:
        HasDeck=1
    if "AttachedGarage" in boolean_features_value:
        AttachedGarage=1
    if "GreatElectric" in boolean_features_value:
        GreatElectric=1
    if "GreatHeat" in boolean_features_value:
        GreatHeat=1
    if "CentralAir" in boolean_features_value:
        CentralAir=1

    buyer_data = [[np.log(GrLivArea_value), np.log(LotArea_value), OverallQual_value, OverallCond_value, Neighborhood_value, BldgType_value, NumBath_value, GarageCars_value,\
        HasFinBsmt, HasFinGarage, HasFireplace, HasPorch, HasDeck, AttachedGarage, GreatElectric, GreatHeat, CentralAir]]
    buyer_x = pd.DataFrame(data = buyer_data, columns = model_cols)
    budget = [[budget_value]]

    predicted_price = np.exp(clf.predict(buyer_x)[0])

    #search for ways to find a good deal
    recommendation   = []
    boolean_features = ['HasFinBsmt', 'HasFinGarage', 'HasFireplace', 'HasPorch', 'HasDeck', 'AttachedGarage', 'GreatElectric', 'GreatHeat', 'CentralAir']
    ordinal_features = ['OverallQual','OverallCond','GarageCars','NumBath']
    area_features    = ['LogGrLivArea', 'LogLotArea']

#     Overbudget!! Lower our cost
    if predicted_price>budget[0]:
        over_under_budget = 'You are over budget. Follow recommendations below to save costs.'
        for feature in boolean_features:
            updated_buyer = buyer_x.copy()
            if updated_buyer[feature][0]>0:
                updated_buyer[feature]=0
                updated_price = np.exp(clf.predict(updated_buyer)[0])
                difference = predicted_price-updated_price
                append_string = 'Removing ' + feature + '-----Savings: ${0:,.2f}'.format(difference) + '-----Predicted Price: ${0:,.2f}'.format(updated_price)
                budget_difference = abs(updated_price-budget_value)
                recommendation.append([budget_difference, append_string])

        for feature in ordinal_features:
            counter = 0
            while (updated_buyer[feature][0]>1) & (counter<2):
                updated_buyer = buyer_x.copy()
                counter=counter +1
                updated_buyer[feature]=updated_buyer[feature]-counter
                updated_price = np.exp(clf.predict(updated_buyer)[0])
                difference = predicted_price-updated_price
                append_string = 'Setting ' + feature + '=' + str(updated_buyer[feature][0]) + '-----Savings: ${0:,.2f}'.format(difference) + '-----Predicted Price: ${0:,.2f}'.format(updated_price)
                budget_difference = abs(updated_price-budget_value)
                recommendation.append([budget_difference, append_string])
        
        for feature in area_features:
            if feature == 'LogGrLivArea':
                feature_text = 'Gross Living Area'
            else:
                feature_text = 'Lot Area'
            counter = 0
            while (np.exp(updated_buyer[feature][0])>100) & (counter<2):
                updated_buyer = buyer_x.copy()
                counter=counter +1
                new_area = np.exp(updated_buyer[feature][0])-counter*100
                updated_buyer[feature]=np.log(new_area)
                updated_price = np.exp(clf.predict(updated_buyer)[0])
                difference = predicted_price-updated_price
                append_string = 'Setting' + feature_text + '= {:.0f}'.format(new_area) + '-----Savings: ${0:,.2f}'.format(difference) + '-----Predicted Price: ${0:,.2f}'.format(updated_price)
                budget_difference = abs(updated_price-budget_value)
                recommendation.append([budget_difference, append_string])
        counter = 0
        while (counter<2) & (updated_buyer['Neighborhood'][0]>1):
            counter=counter +1
            updated_buyer = buyer_x.copy()
            updated_buyer['Neighborhood']=updated_buyer['Neighborhood']-counter
            updated_price = np.exp(clf.predict(updated_buyer)[0])
            difference = predicted_price-updated_price
            new_neighborhood = Neighborhood_dict[updated_buyer['Neighborhood'][0]] 
            append_string = 'Look for homes in ' + new_neighborhood + '-----Savings: ${0:,.2f}'.format(difference) + '-----Predicted Price: ${0:,.2f}'.format(updated_price)
            budget_difference = abs(updated_price-budget_value)
            recommendation.append([budget_difference, append_string])

        counter = 0
        while (counter<2) & (updated_buyer['BldgType'][0]>1):
            counter=counter +1
            updated_buyer = buyer_x.copy()
            updated_buyer['BldgType']=updated_buyer['BldgType']-counter
            updated_price = np.exp(clf.predict(updated_buyer)[0])
            difference = predicted_price-updated_price
            new_BldgType = BldgType_dict[updated_buyer['BldgType'][0]] 
            append_string = 'Looking for ' + new_BldgType + '-----Savings: ${0:,.2f}'.format(difference) + '-----Predicted Price: ${0:,.2f}'.format(updated_price)
            budget_difference = abs(updated_price-budget_value)
            recommendation.append([budget_difference, append_string])

# You are Under Budget. Increase cost
    if predicted_price<budget[0]:
        over_under_budget = 'You are under budget. Follow recommendations below to increase costs.'
        for feature in boolean_features:
            updated_buyer = buyer_x.copy()
            if updated_buyer[feature][0]==0:
                updated_buyer[feature]=1
                updated_price = np.exp(clf.predict(updated_buyer)[0])
                difference = abs(predicted_price-updated_price)
                append_string = 'Adding ' + feature + '-----Increased cost: ${0:,.2f}'.format(difference) + '-----Predicted Price: ${0:,.2f}'.format(updated_price)
                budget_difference = abs(updated_price-budget_value)
                recommendation.append([budget_difference, append_string])

        for feature in ordinal_features:
            counter = 0
            while (updated_buyer[feature][0]<housing[feature].max()) & (counter<2):
                updated_buyer = buyer_x.copy()
                counter=counter +1
                updated_buyer[feature]=updated_buyer[feature]+counter
                updated_price = np.exp(clf.predict(updated_buyer)[0])
                difference = abs(predicted_price-updated_price)
                append_string = 'Setting ' + feature + '=' + str(updated_buyer[feature][0]) + '-----Increased cost: ${0:,.2f}'.format(difference) + '-----Predicted Price: ${0:,.2f}'.format(updated_price)
                budget_difference = abs(updated_price-budget_value)
                recommendation.append([budget_difference, append_string])
        
        for feature in area_features:
            if feature == 'LogGrLivArea':
                feature_text = 'Gross Living Area'
            else:
                feature_text = 'Lot Area'
            counter = 0
            while (abs(np.exp(updated_buyer[feature][0])- np.exp(housing[feature].max()))>100) & (counter<2):
                updated_buyer = buyer_x.copy()
                counter=counter +1
                new_area = np.exp(updated_buyer[feature][0])+counter*100
                updated_buyer[feature]=np.log(new_area)
                updated_price = np.exp(clf.predict(updated_buyer)[0])
                difference = abs(predicted_price-updated_price)
                append_string = 'Setting' + feature_text + '= {:.0f}'.format(new_area) + '-----Increased cost: ${0:,.2f}'.format(difference) + '-----Predicted Price: ${0:,.2f}'.format(updated_price)
                budget_difference = abs(updated_price-budget_value)
                recommendation.append([budget_difference, append_string])
        
        counter = 0
        while (counter<2) & (updated_buyer['Neighborhood'][0]<28):
            counter=counter +1
            updated_buyer = buyer_x.copy()
            updated_buyer['Neighborhood']=updated_buyer['Neighborhood']+counter
            updated_price = np.exp(clf.predict(updated_buyer)[0])
            difference = abs(predicted_price-updated_price)
            new_neighborhood = Neighborhood_dict[updated_buyer['Neighborhood'][0]] 
            append_string = 'Look for homes in ' + new_neighborhood + '-----Increased cost: ${0:,.2f}'.format(difference) + '-----Predicted Price: ${0:,.2f}'.format(updated_price)
            budget_difference = abs(updated_price-budget_value)
            recommendation.append([budget_difference, append_string])

        counter = 0
        while (counter<2) & (updated_buyer['BldgType'][0]<5):
            counter=counter +1
            updated_buyer = buyer_x.copy()
            updated_buyer['BldgType']=updated_buyer['BldgType']+counter
            updated_price = np.exp(clf.predict(updated_buyer)[0])
            difference = abs(predicted_price-updated_price)
            new_BldgType = BldgType_dict[updated_buyer['BldgType'][0]] 
            append_string = 'Looking for ' + new_BldgType + '-----Increased cost: ${0:,.2f}'.format(difference) + '-----Predicted Price: ${0:,.2f}'.format(updated_price)
            budget_difference = abs(updated_price-budget_value)
            recommendation.append([budget_difference, append_string])

    recommendation_string = [rec[1] for rec in sorted(recommendation)]
    print(recommendation_string)

    return 'Predicted Price: ${0:,.2f}'.format(predicted_price), over_under_budget, html.Ol([html.Li(x) for x in recommendation_string[:10]])

if __name__ == '__main__':
    app.run_server(debug=True)