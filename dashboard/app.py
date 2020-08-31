import os

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection

app = dash.Dash(__name__, 
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ],
            external_stylesheets=[dbc.themes.LUMEN])
server = app.server
app.title = 'Ames Housing Tool'

### Pre-Processing ###

# Import clean data 
housing = pd.read_pickle('./housing_data.pkl')

# Feature Selection
model_cols = [ 'LogGrLivArea', 'LogLotArea', 'OverallQual', 'OverallCond',
       'Neighborhood', 'BldgType', 'NumBath', 'GarageCars', 
       'Finished Basement','Finished Garage', 'Fire Place', 'Porch', 'Deck', 'Attached Garage',
       'Great Electric', 'Great Heat', 'Central Air']

x = housing[model_cols]
y = housing.LogSalePrice


### Train model ###
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=.2, random_state=0)

clf = linear_model.Lasso(alpha=9.326033468832199e-05)
clf.fit(x_train, y_train)

# 
def get_price_diff(updated_buyer,predicted,budget):
    updated_price = np.exp(clf.predict(updated_buyer)[0])
    difference = abs(predicted-updated_price)
    budget_difference = abs(budget-updated_price)
    return updated_price, difference, budget_difference

### UI-components ###
app.layout = html.Div(
    children=[
        # Description
        dbc.Row(
            dbc.Col(
                html.H3(children='Ames Housing Price Predictor'),
                width={'size':6}
            )
        ),
        html.Br(),
        dbc.Row(
            dbc.Col(
                html.H4(children='House hunter toolkit to find the perfect house for their budget.'),
                width={"size": 6}
            )
        ),
        dbc.Row(
            dbc.Col(
                html.P(children='This tool uses a lasso-penalized linear model trained with housing sales data from Ames, IA from 2016~2010.'),
                width={"size": 6}
            )
        ),
        html.Br(),

        dbc.Row(
            [
                # First column
                dbc.Col(
                    [
                        # Inputs
                        dbc.FormGroup(
                            [
                                dbc.Label("Budget"),
                                daq.NumericInput(
                                    id='budget',
                                    value=100000,
                                    min=12789, max=400000,
                                    size=200
                                ),
                                dbc.FormText(
                                    "Please enter your budget in $",
                                    color="secondary",
                                ),
                                html.Br(),
                                dbc.Label("Gross Living Area (sqft)"),
                                daq.NumericInput(
                                    id='GrLivArea',
                                    value=1000,
                                    min=334, max=4676,
                                    size=200
                                ),
                                dbc.FormText(
                                    "Total above ground living area",
                                    color="secondary",
                                ),
                                html.Br(),
                                dbc.Label("Lot Area (sqft)"),
                                daq.NumericInput(
                                    id='LotArea',
                                    value=1000,
                                    min=0, max=215000,
                                    size=200
                                ),
                                dbc.FormText(
                                    "Total outside lot area",
                                    color="secondary",
                                ),
                                html.Br(),
                                dbc.Label("Overall quality"),
                                daq.NumericInput(
                                    id='overall_qual',
                                    value=7,
                                    min=1, max=10,
                                    size=200    
                                ),
                                dbc.FormText(
                                    "Overall material and finish of the house",
                                    color="secondary",
                                ),
                                html.Br(),
                                dbc.Label("Overall condition"),
                                daq.NumericInput(
                                    id='overall_cond',
                                    value=7,
                                    min=1, max=10,
                                    size=200
                                ),
                                dbc.FormText(
                                    "Overall condition of the house",
                                    color="secondary",
                                ),
                                html.Br(),
                            ]
                        )
                    ],
                    width='auto'
                ),

                # Second column
                dbc.Col(
                    # Inputs
                    dbc.FormGroup(
                        [   
                            dbc.Label("Neighborhood"),
                            dcc.Dropdown(
                                id='neighborhood',
                                clearable=False,
                                options=[
                                    {'label': 'Meadow Village', 'value': 1},
                                    {'label': 'Briardale', 'value': 2},
                                    {'label': 'Iowa DOT and Rail Road', 'value': 3},
                                    {'label': 'Brookside', 'value': 4},
                                    {'label': 'Old Town', 'value': 5},
                                    {'label': 'Edwards', 'value': 6},
                                    {'label': 'South & West of Iowa State University', 'value': 7},
                                    {'label': 'Sawyer', 'value': 8},
                                    {'label': 'Northpark Villa', 'value': 9},
                                    {'label': 'Bluestem', 'value': 10},
                                    {'label': 'North Ames', 'value': 11},
                                    {'label': 'Mitchell', 'value': 12},
                                    {'label': 'Sawyer West', 'value': 13},
                                    {'label': 'Gilbert', 'value': 14},
                                    {'label': 'Northwest Ames', 'value': 15},
                                    {'label': 'Greens', 'value': 16},
                                    {'label': 'Blmngtn', 'value': 17},
                                    {'label': 'College Creek', 'value': 18},
                                    {'label': 'Crawford', 'value': 19},
                                    {'label': 'Clear Creek', 'value': 20},
                                    {'label': 'Somerset', 'value': 21},
                                    {'label': 'Timberland', 'value': 22},
                                    {'label': 'Veenker', 'value': 23},
                                    {'label': 'Stone Brook', 'value': 24},
                                    {'label': 'Northridge Heights', 'value': 25},
                                    {'label': 'Northridge', 'value': 26}
                                ],
                                value=11
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
                                value=2,
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
                            dbc.Checklist(
                                id='boolean_features',
                                options=[
                                    {'label': 'Finished Basement', 'value': 'Finished Basement'},
                                    {'label': 'Finished Garage', 'value': 'Finished Garage'},
                                    {'label': 'Fire Place', 'value': 'Fire Place'},
                                    {'label': 'Porch', 'value': 'Porch'},
                                    {'label': 'Deck', 'value': 'Deck'},
                                    {'label': 'Attached Garage', 'value': 'Attached Garage'},
                                    {'label': 'Great Electric', 'value': 'Great Electric'},
                                    {'label': 'Great Heat', 'value': 'Great Heat'},
                                    {'label': 'Central Air', 'value': 'Central Air'},
                                ],
                                value=['Deck','Great Electric','Great Heat','Central Air'],
                                switch=True
                            )
                        ]
                    ),
                    width='auto'
                ),

                # Third column
                dbc.Col(
                    [
                        # Outputs
                        html.H4(id='predicted_price'),
                        html.Br(),
                        html.H4(id='over_under_budget'),
                        html.Br(),
                        dbc.Alert(
                            [
                                html.H4('Recommendations:'),
                                html.Ul(
                                    id='recommendations',
                                    style={'font-size': '17px'}
                                )
                            ],
                            color='light'
                        )
                    ],
                    width='auto'
                )
            ]
        )
    ],
    style={'marginLeft': 50, 'marginTop':50}
)


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
def update_recommendations(budget_value, GrLivArea_value, LotArea_value, OverallQual_value, OverallCond_value, Neighborhood_value, BldgType_value, NumBath_value, GarageCars_value, boolean_features_value):

    # Define grouped features, Initialize recommendation arrary
    recommendation   = []
    boolean_features = ['Finished Basement','Finished Garage', 'Fire Place', 'Porch', 'Deck', 'Attached Garage', 'Great Electric', 'Great Heat', 'Central Air']
    
    ordinal_features = ['OverallQual','OverallCond','GarageCars','NumBath']
    area_features    = ['LogGrLivArea', 'LogLotArea']

    # Create dictionary for feature interpretation
    Neighborhood_dict = {1:'Meadow Village',2:'Briardale', 3:'Iowa DOT and Rail Road', 4:'Brookside', 5:'Old Town', 6:'Edwards',7:'South & West of Iowa State University', 8: 'Sawyer',\
                           9:'Northpark Villa', 10:'Bluestem', 11:'North Ames', 12:'Mitchell', 13:'Sawyer West', 14:'Gilbert', 15:'Northwest Ames', 16:'Greens', 17:'Blmngtn',\
                           18:'College Creek', 19:'Crawford', 20:'Clear Creek',21: 'Somerset', 22:'Timberland',23: 'Veenker',24: 'Stone Brook',25:'Northridge Heights', 26:'Northridge'}
    BldgType_dict = {1:'2 Family Condo',2:'Town House (Middle)', 3:'Duplex', 4:'1 Family', 5:'Townhouse (End)'}

    # Grab values for boolean features from input
    HasFinBsmt=0
    HasFinGarage=0
    HasFireplace=0
    HasPorch=0
    HasDeck=0
    AttachedGarage=0
    GreatElectric=0
    GreatHeat=0
    CentralAir=0
    if "Finished Basement" in boolean_features_value:
        HasFinBsmt=1
    if "Finished Garage" in boolean_features_value:
        HasFinGarage=1
    if "Fire Place" in boolean_features_value:
        HasFireplace=1
    if "Porch" in boolean_features_value:
        HasPorch=1
    if "Deck" in boolean_features_value:
        HasDeck=1
    if "Attached Garage" in boolean_features_value:
        AttachedGarage=1
    if "Great Electric" in boolean_features_value:
        GreatElectric=1
    if "Great Heat" in boolean_features_value:
        GreatHeat=1
    if "Central Air" in boolean_features_value:
        CentralAir=1

    # Create feature dataframe with input info
    buyer_data = [[np.log(GrLivArea_value), np.log(LotArea_value), OverallQual_value, OverallCond_value, Neighborhood_value, BldgType_value, NumBath_value, GarageCars_value,\
        HasFinBsmt, HasFinGarage, HasFireplace, HasPorch, HasDeck, AttachedGarage, GreatElectric, GreatHeat, CentralAir]]
    buyer_x = pd.DataFrame(data = buyer_data, columns = model_cols)

    # Use model to predict price
    predicted_price = np.exp(clf.predict(buyer_x)[0])

    # Within $1000 of budget
    if abs(predicted_price-budget_value)<1000:
        over_under_budget='Congratulations. You are within $1000 of your budget.'

    # Overbudget. Lower cost.
    elif predicted_price>budget_value:
        over_under_budget = 'You are Over Budget.'

        # Make True boolean features False
        for feature in boolean_features:
            updated_buyer = buyer_x.copy()
            if updated_buyer[feature][0]>0:
                updated_buyer[feature]=0
                updated_price, difference, budget_difference = get_price_diff(updated_buyer,predicted_price,budget_value)
                append_string = 'Remove ' + feature + ' to save ${0:,.0f}'.format(difference) + '. New predicted price: ${0:,.0f}.'.format(updated_price)
                recommendation.append([budget_difference, append_string])

        # Decrease ordinal features by 1
        for feature in ordinal_features:
            counter = 0
            while (updated_buyer[feature][0]>1) & (counter<2):
                updated_buyer = buyer_x.copy()
                counter=counter +1
                updated_buyer[feature]=updated_buyer[feature]-counter
                updated_price, difference, budget_difference = get_price_diff(updated_buyer,predicted_price,budget_value)
                append_string = 'Reduce ' + feature + ' to ' + str(updated_buyer[feature][0]) + ' to save ${0:,.0f}'.format(difference) + '. New predicted price: ${0:,.0f}'.format(updated_price)
                recommendation.append([budget_difference, append_string])
        
        # Decrease area features by 100
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
                updated_price, difference, budget_difference = get_price_diff(updated_buyer,predicted_price,budget_value)
                append_string = 'Reduce ' + feature_text + ' to {:.0f}sqft'.format(new_area) + ' to save ${0:,.0f}'.format(difference) + '. New predicted price: ${0:,.0f}'.format(updated_price)
                recommendation.append([budget_difference, append_string])

        # Decrease neighborhood by 1
        counter = 0
        while (counter<2) & (updated_buyer['Neighborhood'][0]>1):
            counter=counter +1
            updated_buyer = buyer_x.copy()
            updated_buyer['Neighborhood']=updated_buyer['Neighborhood']-counter
            updated_price, difference, budget_difference = get_price_diff(updated_buyer,predicted_price,budget_value)
            new_neighborhood = Neighborhood_dict[updated_buyer['Neighborhood'][0]] 
            append_string = 'Change neighborhood to ' + new_neighborhood + ' to save ${0:,.0f}'.format(difference) + '. New predicted price: ${0:,.0f}'.format(updated_price)
            recommendation.append([budget_difference, append_string])

        # Decrease building type by 1
        counter = 0
        while (counter<2) & (updated_buyer['BldgType'][0]>1):
            counter=counter +1
            updated_buyer = buyer_x.copy()
            updated_buyer['BldgType']=updated_buyer['BldgType']-counter
            updated_price, difference, budget_difference = get_price_diff(updated_buyer,predicted_price,budget_value)
            new_BldgType = BldgType_dict[updated_buyer['BldgType'][0]] 
            append_string = 'Change building type to ' + new_BldgType + ' to save ${0:,.0f}'.format(difference) + '. New predicted price: ${0:,.0f}'.format(updated_price)
            recommendation.append([budget_difference, append_string])

    # Under Budget. Increase cost.
    elif predicted_price<budget_value:
        over_under_budget = 'You are Under Budget.'

        # Make False Boolean features True
        for feature in boolean_features:
            updated_buyer = buyer_x.copy()
            if updated_buyer[feature][0]==0:
                updated_buyer[feature]=1
                updated_price, difference, budget_difference = get_price_diff(updated_buyer,predicted_price,budget_value)
                append_string = 'Add ' + feature + ' to increase target by ${0:,.0f}'.format(difference) + '. New predicted price: ${0:,.0f}.'.format(updated_price)
                recommendation.append([budget_difference, append_string])

        # Increase ordinal features by 1
        for feature in ordinal_features:
            counter = 0
            while (updated_buyer[feature][0]<housing[feature].max()) & (counter<2):
                updated_buyer = buyer_x.copy()
                counter=counter +1
                updated_buyer[feature]=updated_buyer[feature]+counter
                updated_price, difference, budget_difference = get_price_diff(updated_buyer,predicted_price,budget_value)
                append_string = 'Increase ' + feature + ' to ' + str(updated_buyer[feature][0]) + ' to increase target by ${0:,.0f}'.format(difference) + '. New predicted price: ${0:,.0f}.'.format(updated_price)
                recommendation.append([budget_difference, append_string])
        
        # Increase area features by 100
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
                updated_price, difference, budget_difference = get_price_diff(updated_buyer,predicted_price,budget_value)
                append_string = 'Increase ' + feature_text + ' to {:.0f}'.format(new_area) + 'sqft to increase target by ${0:,.0f}'.format(difference) + '. New predicted price: ${0:,.0f}.'.format(updated_price)
                recommendation.append([budget_difference, append_string])
        
        # Increase neighboorhood by 1
        counter = 0
        while (counter<2) & (updated_buyer['Neighborhood'][0]<28):
            counter=counter +1
            updated_buyer = buyer_x.copy()
            updated_buyer['Neighborhood']=updated_buyer['Neighborhood']+counter
            updated_price = np.exp(clf.predict(updated_buyer)[0])
            difference = abs(predicted_price-updated_price)
            new_neighborhood = Neighborhood_dict[updated_buyer['Neighborhood'][0]] 
            append_string = 'Change neighborhood to ' + new_neighborhood + ' to increase target by ${0:,.2f}'.format(difference) + '. New predicted price: ${0:,.2f}.'.format(updated_price)
            budget_difference = abs(updated_price-budget_value)
            recommendation.append([budget_difference, append_string])

        # Increase building type by 1
        counter = 0
        while (counter<2) & (updated_buyer['BldgType'][0]<5):
            counter=counter +1
            updated_buyer = buyer_x.copy()
            updated_buyer['BldgType']=updated_buyer['BldgType']+counter
            updated_price, difference, budget_difference = get_price_diff(updated_buyer,predicted_price,budget_value)
            new_BldgType = BldgType_dict[updated_buyer['BldgType'][0]] 
            append_string = 'Change building type to ' + new_BldgType + ' to increase your target by ${0:,.0f}'.format(difference) + '. New predicted price: ${0:,.0f}.'.format(updated_price)
            recommendation.append([budget_difference, append_string])

    # Choose top 10 recommendations by closest updated price to budget
    recommendation_string = [rec[1] for rec in sorted(recommendation)]

    return 'Predicted Price: ${0:,.0f}'.format(predicted_price), over_under_budget, html.Ol([html.Li(x) for x in recommendation_string[:10]])

if __name__ == '__main__':
    app.run_server(debug=True)