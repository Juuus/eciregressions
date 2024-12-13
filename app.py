import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer
import requests
from io import BytesIO
import warnings
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

warnings.filterwarnings("ignore", category=RuntimeWarning)

app = dash.Dash(__name__)

# Access the Flask server object
server = app.server

# -----------------------------------------------
# Data paths and options
# -----------------------------------------------
dataframes = {
    "hc": 'data/human_capital.xlsx',
    "cc": 'data/cc.xlsx',
    "ge": 'data/ge.xlsx',
    "pv": 'data/pv.xlsx',
    "rl": 'data/rl.xlsx',
    "rq": 'data/rq.xlsx',
    "va": 'data/va.xlsx',
    "edi": 'data/edi-data.xlsx',
    "nat_res": 'data/data_natural_resources.xlsx'
}

eci_options_available = ['eci_trade', 'eci_tech', 'eci_research']

# -----------------------------------------------
# Layout
# -----------------------------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("Input Section"),
        html.Label("Select ECI type:"),
        dcc.RadioItems(
            id='eci-radio',
            options=[
               {'label': 'ECI (trade)', 'value': 'eci_trade'},
               {'label': 'ECI (tech)', 'value': 'eci_tech'},
               {'label': 'ECI (research)', 'value': 'eci_research'}
            ],
            value='eci_trade',
            className='my-radio-group',
            labelStyle={'display': 'inline-block', 'margin-right': '12px'}
        ),
        html.Br(),
        html.Label("Select covariates:"),
        dcc.Dropdown(
            id='dataframes-dropdown',
            options=[
            {'label': 'Years of schooling', 'value': 'hc'},
            {'label': 'Control of corruption', 'value': 'cc'},
            {'label': 'Government efficiency', 'value': 'ge'},
            {'label': 'Political stability', 'value': 'pv'},
            {'label': 'Rule of law', 'value': 'rl'},
            {'label': 'Regulatory quality', 'value': 'rq'},
            {'label': 'Voice and accountability', 'value': 'va'},
            {'label': 'Economic Diversification Index', 'value': 'edi'},
            {'label': 'Natural resources exports', 'value': 'nat_res'}
        ],
            value=["cc", "ge"],
            multi=True,
            style={'width': '300px'}
        ),
        html.Br(),
        html.Label("Initial year:"),
        dcc.Input(
            id='initial-year-input',
            type='number',
            value=1999,
            min=1999, max=2015, step=1
        ),
        html.Br(), html.Br(),
        html.Label("Time period (in years):"),
        dcc.Input(
            id='time-period-input',
            type='number',
            value=10,
            min=1, max=10, step=1
        ),
        html.Br(), html.Br(),
        html.Div(
        html.Button("Run Regression", id='run-button', n_clicks=0),
        style={'textAlign': 'center', 'marginTop': '20px'}
    ),
    ], style={
        'width': '35%',
        'display': 'inline-block',
        'verticalAlign': 'top',
        'padding': '20px',
        'boxSizing': 'border-box',
        'borderRight': '1px solid #ccc',
    }),

    html.Div([
        html.H2("Regression Results"),
        html.Div(id='output-container', style={'overflow':'auto'})
    ], style={
        'width': '68%',
        'display': 'inline-block',
        'verticalAlign': 'top',
        'padding': '20px',
        'boxSizing': 'border-box'
    })
], style={'display': 'flex'})

# -----------------------------------------------
# Callback
# -----------------------------------------------
@app.callback(
    Output('output-container', 'children'),
    Input('run-button', 'n_clicks'),
    State('eci-radio', 'value'),
    State('dataframes-dropdown', 'value'),
    State('initial-year-input', 'value'),
    State('time-period-input', 'value')
)
def run_regression(n_clicks, selected_eci, selected_dataframes_list, initial_year, T):
    if n_clicks == 0:
        return ""

    # -----------------------------------------------
    # Data Preparation
    # -----------------------------------------------
    years = list(range(initial_year, 2019, T))

    # Download GDP data
    gdp_url = "https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.PP.KD?downloadformat=excel"
    response = requests.get(gdp_url)
    gdp_pc = pd.read_excel(BytesIO(response.content), sheet_name="Data", header=3)
    gdp_pc.rename(columns={'Country Code': 'Country'}, inplace=True)
    years_data = [col for col in gdp_pc.columns if col.isdigit() and int(col) >= 1995]
    columns_to_keep = ['Country'] + years_data
    gdp_pc = gdp_pc[columns_to_keep]

    # Load ECI data
    eci_trade_raw = pd.read_csv('data/Data-ECI-Trade.csv')
    eci_tech_raw = pd.read_csv('data/Data-ECI-Technology.csv')
    eci_research_raw = pd.read_csv('data/Data-ECI-Research.csv')

    # Load selected dataframes
    def load_dataframes(selected_names):
        loaded_dataframes = {}
        for name in selected_names:
            if name in dataframes:
                loaded_dataframes[name] = pd.read_excel(dataframes[name])
        return loaded_dataframes

    loaded_dfs = load_dataframes(selected_dataframes_list)


    eci_trade = eci_trade_raw.copy()
    eci_tech = eci_tech_raw.copy()
    eci_research = eci_research_raw.copy()

    def reshape_df(df, years, value_name):
        df.columns = df.columns.astype(str)
        df = df[['Country'] + list(map(str, years))]
        df = df.melt(id_vars=['Country'], var_name='Year', value_name=value_name)
        return df

    # Calculate growth
    growth_df = gdp_pc[['Country']].copy()
    for year in gdp_pc.columns[1:-T]:
        future_year = str(int(year) + T)
        if future_year in gdp_pc.columns:
            growth_df[year] = 100*((gdp_pc[future_year] / gdp_pc[year])** (1 / T)  - 1)
        else:
            growth_df[year] = np.nan

    # Reshape ECI
    eci_trade_reshaped = reshape_df(eci_trade, years, 'eci_trade')
    eci_research_reshaped = reshape_df(eci_research, years, 'eci_research')
    eci_tech_reshaped = reshape_df(eci_tech, years, 'eci_tech')
    gdp_pc_reshaped = reshape_df(gdp_pc, years, 'GDP_Per_Capita')
    growth_reshaped = reshape_df(growth_df, years, 'growth')

    eci_options_dict = {
        'eci_trade': eci_trade_reshaped,
        'eci_tech': eci_tech_reshaped,
        'eci_research': eci_research_reshaped
    }

    if selected_eci not in eci_options_dict:
        return html.Div("Invalid ECI selection.", style={'color': 'red'})

    selected_eci_df = eci_options_dict[selected_eci]

    def reshape_loaded_dataframes(loaded_dataframes, years):
        reshaped_dataframes = {}
        for name, df in loaded_dataframes.items():
            reshaped_name = f"{name}_reshaped"
            reshaped_dataframes[reshaped_name] = reshape_df(df, years, name)
        return reshaped_dataframes

    reshaped_dfs = reshape_loaded_dataframes(loaded_dfs, years)

    # Merge
    regression_df = selected_eci_df.merge(gdp_pc_reshaped, on=['Country', 'Year'], how='left') \
                                   .merge(growth_reshaped, on=['Country', 'Year'], how='left')

    for name, df in reshaped_dfs.items():
        regression_df = regression_df.merge(df, on=['Country', 'Year'], how='left')

    # Transform
    regression_df['gdp_pc'] = np.log(regression_df['GDP_Per_Capita'])
    regression_df['gdp_pc'] = zscore(regression_df['gdp_pc'], nan_policy='omit')
    regression_df = regression_df.dropna()

    # Regression
    covariate_order = [selected_eci] + selected_dataframes_list + ['gdp_pc']
    regression_df['Year'] = pd.Categorical(regression_df['Year'])

    model_formulas = {}
    model_counter = 1
    # Model 1
    model_formulas[model_counter] = 'growth ~ gdp_pc + C(Year)'
    model_counter += 1
    # Model 2
    model_formulas[model_counter] = f'growth ~ gdp_pc + C(Year) + {selected_eci}'
    model_counter += 1

    # Add models for each df
    for cov in selected_dataframes_list:
        model_formulas[model_counter] = f'growth ~ gdp_pc + C(Year) + {cov}'
        model_counter += 1
        model_formulas[model_counter] = f'growth ~ gdp_pc + C(Year) + {selected_eci} + {cov}'
        model_counter += 1

    # Model with all selected dataframes (no ECI)
    model_formulas[model_counter] = 'growth ~ gdp_pc + C(Year) + ' + ' + '.join(selected_dataframes_list)
    model_counter += 1
    # Model with all selected dataframes + ECI
    model_formulas[model_counter] = f'growth ~ gdp_pc + C(Year) + {selected_eci} + ' + ' + '.join(selected_dataframes_list)

    models = {}
    for model_num, formula in model_formulas.items():
        model = smf.ols(formula, data=regression_df).fit(cov_type='HC1')
        models[model_num] = model

    stargazer = Stargazer([models[i] for i in sorted(models.keys())])
    stargazer.covariate_order(covariate_order)

    # Map the raw variable names to more descriptive labels
    rename_dict = {
        'eci_trade': 'ECI (trade)',
        'eci_tech': 'ECI (tech)',
        'eci_research': 'ECI (research)',
        'hc': 'Years of schooling',
        'cc': 'Control of corruption',
        'ge': 'Government efficiency',
        'pv': 'Political stability',
        'rl': 'Rule of law',
        'rq': 'Regulatory quality',
        'va': 'Voice & accountability',
        'edi': 'Economic Diversification Index',
        'nat_res': 'Natural resources exports',
        'gdp_pc': 'Log of initial GDP per capita'
    }

    stargazer.rename_covariates(rename_dict)
    # Create a string representing the intervals, e.g. "1999-2009, 2009-2019"
    intervals_str = ", ".join([f"{y}-{y+T}" for y in years])

    # Construct your dependent variable label
    dep_var_label = f"Annualized growth of GDP per capita (in PPP constant 2021 USD)<br>({intervals_str})"

    # After generating the Stargazer HTML:
    html_output = stargazer.render_html()

    # Replace the default dependent variable line with your custom label
    html_output = html_output.replace(
        "<em>Dependent variable: growth</em>",
        f"<em>Dependent variable: {dep_var_label}</em>"
    )

    return html.Div([
        html.Iframe(
            srcDoc=html_output,
            style={'width': '100%', 'height': '600px', 'border': 'none'}
        )
    ])

if __name__ == '__main__':
    app.run_server(debug=True)