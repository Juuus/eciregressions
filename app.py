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
import itertools

scaler = MinMaxScaler()

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
    "nat_res": 'data/data_natural_resources.xlsx',
    "nat_rents": 'NY.GDP.TOTL.RT.ZS',
    "gcf": 'NE.GDI.FTOT.ZS',
    "money": 'FM.LBL.BMNY.GD.ZS',
    "inflation": 'FP.CPI.TOTL.ZG',
    "diversity_trade": 'data/diversity_trade.csv',
    "diversity_patents": 'data/diversity_patents.csv',
    "diversity_publications": 'data/diversity_publications.csv',
    "hhi_trade": 'data/hhi_trade.csv',
    "hhi_patents": 'data/hhi_patents.csv',
    "hhi_publications": 'data/hhi_publications.csv', 
    "entropy_trade": 'data/entropy_trade.csv',
    "entropy_patents": 'data/entropy_patents.csv',
    "entropy_publications": 'data/entropy_publications.csv',
    "intensity_trade": 'data/intensity_trade.csv',
    "intensity_patents": 'data/intensity_patents.csv',
    "intensity_publications": 'data/intensity_publications.csv'    
}

eci_options_available = ['eci_trade', 'eci_tech', 'eci_research']

# -----------------------------------------------
# Layout
# -----------------------------------------------

app.layout = html.Div([
    html.Div([
        html.H1("Inputs:"),
        html.Label("Select ECI type(s):"),
        dcc.Checklist(
            id='eci-checklist',
            options=[
               {'label': 'ECI (trade)', 'value': 'eci_trade'},
               {'label': 'ECI (tech)', 'value': 'eci_tech'},
               {'label': 'ECI (research)', 'value': 'eci_research'}
            ],
            value=['eci_trade'],  # Default selected
            labelStyle={'display': 'block', 'margin-bottom': '5px'}
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
            {'label': 'Natural rents (% of GDP)', 'value': 'nat_rents'},
            {'label': 'Natural resources exports per capita', 'value': 'nat_res'},
            {'label': 'Gross fixed capital (% of GDP)', 'value': 'gcf'},
            {'label': 'Broad money (% of GDP)', 'value': 'money'},
            {'label': 'Inflation', 'value': 'inflation'},
            {'label': 'Population', 'value': 'pop'},
            {'label': 'Diversity (trade)', 'value': 'diversity_trade'},
            {'label': 'Diversity (tech)', 'value': 'diversity_patents'},
            {'label': 'Diversity (research)', 'value': 'diversity_publications'},
            {'label': 'Exports per capita', 'value': 'intensity_trade'},
            {'label': 'Patents per capita', 'value': 'intensity_patents'},
            {'label': 'Publications per capita', 'value': 'intensity_publications'},
            {'label': 'HHI (trade)', 'value': 'hhi_trade'},
            {'label': 'HHI (tech)', 'value': 'hhi_patents'},
            {'label': 'HHI (research)', 'value': 'hhi_publications'},
            {'label': 'Entropy (trade)', 'value': 'entropy_trade'},
            {'label': 'Entropy (tech)', 'value': 'entropy_patents'},
            {'label': 'Entropy (research)', 'value': 'entropy_publications'}            
        ],
            value=["diversity_trade", "ge"],
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
        html.Label("Fixed Effects:"),
        dcc.Checklist(
            id='fixed-effects-toggle',
            options=[
                {'label': 'Time Fixed Effects', 'value': 'time'},
                {'label': 'Country Fixed Effects', 'value': 'country'}
            ],
            value=['time'],  # Default: time fixed effects toggled on
            style={'margin-bottom': '12px'}
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
    State('eci-checklist', 'value'),  # Updated ID
    State('dataframes-dropdown', 'value'),
    State('initial-year-input', 'value'),
    State('time-period-input', 'value'),
    State('fixed-effects-toggle', 'value')  # Fixed effects
)
def run_regression(n_clicks, selected_eci, selected_dataframes_list, initial_year, T, fixed_effects):
    if n_clicks == 0:
        return ""

    # -----------------------------------------------
    # Data Preparation
    # -----------------------------------------------
    years = list(range(initial_year, 2019, T))

    # Define the maximum allowed year
    max_year = 2023 - T

    # Create a new list with years <= max_year
    years = [year for year in years if year <= max_year]

    # Download GDP data
    gdp_url = "https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.PP.KD?downloadformat=excel"
    response = requests.get(gdp_url)
    gdp_pc = pd.read_excel(BytesIO(response.content), sheet_name="Data", header=3)
    gdp_pc.rename(columns={'Country Code': 'Country'}, inplace=True)
    years_data = [col for col in gdp_pc.columns if col.isdigit() and int(col) >= 1995]
    columns_to_keep = ['Country'] + years_data
    gdp_pc = gdp_pc[columns_to_keep]

    pop_url = "https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=excel"
    response = requests.get(pop_url)
    pop = pd.read_excel(BytesIO(response.content), sheet_name="Data", header=3)
    pop.rename(columns={'Country Code': 'Country'}, inplace=True)
    years_data = [col for col in pop.columns if col.isdigit() and int(col) >= 1995]
    columns_to_keep = ['Country'] + years_data
    pop = pop[columns_to_keep]


    # Load ECI data
    #eci_trade_raw = pd.read_csv('data/Data-ECI-Trade.csv')
    eci_trade_raw = pd.read_csv('data/eci_trade.csv')
    #eci_tech_raw = pd.read_csv('data/Data-ECI-Technology.csv')
    eci_tech_raw = pd.read_csv('data/eci_patents.csv')
    #eci_research_raw = pd.read_csv('data/Data-ECI-Research.csv')
    eci_research_raw = pd.read_csv('data/eci_publications.csv')

        # Define load_dataframes function
    def load_dataframes(selected_names):
        loaded_dataframes = {}
        for name in selected_names:
            if name in dataframes:
                # Check if the name requires API fetching
                if name in ["nat_rents", "gcf", "money", "inflation"]:
                    indicator_code = dataframes[name]
                    dataframe_url = f"https://api.worldbank.org/v2/en/indicator/{indicator_code}?downloadformat=excel"
                    response = requests.get(dataframe_url)
                    if response.status_code == 200:
                        loaded_data = pd.read_excel(BytesIO(response.content), sheet_name="Data", header=3)
                        loaded_data.rename(columns={'Country Code': 'Country'}, inplace=True)
                        years_data = [col for col in loaded_data.columns if col.isdigit() and int(col) >= 1995]
                        columns_to_keep = ['Country'] + years_data
                        loaded_dataframes[name] = loaded_data[columns_to_keep]
                    else:
                        print(f"Failed to fetch data for {name}: {response.status_code}")
                elif name in ["eci_trade", "eci_tech", "eci_research"]:
                    try:
                        # Attempt to read as an Excel file
                        datax = pd.read_excel(dataframes[name])
                    except Exception as e:
                        print(f"Failed to read {name} as Excel. Trying CSV. Error: {e}")
                        try:
                            # If Excel read fails, attempt to read as a CSV
                            datax = pd.read_csv(dataframes[name])
                        except Exception as e_csv:
                            print(f"Failed to read {name} as CSV as well. Error: {e_csv}")
                    for column in datax.columns:
                        if datax[column].dtype in ['float64', 'int64']:  # Only scale numeric columns      
                            # Transform only the non-NaN values
                            scaler.fit(datax[column].values.reshape(-1, 1))
                            scaled_values = scaler.transform(datax[column].values.reshape(-1, 1))

                            # Create a new column with scaled values, retaining NaNs
                            datax[column] = scaled_values
                    
                    loaded_dataframes[name] = datax.copy()
                    
                else:
                    # Load from local file
                    try:
                        # Attempt to read as an Excel file
                        loaded_dataframes[name] = pd.read_excel(dataframes[name])
                    except Exception as e:
                        print(f"Failed to read {name} as Excel. Trying CSV. Error: {e}")
                        try:
                            # If Excel read fails, attempt to read as a CSV
                            loaded_dataframes[name] = pd.read_csv(dataframes[name])
                        except Exception as e_csv:
                            print(f"Failed to read {name} as CSV as well. Error: {e_csv}")

        return loaded_dataframes  # Properly return the dictionary

    # Call load_dataframes and store the result
    loaded_dfs = load_dataframes(selected_dataframes_list)


    eci_trade = eci_trade_raw.copy()
    for column in eci_trade_raw.columns:
        if eci_trade_raw[column].dtype in ['float64', 'int64']:  # Only scale numeric columns      
            # Transform only the non-NaN values
            scaler.fit(eci_trade_raw[column].values.reshape(-1, 1))
            scaled_values = scaler.transform(eci_trade_raw[column].values.reshape(-1, 1))
            
            # Create a new column with scaled values, retaining NaNs
            eci_trade[column] = scaled_values
            
    eci_tech = eci_tech_raw.copy()
    for column in eci_tech_raw.columns:
        if eci_tech_raw[column].dtype in ['float64', 'int64']:  # Only scale numeric columns      
            # Transform only the non-NaN values
            scaler.fit(eci_tech_raw[column].values.reshape(-1, 1))
            scaled_values = scaler.transform(eci_tech_raw[column].values.reshape(-1, 1))
            
            # Create a new column with scaled values, retaining NaNs
            eci_tech[column] = scaled_values
            
    eci_research = eci_research_raw.copy()
    for column in eci_research_raw.columns:
        if eci_research_raw[column].dtype in ['float64', 'int64']:  # Only scale numeric columns      
            # Transform only the non-NaN values
            scaler.fit(eci_research_raw[column].values.reshape(-1, 1))
            scaled_values = scaler.transform(eci_research_raw[column].values.reshape(-1, 1))
            
            # Create a new column with scaled values, retaining NaNs
            eci_research[column] = scaled_values

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
    pop_reshaped = reshape_df(pop, years, 'pop')
    eci_options_dict = {
        'eci_trade': eci_trade_reshaped,
        'eci_tech': eci_tech_reshaped,
        'eci_research': eci_research_reshaped
    }


    # Start with a base DataFrame that has GDP, growth, population, etc.
    regression_df = gdp_pc_reshaped.merge(growth_reshaped, on=['Country', 'Year'], how='left') \
                                   .merge(pop_reshaped,    on=['Country', 'Year'], how='left')


    # Now loop over only the ECI variables you want
    for eci_var in selected_eci:
        # e.g., eci_var = 'eci_trade'
        if eci_var in eci_options_dict:
            regression_df = regression_df.merge(eci_options_dict[eci_var],
                                                on=['Country', 'Year'],
                                                how='left')
        else:
            print(f"Warning: {eci_var} not in eci_options_dict!")
            
            
    def reshape_loaded_dataframes(loaded_dataframes, years):
        reshaped_dataframes = {}
        for name, df in loaded_dataframes.items():
            reshaped_name = f"{name}_reshaped"
            reshaped_dataframes[reshaped_name] = reshape_df(df, years, name)
        return reshaped_dataframes

    reshaped_dfs = reshape_loaded_dataframes(loaded_dfs, years)



    for name, df in reshaped_dfs.items():
        regression_df = regression_df.merge(df, on=['Country', 'Year'], how='left')

    # Transform
    regression_df['gdp_pc'] = np.log(regression_df['GDP_Per_Capita'])

    target_columns = ["intensity_trade", "intensity_patents", "intensity_publications", "nat_res"]

    # Apply log transformation only if the column exists
    for column_name in target_columns:
        if column_name in regression_df.columns:
            regression_df[column_name] = np.log(regression_df[column_name] / regression_df['pop'])  



    regression_df['pop'] = np.log(regression_df['pop'])

    target_columns = ["nat_rents", "gcf", "money", "hc"]

    # Apply log transformation only if the column exists
    for column_name in target_columns:
        if column_name in regression_df.columns:
            regression_df[column_name] = np.log(regression_df[column_name])

    regression_df = regression_df.dropna(subset=selected_eci)

    # Number of rows before dropping
    original_len = len(regression_df['Country'].unique())

    # Drop missing
    regression_df = regression_df.dropna()

    # Number of rows after dropping
    after_len = len(regression_df['Country'].unique())

    # Fraction (or percentage) of lost data
    fraction_lost = (original_len - after_len) / original_len

    # -----------------------------------------------
    # Dynamically Create Regression Formulas
    # -----------------------------------------------
    #covariate_order = [selected_eci] + selected_dataframes_list + ['gdp_pc']

    # Initialize the list with individual ECI variables
    interaction_terms = selected_eci.copy()

    # Generate interaction terms for combinations of size 2 to n
    for i in range(2, len(selected_eci) + 1):
        # Generate all combinations of the current size
        for combo in itertools.combinations(selected_eci, i):
            # Join the variables with ':' to represent interaction
            interaction_term = ':'.join(combo)
            interaction_terms.append(interaction_term)

    # Final covariate_order list
    covariate_order = interaction_terms + selected_dataframes_list + ['gdp_pc']

    # Initialize base fixed effects terms
    fixed_effects_terms = []
    if 'time' in fixed_effects:
        fixed_effects_terms.append('C(Year)')
    if 'country' in fixed_effects:
        fixed_effects_terms.append('C(Country)')

    fixed_effects_str = " + ".join(fixed_effects_terms)  # Combine fixed effects

    model_formulas = {}
    model_counter = 1

    # Model 1: Base model
    model_formulas[model_counter] = f'growth ~ gdp_pc' + (f' + {fixed_effects_str}' if fixed_effects_str else "")
    model_counter += 1

    # All subsets of ECI with interactions
    for r in range(1, len(selected_eci) + 1):
        for combo in itertools.combinations(selected_eci, r):
            eci_part = " * ".join(combo)  # e.g. "eci_trade * eci_tech"
            formula = f"growth ~ gdp_pc + {eci_part}"
            if fixed_effects_str:
                formula += f" + {fixed_effects_str}"
            
            model_formulas[model_counter] = formula
            model_counter += 1

    # 3) Fit all models, pick the best by adjusted R^2
    models = {}
    best_model_num = None
    best_adj_r2 = float('-inf')

    for num, formula in model_formulas.items():
        mod = smf.ols(formula, data=regression_df).fit(cov_type='HC1')
        models[num] = mod
        
        pvalues = mod.pvalues

        non_year_pvalues = pvalues[~pvalues.index.str.startswith(('C(Year)', 'C(Country)'))]

        non_year_pvalues = non_year_pvalues.drop('Intercept', errors='ignore')
        
        # Check if all non-Year covariates are significant at 5% level
        if (non_year_pvalues < 0.05).all() and mod.rsquared_adj > best_adj_r2:
            best_adj_r2 = mod.rsquared_adj
            best_model_num = num

    best_model = models[best_model_num]
    best_formula = model_formulas[best_model_num]

    # Add models for each covariate
    for cov in selected_dataframes_list:
        model_formulas[model_counter] = f'growth ~ gdp_pc + {cov}' + (f' + {fixed_effects_str}' if fixed_effects_str else "")
        model_counter += 1
        model_formulas[model_counter] = f'{best_formula} + {cov}'
        model_counter += 1

    # Model with all covariates (no ECI)
    if len(selected_dataframes_list) > 1:
        all_covariates = ' + '.join(selected_dataframes_list)
        model_formulas[model_counter] = f'growth ~ gdp_pc + {all_covariates}' + (f' + {fixed_effects_str}' if fixed_effects_str else "")
        model_counter += 1

        # Model with all covariates + ECI
        model_formulas[model_counter] = f'{best_formula} + {all_covariates}'

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
        'gdp_pc': 'Log of initial GDP per capita',
        'nat_res': 'Natural resource exports per capita',
        'nat_rents': 'Log of natural rents (% of GDP) ',
        'gcf': 'Log of gross fixed capital (% of GDP)',
        'money': 'Log of broad money (% of GDP)',
        'inflation': 'Inflation',
        'pop': 'Log of population',
        'diversity_trade': 'Diversity (trade)',
        'diversity_patents': 'Diversity (tech)',
        'diversity_publications': 'Diversity (research)',
        'hhi_trade': 'HHI (trade)',
        'hhi_patents': 'HHI (tech)',
        'hhi_publications': 'HHI (research)',
        'entropy_trade': 'Entropy (trade)',
        'entropy_patents': 'Entropy (tech)',
        'entropy_publications': 'Entropy (research)',
        'intensity_trade': 'Log of exports per capita',
        'intensity_patents': 'Log of patents per capita',
        'intensity_publications': 'Log of publications per capita',
        'eci_trade:eci_tech': 'ECI (trade) x ECI (tech)',
        'eci_trade:eci_research': 'ECI (trade) x ECI (research)',    
        'eci_tech:eci_research': 'ECI (tech) x ECI (research)',
        'eci_trade:eci_tech:eci_research': 'ECI (trade) x ECI (tech) x ECI (research)'    
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


    # Build a note about lost data
    lost_data_note = (
        f"Data used: {after_len} of {original_len} countries "
        f"({fraction_lost:.2%} of countries dropped due to missing data)."
    )

    return html.Div([
        html.Iframe(
            srcDoc=html_output,
            style={'width': '100%', 'height': '600px', 'border': 'none'}
        ),
        html.P(lost_data_note)
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
