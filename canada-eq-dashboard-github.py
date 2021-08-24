# python 37
# elsa@promasta.com

import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import math
from scipy.stats import pareto, lognorm, gamma, binom, poisson, nbinom


"""
# Sniplet to load and clean data from canada.ca

# Load
df = pd.read_csv('http://ftp.maps.canada.ca/pub/nrcan_rncan/Earthquakes_Tremblement-de-terre/canadian-earthquakes_tremblements-de-terre-canadien/eqarchive-en.csv')

# Remove unnamed
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Set types
df.astype(dtype={'latitude' : float, 'longitude' : float, 'depth' : float, 'magnitude' : float, 'magnitude type' : str, 'place' : str}, copy=True)

# Remove eq < 4
df = df.loc[df['magnitude']>=4.0,:]
"""

# Load cleaned data from elsaburren/github (comment this line out if above sniplet is used) 
df = pd.read_csv('https://raw.githubusercontent.com/elsaburren/canada-eq-dashboard/main/eqarchive-en.csv')

# Set types
df.astype(dtype={'latitude' : float, 'longitude' : float, 'depth' : float, 'magnitude' : float, 'magnitude type' : str, 'place' : str}, copy=True)

# Define year and a count variable
df['datetime'] = pd.to_datetime(df['date'])
df['year'] = pd.DatetimeIndex(df['datetime']).year
df['count'] = 1

# Create necessary variables for the dashboard interface
year_min, year_max = df['year'].min(), df['year'].max()
years = [x for x in range(year_min, year_max+1)]
str_years = [str(x) for x in years]
format_frq = pd.DataFrame.from_dict(dict({'year':years, 'count':[0 for x in years]}))
magnitude_min, magnitude_max = df['magnitude'].min(), df['magnitude'].max()
magnitude_options = [x for x in range(math.floor(magnitude_min), math.ceil(magnitude_max))]

# Create functions for fitting
def create_survival(df,str_single_col, obj_dist):
    df_out = df[[str_single_col]].value_counts(normalize=True).to_frame('prob').sort_values(by=str_single_col,ascending=True)
    df_out['sf'] = 1.0 - df_out['prob'].cumsum(skipna=False)
    df_out['sf_type'] = 'observed'
    df_out = df_out.drop(columns='prob').reset_index(level=str_single_col)
    df_theoretical = pd.DataFrame()
    df_theoretical[str_single_col] = df_out.loc[:,str_single_col]
    df_theoretical['sf'] = df_theoretical[str_single_col].apply(lambda x:obj_dist.sf(x))
    df_theoretical['sf_type'] = obj_dist.type()
    df_out = pd.concat([df_out, df_theoretical])

    return df_out

# Classes
class Frequency:
    def __init__(self, **kwargs):
        self._mean = kwargs['mean'] if 'mean' in kwargs else 0.0
        self._variance = kwargs['variance'] if 'variance' in kwargs else kwargs['mean'] if 'mean' in kwargs else 0.0
        self._type = 'nbinom' if self._variance > self._mean else 'binom' if self._variance < self._mean else 'poisson'
        
    def mean(self, v=None):
        if v: self._mean = v
        try: return self._mean
        except AttributeError: return None

    def variance(self, v=None):
        if v: self._variance = v
        try: return self._variance
        except AttributeError: return None
    
    def sd(self):
        try: return math.sqrt(self._variance)
        except AttributeError: return None
    
    def type(self):
        try: return self._type
        except AttributeError: return None
        
    def sf(self, k=0):
        if self._type == 'poisson':
            return poisson.sf(k, self._mean, loc=0)
        elif self._type == 'nbinom':
            n = self._mean * self._mean  / (self._variance - self._mean)
            p = self._mean / self._variance
            return nbinom.sf(k, n, p)
        else:
            n = self._mean / (1.0 - (self._variance / self._mean))
            p = 1.0 - self._variance / self._mean
            return binom.sf(k, n, p)
        
class Severity:
    def __init__(self, **kwargs):
        self._shape = kwargs['shape'] if 'shape' in kwargs else 1.0
        self._scale = kwargs['scale'] if 'scale' in kwargs else 1.0
        self._loc   = kwargs['loc']   if 'loc' in kwargs else 0.0
        self._type  = kwargs['type']  if 'type'  in kwargs else 'undefined'
        
    def shape(self, v=None):
        if v: self._shape = v
        try: return self._shape
        except AttributeError: return None

    def scale(self, v=None):
        if v: self._scale = v
        try: return self._scale
        except AttributeError: return None
    
    def loc(self, v=None):
        if v: self._loc = v
        try: return self._loc
        except AttributeError: return None

    def type(self, v=None):
        allowed_types = ('pareto', 'gamma', 'lognorm')
        if v in allowed_types: self._type = v
        try: return self._type
        except AttributeError: return None
        
    def sf(self, x):
        if self._type == 'pareto':
            return pareto.sf(x, self._shape, 0.0, self._scale)
        elif self._type == 'gamma':
            return gamma.sf(x, self._shape, self._loc, self._scale)
        elif self._type == 'lognorm':
            return lognorm.sf(x, self._shape, self._loc, 1.0)
        else:
            return 'undefined'
    
# App Layout
app = dash.Dash(__name__, meta_tags=[{'name':'viewport', 'content':'width=device-width, initial-scale=1.0'}],)

server = app.server

app.layout = html.Div([
    html.H1('Canadian Earthquakes Dashboard', style={'text-align': 'left'}),
    
    html.Table([html.Div('Select Magnitude Threshold: ', style={'float':'left', 'padding': '5px 5px 5px 3px', 'border': '1px solid rgb(200,200,200)', 'width':'350px', 'height':'24px'}), dcc.Dropdown(
        id='slct_magnitude', 
        options=[{'label' : x, 'value' : x} for x in magnitude_options],
        multi=False,
        value=5,
        style={'width':'200px', 'float':'left'}
    )], style={'height':'24px'}),
    
    html.Table([html.Div(id='out_mag', children=[]), 
                html.Div(id='out_mag_types'), 
                html.Div('Source of data: Canada.ca, EQs in period: 01 jan 1985 to 31 dec 2019'), 
                html.Div(children=['Author: Elsa Burren, elsa@promasta.com, ', html.A('www.promasta.com', href='https://www.promasta.com', target='_blank'), '. Feel free to contact me!'], style={'margin-top':'10px'})
               ], style={'width':'100vw', 'margin-top':'5px'}),
    
    html.P([
        html.Hr(style={'width':'100%', 'margin-top':'1vh', 'float':'left'}),
        html.H2('Map and Frequency', style={'text-align': 'left', 'width':'100%'}),
    
        html.Div([
            dcc.Graph(id='fig_map', figure={}, style={'float':'left', 'padding':'1vw', 'min-width':'650px'}),
            dcc.Graph(id='fig_frq', figure={}, style={'float':'left', 'padding':'1vw', 'min-width':'650px'})
        ], style={'width':'100%', 'float':'left'}),
    ], style={'max-width':'1500px'}),
    
    html.P([
        html.Br(),
        html.Hr(style={'width':'100%', 'margin-top':'1vh', 'float':'left'}),
        html.H2('Fitting Distributions', style={'text-align': 'left', 'width':'100%'}),
    
        html.Div([
            html.Div([
                html.P([
                    html.Div('Select Distribution for EQ Magnitudes: ', style={'float':'left', 'padding': '5px 5px 5px 3px', 'border': '1px solid rgb(200,200,200)', 'width':'350px', 'height':'24px'}),
                    dcc.Dropdown(id='slct_sev_dist', 
                                 options=[{'label' : 'Pareto',     'value' : 'pareto'},
                                          {'label' : 'Gamma',      'value' : 'gamma'},
                                          {'label' : 'Lognormal',  'value' : 'lognorm'}],
                                multi=False,
                                value='pareto',
                                style={'float':'left', 'min-width':'200px'})
                ], style={'height':'24px'}),
                dcc.Graph(id='fig_sev_dist', figure={}),
                html.P(id='out_sev_dist', children=[])
            ], style={'float':'left', 'padding':'1vw', 'min-width':'650px'}),
            html.Div([
                html.P([
                    html.Div('Select Distribution for Annual Nb of EQ: ', style={'float':'left', 'padding': '5px 5px 5px 3px', 'border': '1px solid rgb(200,200,200)', 'width':'350px', 'height':'24px'}),
                    dcc.Dropdown(
                        id='slct_frq_dist', 
                        options=[{'label' : 'Poisson',            'value' : 'poisson'},
                                 {'label' : 'Flexible',           'value' : 'flexible'}],
                        multi=False,
                        value='poisson',
                        style={'float':'left', 'min-width':'200px'})
                ], style={'height':'24px'}),
                dcc.Graph(id='fig_frq_dist', figure={}),
                html.P(id='out_frq_dist', children=[])
            ], style={'float':'left', 'padding':'1vw', 'min-width':'650px'})
        ], style={'width':'100%', 'float':'left'})
    ], style={'max-width':'1500px'})
])

# Create and connect Plotly graphs with Dash Components

# Magnitude selection
@app.callback(
        [Output(component_id='out_mag', component_property='children'),
         Output(component_id='out_mag_types', component_property='children'),         
         Output(component_id='fig_map', component_property='figure'),
         Output(component_id='fig_frq', component_property='figure'),
         Output(component_id='slct_frq_dist', component_property='value')],
        [Input(component_id='slct_magnitude', component_property='value')]
)
def update_graphs(slct_magnitude):
    # Filter data
    dff = df[df['magnitude'] >= slct_magnitude]

    # Create location map
    fig_map = px.scatter_geo(dff,
        lat = 'latitude',
        lon = 'longitude',
        hover_name = dff['place'] + ': ' + dff['magnitude'].astype(str) + dff['magnitude type'] + ' in year ' + dff['year'].astype(str),
        color = 'magnitude',
        color_continuous_scale='YlOrRd',
        range_color=[magnitude_min,magnitude_max],
        opacity = (dff['magnitude']-magnitude_min+.1)/(magnitude_max-magnitude_min+.1),
        title='Location of EQ with Magnitude >= {}'.format(slct_magnitude)
       # size = 'magnitude'
    )
    
    # Create frequency bar chart
    fig_freq = px.histogram(dff['year'].astype(int), x='year', range_x=[year_min, year_max], nbins=year_max-year_min+1, labels=dict(x=str_years), title='Nb of EQ with magnitude >= {}'.format(slct_magnitude))
    fig_freq.update_layout(bargap=0.1)
    
    # Create text output and return
    out_mag = 'All eq with magnitude >= {} will be considered, all magnitude types together.'.format(slct_magnitude)
    out_mag_types = 'The distribution of the magnitude types is: ' + ' '.join((dff['magnitude type'].value_counts(normalize=True).to_frame().astype(float).apply(lambda row : round(row*100.0,2), axis=1).
                                                                                               astype(str)+'%').to_string(index=True,header=False).replace('\n',', ').split())
    return out_mag, out_mag_types, fig_map, fig_freq, ''

# Severity fitting
@app.callback(
    [Output(component_id='fig_sev_dist', component_property='figure'),
     Output(component_id='out_sev_dist', component_property='children')],
    [Input(component_id='slct_magnitude', component_property='value'),
     Input(component_id='slct_sev_dist', component_property='value')]
)
def update_sev_dist(slct_magnitude, slct_dist):
    # Filter data
    dff = df.loc[df['magnitude'] >= slct_magnitude, ['magnitude']]
    
    # Fit
    if slct_dist == 'pareto':
        param = pareto.fit(dff['magnitude'], method='MLE', floc=0.0, fscale=slct_magnitude)
    if slct_dist == 'gamma':
        param = gamma.fit(dff['magnitude'], method='MLE')
    if slct_dist == 'lognorm':
        param = lognorm.fit(dff['magnitude'], method='MLE', fscale=1.0)
    fitted_dist = Severity(shape=param[0], loc=param[1], scale=param[2], type=slct_dist)
    format_dist_name = {'pareto':'Pareto', 'gamma':'Gamma', 'lognorm':'Lognormal'}
    
    # Compute survival probability
    dff = create_survival(dff, 'magnitude', fitted_dist)
    
    # Create graph output
    fig_dist = px.scatter(dff, x='magnitude', y='sf', color='sf_type', title='Observed vs {} Survival Prob for Magnitudes >= {}'.format(format_dist_name[fitted_dist.type()], slct_magnitude))
    fig_dist.update_layout(legend=dict(title='Distribution'), yaxis=dict(title='survival probability (1-CDF)'))

    # Create text output and return
    out_dist = 'Fitted {0} parameters are: shape={1:.4f}, scale={2:.4f}, location={3:.4f}'.format(format_dist_name[fitted_dist.type()], fitted_dist.shape(), fitted_dist.scale(), fitted_dist.loc())
    
    return fig_dist, out_dist

# Frequency fitting
@app.callback(
    [Output(component_id='fig_frq_dist', component_property='figure'),
     Output(component_id='out_frq_dist', component_property='children')],
    [Input(component_id='slct_magnitude', component_property='value'),
     Input(component_id='slct_frq_dist', component_property='value')]
)
def update_frq_dist(slct_magnitude, slct_dist):
    # Filter data
    dff = df.loc[df['magnitude'] >= slct_magnitude
                 , ['year', 'count']]
    
    # Compute annual nb of EQ
    dff = pd.concat([dff, format_frq])
    dff = dff.groupby(['year'], as_index=False)[['count']].apply(lambda x: x.sum())
    
    # Fit
    fitted_dist = Frequency(mean=dff['count'].mean()) if slct_dist =='poisson' else Frequency(mean=dff['count'].mean(), variance=dff['count'].var())
    format_dist_name = {'poisson':'Poisson', 'binom':'Binomial', 'nbinom':'Negative Binomial'}
    
    # Compute survival probabilities
    dff = create_survival(dff, 'count', fitted_dist)

    # Create graph output
    fig_dist = px.scatter(dff, x='count', y='sf', color='sf_type', title='Observed vs {} Survival Prob for Nb of EQs'.format(format_dist_name[fitted_dist.type()]))
    fig_dist.update_layout(legend=dict(title='Distribution'), yaxis=dict(title='survival probability (1-CDF)'), xaxis=dict(title='number EQs with magnitude >= {}'.format(slct_magnitude)))
    
    # Create output and return
    out_dist = 'Fitted {0} parameters are: mean={1:.4f}, standard deviation={2:.4f}'.format(format_dist_name[fitted_dist.type()], fitted_dist.mean(), fitted_dist.sd())
    
    return fig_dist, out_dist

# Run the app
if __name__=='__main__':
    app.run_server(debug=True)
