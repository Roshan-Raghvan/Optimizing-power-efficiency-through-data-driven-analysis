import pandas as pd
import plotly.graph_objects as go
import numpy as np

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import base64
import io

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = 'CyberCorps.AI'

# Define the dropdown options
party_data = [
    {'label': 'Time-Series Plot', 'value': 'Time-Series Plot'},
    {'label': 'Appliance-wise Consumption', 'value': 'Appliance-wise Consumption'},
    {'label': 'Electricity Consumption Forecast', 'value': 'Electricity Consumption Forecast'},
    {'label': 'Faulty Devices', 'value': 'Faulty Devices'}
]

# Define the app layout
app.layout = html.Div([
    html.H2('CyberCrops.AI Dashboard', style={'textAlign': 'center', 'font-family': 'Trebuchet MS', 'color': '#90E219'}),

    # Upload Components
    html.Div([
        html.Div([
            html.Label('Upload Electricity Data CSV'),
            dcc.Upload(
                id='upload-electrics',
                children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px 0'
                },
                multiple=False
            ),
        ], className='six columns'),

        html.Div([
            html.Label('Upload Electrical Appliance Consumption CSV'),
            dcc.Upload(
                id='upload-appliance-consumption',
                children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px 0'
                },
                multiple=False
            ),
        ], className='six columns'),
    ], className='row'),

    html.Div([
        html.Div([
            html.Label('Upload Electrical Forecast CSV'),
            dcc.Upload(
                id='upload-forecast',
                children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px 0'
                },
                multiple=False
            ),
        ], className='six columns'),

        html.Div([
            html.Label('Upload Electricity Appliance Wise Data CSV'),
            dcc.Upload(
                id='upload-appliance-wise',
                children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px 0'
                },
                multiple=False
            ),
        ], className='six columns'),
    ], className='row'),

    # Dropdown for plot selection
    html.Div([
        html.Label('Select Plot Type', style={'color': '#90E219', 'font-weight': 'bold'}),
        dcc.Dropdown(
            id='stockselector',
            options=party_data,
            value='Time-Series Plot',
            style={'backgroundColor': '#1E1E1E', 'color': '#90E219'}
        )
    ], style={'width': '50%', 'margin': '20px auto'}),

    # Graph Component
    html.Div([
        dcc.Graph(id='timeseries', config={'displayModeBar': False}),
    ], style={'width': '90%', 'margin': '0 auto'}),
], style={'backgroundColor': '#121212', 'padding': '20px'})

def parse_contents(contents, filename):
    """
    Parses the uploaded file contents into a Pandas DataFrame.
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename.lower():
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df
        else:
            return None
    except Exception as e:
        print(f"Error parsing file {filename}: {e}")
        return None

@app.callback(
    Output('timeseries', 'figure'),
    [
        Input('stockselector', 'value'),
        Input('upload-electrics', 'contents'),
        Input('upload-electrics', 'filename'),
        Input('upload-appliance-consumption', 'contents'),
        Input('upload-appliance-consumption', 'filename'),
        Input('upload-forecast', 'contents'),
        Input('upload-forecast', 'filename'),
        Input('upload-appliance-wise', 'contents'),
        Input('upload-appliance-wise', 'filename'),
    ]
)
def update_graph(
    selected_plot,
    contents_electrics, filename_electrics,
    contents_appliance_consumption, filename_appliance_consumption,
    contents_forecast, filename_forecast,
    contents_appliance_wise, filename_appliance_wise
):
    """
    Updates the graph based on the selected plot type and uploaded CSV files.
    """
    # Parse uploaded files
    df = parse_contents(contents_electrics, filename_electrics) if contents_electrics else None
    df2 = parse_contents(contents_appliance_consumption, filename_appliance_consumption) if contents_appliance_consumption else None
    df3 = parse_contents(contents_forecast, filename_forecast) if contents_forecast else None
    df4 = parse_contents(contents_appliance_wise, filename_appliance_wise) if contents_appliance_wise else None

    # Validate that required DataFrames are available based on the selected plot
    if selected_plot == 'Time-Series Plot' and df is not None:
        # Process df for Time-Series Plot
        df = df[1110:].reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Total_Consumption'],
            mode='lines',
            name='Total Consumption',
            line=dict(color="#19E2C5")
        ))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            margin={'b': 15},
            hovermode='x',
            autosize=True,
            yaxis_title="Consumption (kWh)",
            xaxis_title="Date",
            title={'text': 'Time-Series Plot of Electricity Consumption for this year', 'font': {'color': 'white'}, 'x': 0.5},
            xaxis={'range': [df.index.min(), df.index.max()]},
        )
        return fig

    elif selected_plot == 'Electricity Consumption Forecast' and df3 is not None:
        # Process df3 for Forecast Plot
        df_sub = df3.copy()
        df_sub['Date'] = pd.to_datetime(df_sub['Date'])
        df_sub.set_index('Date', inplace=True)

        # Identify anomalies where MAE >= 15
        df_anoms = df_sub[df_sub['MAE'] >= 15]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sub.index,
            y=df_sub['Total_Consumption'],
            mode='lines',
            name='Actual Consumption',
            line=dict(color="#19E2C5")
        ))
        fig.add_trace(go.Scatter(
            x=df_sub.index,
            y=df_sub['Predicted_Consumption'],
            mode='lines',
            name='Predicted Consumption',
            line=dict(color="#C6810B")
        ))
        fig.add_trace(go.Scatter(
            x=df_anoms.index,
            y=df_anoms['Total_Consumption'],
            mode='markers',
            name='Excess Consumption',
            marker=dict(size=10, color='#C60B0B')
        ))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            margin={'b': 15},
            autosize=True,
            yaxis_title="Consumption (kWh)",
            xaxis_title="Date",
            title={'text': 'Time-Series Plot & Forecasting Electricity Consumption for this year', 'font': {'color': 'white'}, 'x': 0.5}
        )
        return fig

    elif selected_plot == 'Faulty Devices' and df4 is not None:
        # Process df4 for Faulty Devices Plot
        df_sub = df4.copy()
        df_sub['Date'] = pd.to_datetime(df_sub['Date'])
        df_sub = df_sub[df_sub['Date'].dt.year == 2021].reset_index(drop=True)

        def zscore(x, window):
            r = x.rolling(window=window)
            m = r.mean().shift(1)
            s = r.std(ddof=0).shift(1)
            z = (x - m) / s
            return z

        # Calculate Z-scores for each appliance
        df_sub['kap_zscore'] = zscore(df_sub['Kitchen Appliances'], 30)
        df_sub['fridge_zscore'] = zscore(df_sub['Fridge'], 30)
        df_sub['ac_zscore'] = zscore(df_sub['AC'], 30)
        df_sub['oap_zscore'] = zscore(df_sub['Other Appliances'], 30)
        df_sub['wm_zscore'] = zscore(df_sub['Washing Machine'], 3)

        # Handle infinite and NaN values
        df_sub.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_sub.fillna(0, inplace=True)

        # Identify anomalies where Z-score > 5
        anomalies = {
            'Fridge': df_sub[df_sub['fridge_zscore'] > 5],
            'AC': df_sub[df_sub['ac_zscore'] > 5],
            'Other Appliances': df_sub[df_sub['oap_zscore'] > 5],
            'Kitchen Appliances': df_sub[df_sub['kap_zscore'] > 5],
            'Washing Machine': df_sub[df_sub['wm_zscore'] > 5]
        }

        fig_anom = go.Figure()

        # Add traces for each appliance
        for appliance, anomaly_df in anomalies.items():
            fig_anom.add_trace(go.Scatter(
                x=df_sub['Date'],
                y=df_sub[appliance],
                mode='lines',
                name=f'{appliance} Consumption',
                line=dict(color="#19E2C5")
            ))
            fig_anom.add_trace(go.Scatter(
                x=anomaly_df['Date'],
                y=anomaly_df[appliance],
                mode='markers',
                name=f'Anomalies in {appliance}',
                marker=dict(size=10, color='#C60B0B')
            ))

        # Update layout with interactive buttons for each appliance
        updatemenus = [
            dict(
                active=0,
                buttons=[
                    dict(label=appliance,
                         method="update",
                         args=[{
                             "visible": [
                                 i % 2 == 0 and (i//2 == idx) for i in range(len(fig_anom.data))
                             ]
                         },
                             {"title": f"Anomalies in power consumption of {appliance}"}]
                         )
                    for idx, appliance in enumerate(anomalies.keys())
                ],
                direction="down",
                showactive=True,
                x=0.1,
                y=1.15,
                xanchor='left',
                yanchor='top'
            )
        ]

        fig_anom.update_layout(
            updatemenus=updatemenus,
            template='plotly_dark',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            title_font_color="#90E219",
            font_color="#90E219",
            autosize=True,
            margin=dict(t=100),
            title={'text': 'Anomalies in Power Consumption', 'font': {'color': '#90E219'}, 'x': 0.5},
        )
        return fig_anom

    elif selected_plot == 'Appliance-wise Consumption' and df2 is not None:
        # Process df2 for Appliance-wise Consumption Plot
        df_sub = df2.copy()

        # Extract consumption data for each appliance by month
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
        appliances = ['Fridge', 'Kitchen Appliances', 'AC', 'Washing Machine', 'Other Appliances']
        irises_colors = ['rgb(33, 75, 99)', 'rgb(79, 129, 102)', 'rgb(151, 179, 100)',
                         'rgb(175, 49, 35)', 'rgb(36, 73, 147)']

        # Create pie charts for each month
        pies = []
        for month in months:
            values = [
                df_sub[df_sub['month'] == month]['Fridge'].values[0],
                df_sub[df_sub['month'] == month]['Kitchen Appliances'].values[0],
                df_sub[df_sub['month'] == month]['AC'].values[0],
                df_sub[df_sub['month'] == month]['Washing Machine'].values[0],
                df_sub[df_sub['month'] == month]['Other Appliances'].values[0]
            ]
            pies.append(go.Pie(
                labels=appliances,
                values=values,
                marker_colors=irises_colors,
                hole=0.3,
                name=f'{month} 2021'
            ))

        fig = go.Figure(data=pies)

        # Update layout with interactive buttons for each month
        updatemenus = [
            dict(
                active=0,
                buttons=[
                    dict(label=month,
                         method="update",
                         args=[{"visible": [m == month for m in months]},
                               {"title": f"{month} Consumption Distribution (%) by each Appliance"}])
                    for month in months
                ],
                direction="down",
                showactive=True,
                x=0.1,
                y=1.15,
                xanchor='left',
                yanchor='top'
            )
        ]

        # Initially, show only the first month's pie
        visibility = [m == months[0] for m in months]

        fig.update_layout(
            updatemenus=updatemenus,
            template='plotly_dark',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            title={'text': f"{months[0]} Consumption Distribution (%) by each Appliance", 'font': {'color': '#90E219'}, 'x': 0.5},
            width=700,
            height=700,
            font_color="#90E219",
            title_font_color="#90E219"
        )

        # Set initial visibility
        for i, pie in enumerate(fig.data):
            fig.data[i].visible = visibility[i]

        return fig

    else:
        # If no plot is selected or required data is missing
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            margin={'b': 15},
            hovermode='x',
            autosize=True,
            yaxis_title="Consumption (kWh)",
            xaxis_title="Date",
            title={'text': 'Please upload the required CSV files and select a plot type.', 'font': {'color': 'white'}, 'x': 0.5},
        )
        return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True,port=8080)