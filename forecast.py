import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import base64
import io
import statsmodels.api as sm
from flask import Flask
# Initialize the Dash app
app = dash.Dash(__name__)
app.title = 'CyberCorps.AI - Forecast Generator'

# Define the layout of the app
app.layout = html.Div([
    html.H1('CyberCorps.AI - Electrical Forecast Generator', style={'textAlign': 'center', 'font-family': 'Trebuchet MS', 'color': '#90E219'}),

    html.Div([
        html.H3('Upload Electrical Data CSV', style={'color': '#90E219'}),
        dcc.Upload(
            id='upload-electrical-data',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px 0',
                'backgroundColor': '#f0f0f0'
            },
            multiple=False
        ),
        html.Button('Generate Forecast', id='generate-forecast', n_clicks=0, style={'margin-top': '10px', 'backgroundColor': '#90E219', 'color': 'white'}),
        html.Div(id='forecast-output', style={'margin-top': '20px', 'color': '#90E219'})
    ], style={'margin': '20px 0', 'padding': '20px', 'backgroundColor': '#2B2B2B', 'borderRadius': '10px'}),
], style={'padding': '50px', 'backgroundColor': '#121212'})

# Callback to generate forecast and save the CSV
@app.callback(
    Output('forecast-output', 'children'),
    [Input('generate-forecast', 'n_clicks')],
    [State('upload-electrical-data', 'contents'),
     State('upload-electrical-data', 'filename')]
)
def generate_forecast(n_clicks, contents, filename):
    if n_clicks > 0 and contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[df['Date'].dt.year == 2021]
            df.reset_index(drop=True, inplace=True)

            actual_vals = df.Total_Consumption.values
            train, test = actual_vals[0:-80], actual_vals[-80:]
            train_log, test_log = np.log10(train), np.log10(test)
            history = [x for x in train_log]
            predictions = []

            for t in range(len(test_log)):
                model = sm.tsa.SARIMAX(history, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                                       enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = 10**output[0]
                predictions.append(yhat)
                obs = test_log[t]
                history.append(obs)

            df_preds = df[-80:]
            df_preds['Predicted_Consumption'] = predictions

            # Add MAE Calculation
            new_dates = pd.date_range(start='2021/07/31', end='2021/09/30')
            train = df.Total_Consumption.values

            model = sm.tsa.SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 62),
                                   enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit()
            output = model_fit.forecast(62)
            listofzeros = [np.NaN] * 62
            data_new = {'Date': new_dates,
                        'Total_Consumption': listofzeros,
                        'Predicted_Consumption': output}

            df_new_preds = pd.DataFrame(data_new)

            df_all = pd.concat([df_preds, df_new_preds], ignore_index=True)
            df_all['MAE'] = df_all['Total_Consumption'] - df_all['Predicted_Consumption']
            df_anoms = df_all[df_all['MAE'] >= 15]
            df_anoms.reset_index(drop=True, inplace=True)

            # Save the forecast to a CSV file
            output_csv = 'Sarimax_Predicted_outcome.csv'
            df_anoms.to_csv(output_csv, index=False)

            return html.Div([
                html.H5(f'Forecast generated successfully!'),
                html.A('Download Forecast CSV', href=f'/download/{output_csv}', download=output_csv, style={'color': '#90E219'})
            ])

        except Exception as e:
            return html.Div([f'Error processing file: {str(e)}'], style={'color': 'red'})
    return html.Div()

# Serve the CSV file for download
@app.server.route('/download/<path:filename>')
def download_file(filename):
    return Flask.send_from_directory(directory='.', filename=filename, as_attachment=True)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
