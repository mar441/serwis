import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
from scipy.stats import t
import numpy as np
import os

def load_displacement_data(file_path, file_label):
    df = pd.read_csv(file_path)
    df = df.melt(id_vars=['Date'], 
                 var_name='pid', 
                 value_name='displacement')
    df['timestamp'] = pd.to_datetime(df['Date'])
    df.drop(columns=['Date'], inplace=True)
    df['file'] = file_label
    return df

geo_data_1 = pd.read_csv('mos_1.csv')
geo_data_2 = pd.read_csv('mos_2.csv')
geo_data = pd.concat([geo_data_1, geo_data_2], ignore_index=True)


displacement_data_1 = load_displacement_data('mz2_10.csv', 'Descending 175')
displacement_data_2 = load_displacement_data('mz4_3.csv', 'Ascending 124')

all_data_1 = pd.merge(displacement_data_1, geo_data, on='pid', how='left')
all_data_2 = pd.merge(displacement_data_2, geo_data, on='pid', how='left')

all_data = pd.concat([all_data_1, all_data_2], ignore_index=True)


prediction_data_1 = pd.read_csv('predictions_values.csv')
prediction_data_1 = prediction_data_1.melt(var_name='pid', 
                                           value_name='predicted_displacement')
prediction_data_1['label'] = 'Prediction Set 1'
prediction_data_1['step'] = prediction_data_1.groupby('pid').cumcount()

prediction_data_2 = pd.read_csv('predictions_values2.csv') 
prediction_data_2 = prediction_data_2.melt(var_name='pid', 
                                           value_name='predicted_displacement')
prediction_data_2['label'] = 'Prediction Set 2'
prediction_data_2['step'] = prediction_data_2.groupby('pid').cumcount()

all_prediction_data = pd.concat([prediction_data_1, prediction_data_2], ignore_index=True)


app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='map'),
    html.Div(id='displacement-container', children=[
        dcc.Graph(id='displacement-graph')
    ], style={'display': 'none'}) 
])

@app.callback(
    Output('map', 'figure'),
    Input('map', 'id')
)
def update_map(_):
    fig = px.scatter_mapbox(
        all_data.drop_duplicates(subset=['pid']), 
        lat='latitude', 
        lon='longitude', 
        hover_name='pid', 
        hover_data={'height': True},  
        color='file',
        zoom=5,  
        height=600  
    )
    
    min_lat = all_data['latitude'].min()
    max_lat = all_data['latitude'].max()
    min_lon = all_data['longitude'].min()
    max_lon = all_data['longitude'].max()

    fig.update_layout(mapbox_style="open-street-map", 
                      mapbox_center={"lat": (min_lat + max_lat) / 2, "lon": (min_lon + max_lon) / 2},
                      mapbox_zoom=5)

    fig.update_layout(mapbox_bounds={"west": min_lon - 0.0005, "east": max_lon + 0.0005, "south": min_lat - 0.0005, 
                                     "north": max_lat + 0.0005})
    
    fig.update_layout(legend_title_text='Relative orbit:') 
    return fig

@app.callback(
    [Output('displacement-graph', 'figure'),
     Output('displacement-container', 'style')],
    Input('map', 'clickData')
)
def display_displacement(clickData):
    if clickData is None:
        return {}, {'display': 'none'}
    
    point_id = clickData['points'][0]['hovertext']  
    filtered_data = all_data[all_data['pid'] == point_id].copy()
    filtered_predictions = all_prediction_data[all_prediction_data['pid'] == point_id].copy()

    if len(filtered_data) >= 60:
        prediction_timestamps = filtered_data['timestamp'].iloc[-60:].values
        filtered_predictions = filtered_predictions.iloc[-len(prediction_timestamps):].copy()
        filtered_predictions.loc[:, 'timestamp'] = prediction_timestamps

    filtered_data.set_index('timestamp', inplace=True)
    filtered_predictions.set_index('timestamp', inplace=True)

    filtered_data['residuals'] = filtered_data['displacement'] - filtered_predictions['predicted_displacement']
    
    degrees_of_freedom = 60
    residuals_std = filtered_data['residuals'].std()

    margin_of_error = t.ppf(0.95, degrees_of_freedom) * residuals_std
    filtered_predictions['yhat_lower'] = filtered_predictions['predicted_displacement'] - margin_of_error
    filtered_predictions['yhat_upper'] = filtered_predictions['predicted_displacement'] + margin_of_error

    filtered_data = filtered_data.join(filtered_predictions[['yhat_lower', 'yhat_upper']], how='left')

    filtered_data['anomaly'] = (filtered_data['displacement'] < filtered_data['yhat_lower']) | (filtered_data['displacement'] > filtered_data['yhat_upper'])

    fig = px.line(filtered_data.reset_index(), x='timestamp', y='displacement', 
                  title=f"Displacement for point {point_id}",
                  markers=True, 
                  labels={'displacement': 'Displacement (mm)'})

    fig.add_scatter(x=filtered_data.index, y=filtered_data['displacement'], 
                    mode='lines+markers', 
                    name='Actual Displacement', 
                    line=dict(color='blue'))

    fig.add_scatter(x=filtered_predictions.index, y=filtered_predictions['predicted_displacement'], 
                    mode='lines+markers', 
                    name='Predicted Displacement', 
                    line=dict(color='orange'))
    
    fig.add_scatter(x=filtered_predictions.index, 
                    y=filtered_predictions['yhat_upper'], 
                    mode='lines', 
                    line=dict(color='gray', dash='dash'),
                    name='Upper Bound')

    fig.add_scatter(x=filtered_predictions.index, 
                y=filtered_predictions['yhat_lower'], 
                mode='lines', 
                line=dict(color='gray', dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.2)', 
                name='Lower Bound')

    anomalies = filtered_data[filtered_data['anomaly']]
    fig.add_scatter(x=anomalies.index, y=anomalies['displacement'], 
                    mode='markers', 
                    name='Anomaly', 
                    marker=dict(color='red', size=10))

    fig.update_layout(xaxis_title='Date', yaxis_title='Displacement (mm)', legend_title="Data Type")
    
    return fig, {'display': 'block'}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)