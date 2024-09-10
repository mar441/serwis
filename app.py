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

def load_anomaly_data(file_path, file_label):
    df = pd.read_csv(file_path)
    df['file'] = file_label
    return df

geo_data_1 = pd.read_csv('mos_1.csv')
geo_data_2 = pd.read_csv('mos_2.csv')
geo_data_3 = pd.read_csv('msz_1.csv')
geo_data_4 = pd.read_csv('msz_2.csv')
geo_data = pd.concat([geo_data_1, geo_data_2, geo_data_3, geo_data_4], ignore_index=True)

displacement_data_1 = load_displacement_data('mz2_10.csv', 'Descending 175')
displacement_data_2 = load_displacement_data('mz4_3.csv', 'Ascending 124')
displacement_data_3 = load_displacement_data('msz4_3.csv', 'Descending 175')
displacement_data_4 = load_displacement_data('msz2_3.csv', 'Ascending 124')

all_data_1 = pd.merge(displacement_data_1, geo_data, on='pid', how='left')
all_data_2 = pd.merge(displacement_data_2, geo_data, on='pid', how='left')
all_data_3 = pd.merge(displacement_data_3, geo_data, on='pid', how='left')
all_data_4 = pd.merge(displacement_data_4, geo_data, on='pid', how='left')

all_data = pd.concat([all_data_1, all_data_2, all_data_3, all_data_4], ignore_index=True)


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

prediction_data_3 = pd.read_csv('predictions_values3.csv') 
prediction_data_3 = prediction_data_3.melt(var_name='pid', 
                                           value_name='predicted_displacement')
prediction_data_3['label'] = 'Prediction Set 3'
prediction_data_3['step'] = prediction_data_3.groupby('pid').cumcount()

prediction_data_4 = pd.read_csv('predictions_values4.csv') 
prediction_data_4 = prediction_data_4.melt(var_name='pid', 
                                           value_name='predicted_displacement')
prediction_data_4['label'] = 'Prediction Set 4'
prediction_data_4['step'] = prediction_data_4.groupby('pid').cumcount()

all_prediction_data = pd.concat([prediction_data_1, prediction_data_2, prediction_data_3, prediction_data_4], ignore_index=True)

anomaly_data_1 = load_anomaly_data('anomaly1.csv', 'Anomaly Set 1')
anomaly_data_2 = load_anomaly_data('anomaly2.csv', 'Anomaly Set 2')
anomaly_data_3 = load_anomaly_data('anomaly3.csv', 'Anomaly Set 3')
anomaly_data_4 = load_anomaly_data('anomaly4.csv', 'Anomaly Set 4')

all_anomaly_data = pd.concat([anomaly_data_1, anomaly_data_2, anomaly_data_3, anomaly_data_4], ignore_index=True)

all_data.sort_values(by=['pid', 'timestamp'], inplace=True)
all_data['displacement_diff'] = all_data.groupby('pid')['displacement'].diff()
all_data['time_diff'] = all_data.groupby('pid')['timestamp'].diff().dt.days
all_data['displacement_speed'] = (all_data['displacement_diff'] / all_data['time_diff']) * 365

mean_velocity_data = all_data.groupby('pid')['displacement_speed'].mean().reset_index()
mean_velocity_data.rename(columns={'displacement_speed': 'mean_velocity'}, inplace=True)
all_data = pd.merge(all_data, mean_velocity_data, on='pid', how='left')

px.set_mapbox_access_token('pk.eyJ1IjoibWFycGllayIsImEiOiJjbTBxbXBsMGQwYjgyMmxzN3RpdmlhZDVrIn0.YWJh1RM6HKfN_pbH-jtJ6A')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3("Select Map and Data Visualization Options"),  

    html.Div([ 
        html.Div([
            html.Label("Map Style"),
            dcc.Dropdown(
                id='map-style-dropdown',
                options=[
                    {'label': 'Satellite', 'value': 'satellite'},
                    {'label': 'Outdoors', 'value': 'outdoors'},
                    {'label': 'Light', 'value': 'light'},
                    {'label': 'Dark', 'value': 'dark'},
                    {'label': 'Streets', 'value': 'streets'},
                ],
                value='satellite',
                clearable=False,
                style={'width': '90%'}
            )
        ], style={'display': 'inline-block', 'width': '45%', 'padding': '10px'}),

        html.Div([
            html.Label("Visualization Option"),
            dcc.Dropdown(
                id='color-mode-dropdown',
                options=[
                    {'label': 'Orbit Type', 'value': 'orbit'},
                    {'label': 'Displacement Mean Velocity [mm/year]', 'value': 'speed'},
                    {'label': 'Anomaly Type', 'value': 'anomaly_type'},
                ],
                value='orbit',
                clearable=False,
                style={'width': '90%'}
            )
        ], style={'display': 'inline-block', 'width': '45%', 'padding': '10px'})  
    ]),

    dcc.Graph(id='map', style={'height': '80vh', 'width': '95vw'}, config={'scrollZoom': True }),

    html.Div(id='displacement-container', 
             children=[dcc.Graph(id='displacement-graph', style={'height': '50vh', 'width': '95vw'})], 
             style={'display': 'none'})
])

@app.callback(
    Output('map', 'figure'),
    [Input('map-style-dropdown', 'value'),
     Input('color-mode-dropdown', 'value')]
)
def update_map(map_style, color_mode):
    data = all_data.drop_duplicates(subset=['pid'])

    if color_mode == 'orbit':
        fig = px.scatter_mapbox(data,
                                lat='latitude',
                                lon='longitude',
                                hover_name='pid',
                                hover_data={'height': True, 'mean_velocity': True},
                                color='file',
                                zoom=18,
                                height=800)

        fig.update_layout(
            legend_title_text='Orbit type', 
            legend=dict(yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99)
        )

    elif color_mode == 'speed':
        fig = px.scatter_mapbox(data,
                                lat='latitude',
                                lon='longitude',
                                hover_name='pid',
                                hover_data={'height': True, 'mean_velocity': True}, 
                                color='mean_velocity',
                                color_continuous_scale='Jet', 
                                range_color=(5, -5),
                                zoom=18,
                                height=800)

        fig.update_layout(coloraxis=dict(
            colorscale='Jet',  
            cmin=-5,  
            cmax=5,   
            colorbar=dict(
                title="Mean Velocity (mm/year)",
                tickvals=[5, 2.5, 0, -2.5, -5], 
                ticktext=['-5', '-2.5', '0', '2.5', '5'],
                tickmode='array' 
            )
        ))

    elif color_mode == 'anomaly_type':  
        merged_data = data.merge(all_anomaly_data[['pid', 'is_anomaly']], on='pid', how='left')
        merged_data['anomaly_status'] = merged_data['is_anomaly'].fillna(False).astype(bool)

        fig = px.scatter_mapbox(merged_data,
                                lat='latitude',
                                lon='longitude',
                                hover_name='pid',
                                hover_data={'height': True, 'mean_velocity': True},
                                color=merged_data['anomaly_status'].map({True: 'Anomaly', False: 'No Anomaly'}),
                                color_discrete_map={'Anomaly': 'red', 'No Anomaly': 'green'},
                                zoom=18,
                                height=800)

    min_lat = data['latitude'].min()
    max_lat = data['latitude'].max()
    min_lon = data['longitude'].min()
    max_lon = data['longitude'].max()

    fig.update_layout(mapbox_style=map_style,  
                      mapbox_center={"lat": (min_lat + max_lat) / 2, "lon": (min_lon + max_lon) / 2},
                      mapbox_zoom=14)

    fig.update_layout(mapbox_bounds={"west": min_lon - 2, "east": max_lon + 2, "south": min_lat - 2, 
                                     "north": max_lat + 2})
    
    fig.update_layout(legend=dict(yanchor="top",
                                  y=0.99,
                                  xanchor="right",
                                  x=0.99))

    return fig

@app.callback(
    [Output('displacement-graph', 'figure'), Output('displacement-container', 'style')],
    Input('map', 'clickData')
)
def display_displacement(clickData):
    if clickData is None:
        return {}, {'display': 'none'}

    point_id = clickData['points'][0]['hovertext']  
    filtered_data = all_data[all_data['pid'] == point_id].copy()
    filtered_anomalies = all_anomaly_data[all_anomaly_data['pid'] == point_id].copy()

    if filtered_anomalies.empty:
        return {}, {'display': 'none'}
    
    if len(filtered_data) >= 60:
        filtered_anomalies = filtered_anomalies.tail(60)
        filtered_anomalies['timestamp'] = filtered_data['timestamp'].tail(60).values
    
    filtered_anomalies.set_index('timestamp', inplace=True)
    filtered_data.set_index('timestamp', inplace=True)

    filtered_data = filtered_data.join(filtered_anomalies[['predicted_value', 'upper_bound', 'lower_bound', 'is_anomaly']], 
                                       how='left')

    fig = px.line(filtered_data.reset_index(), x='timestamp', y='displacement', 
                  title=f"Displacement for point {point_id}",
                  markers=True, 
                  labels={'displacement': 'Displacement (mm)'})

    fig.add_scatter(x=filtered_data.index, y=filtered_data['displacement'], 
                    mode='lines+markers', 
                    name='Actual Displacement', 
                    line=dict(color='blue'))

    fig.add_scatter(x=filtered_data.index, y=filtered_data['predicted_value'], 
                    mode='lines+markers', 
                    name='Predicted Displacement', 
                    line=dict(color='orange'))

    fig.add_scatter(x=filtered_data.index, 
                    y=filtered_data['upper_bound'], 
                    mode='lines', 
                    line=dict(color='gray', dash='dash'),
                    name='Upper Bound')

    fig.add_scatter(x=filtered_data.index, 
                    y=filtered_data['lower_bound'],
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.2)',
                    name='Lower Bound')

    anomalies = filtered_data[filtered_data['is_anomaly'] == 1]
    fig.add_scatter(x=anomalies.index, y=anomalies['displacement'], 
                    mode='markers', 
                    name='Anomaly', 
                    marker=dict(color='red', size=10))

    fig.update_layout(xaxis_title='Date', yaxis_title='Displacement (mm)', legend_title="Data Type")
  
    fig.update_layout(legend=dict(yanchor="top",
                                  y=1,
                                  xanchor="left",
                                  x=1.05))

    return fig, {'display': 'block'}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
