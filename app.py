import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
from scipy.stats import t
import numpy as np
import os
from geopy.distance import geodesic

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

anomaly_data_1_95 = load_anomaly_data('anomaly_output_95.csv', 'Anomaly Set 1 (95%)')
anomaly_data_2_95 = load_anomaly_data('anomaly_output2_95.csv', 'Anomaly Set 2 (95%)')
anomaly_data_3_95 = load_anomaly_data('anomaly_output3_95.csv', 'Anomaly Set 3 (95%)')
anomaly_data_4_95 = load_anomaly_data('anomaly_output4_95.csv', 'Anomaly Set 4 (95%)')

all_anomaly_data_95 = pd.concat([anomaly_data_1_95, anomaly_data_2_95, 
                                 anomaly_data_3_95, anomaly_data_4_95], ignore_index=True)

anomaly_data_1_99 = load_anomaly_data('anomaly_output_99.csv', 'Anomaly Set 1 (99%)')
anomaly_data_2_99 = load_anomaly_data('anomaly_output2_99 .csv', 'Anomaly Set 2 (99%)')
anomaly_data_3_99 = load_anomaly_data('anomaly_output3_99.csv', 'Anomaly Set 3 (99%)')
anomaly_data_4_99 = load_anomaly_data('anomaly_output4_99.csv', 'Anomaly Set 4 (99%)')

all_anomaly_data_99 = pd.concat([anomaly_data_1_99, anomaly_data_2_99, 
                                 anomaly_data_3_99, anomaly_data_4_99], ignore_index=True)

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
                    {'label': 'Streets', 'value': 'streets'}
                ],
                value='satellite', 
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '24%', 'padding': '10px'}), 
        
        html.Div([
            html.Label("Visualization Option"),
            dcc.Dropdown(
                id='color-mode-dropdown',
                options=[
                    {'label': 'Orbit Type', 'value': 'orbit'},
                    {'label': 'Displacement Mean Velocity [mm/year]', 'value': 'speed'},
                    {'label': 'Anomaly Type', 'value': 'anomaly_type'}
                ],
                value='orbit',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '24%', 'padding': '10px'}),

        html.Div([
            html.Label("Filter by Orbit Type"),
            dcc.Dropdown(
                id='orbit-filter-dropdown',
                options=[
                    {'label': 'Ascending', 'value': 'Ascending 124'},
                    {'label': 'Descending', 'value': 'Descending 175'}
                ],
                value='Ascending 124',
                multi=True,
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '24%', 'padding': '10px'}),

        html.Div([
            html.Label("Enable Distance Calculation"),
            dcc.Dropdown(
                id='distance-calc-dropdown',
                options=[
                    {'label': 'Yes', 'value': 'yes'},
                    {'label': 'No', 'value': 'no'}
                ],
                value='no',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '24%', 'padding': '10px'})
    ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),
    
    html.Div(id='distance-output', style={'font-size': '16px', 'padding': '10px', 'color': 'black'}),

    dcc.Graph(id='map', style={'height': '80vh', 'width': '95vw'}, config={'scrollZoom': True}),

    dcc.Store(id='selected-points', data={'point_1': None, 'point_2': None}),

    html.Div(id='displacement-container', children=[
        html.Div([
            html.Label("Select Date Range", style={'font-size': '16px'}),
            dcc.DatePickerRange(
                id='date-range-picker',
                start_date=all_data['timestamp'].min(),
                end_date=all_data['timestamp'].max(),
                display_format='YYYY-MM-DD',
                style={'height': '5px', 
                'width': '300px', 
                'font-family': 'Arial', 
                'font-size': '4px', 
                'display': 'inline-block',
                'padding': '5px' }
            )
        ], style={'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("Set Y-Axis Range (mm)"),
            dcc.Input(
                id='y-axis-min',
                type='number',
                placeholder='Min',
                style={'width': '20%', 'margin-right': '10px'}
            ),
            dcc.Input(
                id='y-axis-max',
                type='number',
                placeholder='Max',
                style={'width': '20%'}
            ),
        ], style={'display': 'inline-block', 'padding': '10px'}),

        dcc.Graph(id='displacement-graph', style={'height': '50vh', 'width': '95vw'})
    ], style={'display': 'none'})
])
@app.callback(
    Output('map', 'figure'),
    [Input('map-style-dropdown', 'value'),
     Input('color-mode-dropdown', 'value'),
     Input('orbit-filter-dropdown', 'value')]
)
def update_map(map_style, color_mode, orbit_filter):
    data = all_data.drop_duplicates(subset=['pid'])

    if isinstance(orbit_filter, str):
        orbit_filter = [orbit_filter]

    filtered_data = data[data['file'].isin(orbit_filter)]
    filtered_data.loc[:, 'mean_velocity'] = filtered_data['mean_velocity'].round(1)

    if color_mode == 'orbit':
        fig = px.scatter_mapbox(filtered_data,
                                lat='latitude', lon='longitude',
                                hover_name='pid',
                                hover_data={
                                    'latitude': True,
                                    'longitude': True,
                                    'height': True,
                                    'mean_velocity': True
                                },
                                labels={
                                    'latitude': 'Latitude',
                                    'longitude': 'Longitude',
                                    'height': 'Height',
                                    'mean_velocity': 'Mean Velocity'
                                },
                                color='file',
                                zoom=14)

        fig.update_layout(legend_title_text='Orbit Type')


    elif color_mode == 'speed':
        fig = px.scatter_mapbox(filtered_data,
                                lat='latitude', lon='longitude',
                                hover_name='pid',
                                hover_data={
                                    'latitude': True,
                                    'longitude': True,
                                    'height': True,
                                    'mean_velocity': True
                                },
                                color='mean_velocity',
                                color_continuous_scale='Jet', 
                                range_color=(-5, 5), 
                                labels={
                                    'latitude': 'Latitude',
                                    'longitude': 'Longitude',
                                    'height': 'Height',
                                    'mean_velocity': 'Mean Velocity'
                                },
                                zoom=14)

        fig.update_layout(legend_title_text='Mean Velocity [mm/year]')

    elif color_mode == 'anomaly_type':

        merged_data = filtered_data.merge(all_anomaly_data_99[['pid', 'is_anomaly']], on='pid', how='left')

        merged_data['is_anomaly'] = merged_data['is_anomaly'].fillna(False)
        merged_data['is_anomaly'] = merged_data['is_anomaly'].infer_objects().astype(bool)

        merged_data['anomaly_group'] = (
            (merged_data['is_anomaly'] != merged_data['is_anomaly'].shift()) 
            | (merged_data['pid'] != merged_data['pid'].shift())
        ).cumsum()

        merged_data['anomaly_streak_count'] = merged_data.groupby(['pid', 'anomaly_group'])['is_anomaly'].transform('sum')

        merged_data['true_anomaly'] = merged_data.apply(
            lambda row: True if row['anomaly_streak_count'] > 3 and row['is_anomaly'] else False, axis=1
        )

        fig = px.scatter_mapbox(merged_data,
                                lat='latitude', lon='longitude',
                                hover_name='pid',
                                hover_data={
                                    'latitude': True,
                                    'longitude': True,
                                    'height': True,
                                    'mean_velocity': True
                                },
                                labels={
                                    'latitude': 'Latitude',
                                    'longitude': 'Longitude',
                                    'height': 'Height',
                                    'mean_velocity': 'Mean Velocity'
                                },
                                color=merged_data['true_anomaly'].map({True: 'Anomaly', False: 'No Anomaly'}),
                                color_discrete_map={'Anomaly': 'red', 'No Anomaly': 'green'},
                                zoom=14)

        fig.update_layout(
            legend_title_text='Anomaly Type',
            mapbox_style=map_style,
            autosize=True,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
    fig.update_layout(
        mapbox_style=map_style,
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0))
        
    return fig

@app.callback(
    Output('selected-points', 'data'),
    [Input('map', 'clickData')],
    [State('selected-points', 'data')]
)
def update_selected_points(clickData, selected_points):
    if clickData is None:
        return selected_points
    
    point_id = clickData['points'][0]['hovertext']
    lat = clickData['points'][0]['lat']
    lon = clickData['points'][0]['lon']
    
    if selected_points['point_1'] is None:
        selected_points['point_1'] = {'pid': point_id, 'lat': lat, 'lon': lon}
    elif selected_points['point_2'] is None:
        selected_points['point_2'] = {'pid': point_id, 'lat': lat, 'lon': lon}
    else:
        selected_points = {'point_1': None, 'point_2': None}

    return selected_points

@app.callback(
    Output('distance-output', 'children'),
    [Input('selected-points', 'data'),
     Input('distance-calc-dropdown', 'value')]
)
def display_distance(selected_points, distance_calc_enabled):
    if distance_calc_enabled == 'no':
        return ""

    point_1 = selected_points['point_1']
    point_2 = selected_points['point_2']
    
    if point_1 is not None and point_2 is not None:
        coords_1 = (point_1['lat'], point_1['lon'])
        coords_2 = (point_2['lat'], point_2['lon'])

        distance_km = geodesic(coords_1, coords_2).kilometers
        
        return html.Div([
            html.H4("Selected Points and Distance"),
            html.Ul([
                html.Li(f"Point 1: {point_1['pid']} (Lat: {point_1['lat']}, Lon: {point_1['lon']})"),
                html.Li(f"Point 2: {point_2['pid']} (Lat: {point_2['lat']}, Lon: {point_2['lon']})"),
                html.Li(f"Distance: {distance_km:.2f} km")
            ], style={'list-style-type': 'none', 'padding': '0', 'margin': '0'})
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px'})
    else:
        return "Select two points on the map to calculate the distance."

@app.callback(
    [Output('displacement-graph', 'figure'), Output('displacement-container', 'style')],
    [Input('map', 'clickData'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('y-axis-min', 'value'),
     Input('y-axis-max', 'value')]
)
def display_displacement(clickData, start_date, end_date, y_min, y_max):
    if clickData is None:
        return {}, {'display': 'none'}

    point_id = clickData['points'][0]['hovertext']
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    full_data = all_data[(all_data['pid'] == point_id)].copy()
    filtered_data = full_data[(full_data['timestamp'] >= start_date) & (full_data['timestamp'] <= end_date)]

    last_n_data = full_data.tail(60)
    
    filtered_anomalies_95 = all_anomaly_data_95[(all_anomaly_data_95['pid'] == point_id)]
    filtered_anomalies_99 = all_anomaly_data_99[(all_anomaly_data_99['pid'] == point_id)]

    if not filtered_anomalies_95.empty:
        if len(filtered_anomalies_95) > len(last_n_data):
            filtered_anomalies_95 = filtered_anomalies_95.tail(len(last_n_data))

        filtered_anomalies_95['timestamp'] = last_n_data['timestamp'].values[:len(filtered_anomalies_95)]
        filtered_anomalies_95.set_index('timestamp', inplace=True)

        last_n_data.set_index('timestamp', inplace=True)

        last_n_data = last_n_data.join(
            filtered_anomalies_95[['predicted_value', 'upper_bound', 'lower_bound', 'is_anomaly']], 
            how='left'
        )

    if not filtered_anomalies_99.empty:
        if len(filtered_anomalies_99) > len(last_n_data):
            filtered_anomalies_99 = filtered_anomalies_99.tail(len(last_n_data))

        filtered_anomalies_99['timestamp'] = last_n_data.index.values[:len(filtered_anomalies_99)]
        filtered_anomalies_99.set_index('timestamp', inplace=True)

        last_n_data = last_n_data.join(
            filtered_anomalies_99[['upper_bound', 'lower_bound', 'is_anomaly']], 
            how='left', rsuffix='_99'
        )

    fig = px.line(filtered_data, x='timestamp', y='displacement', 
                  title=f"Displacement LOS for point {point_id}",
                  markers=True, 
                  labels={'displacement': 'Displacement[mm]'})

    fig.add_scatter(x=filtered_data['timestamp'], y=filtered_data['displacement'], 
                    mode='lines+markers', 
                    name='InSAR measured displacement', 
                    line=dict(color='blue'))

    if not last_n_data.empty and not last_n_data[(last_n_data.index >= start_date) & (last_n_data.index <= end_date)].empty:
        fig.add_scatter(x=last_n_data.index, y=last_n_data['predicted_value'], 
                        mode='lines+markers', 
                        name='Predicted Displacement', 
                        line=dict(color='orange'))

        fig.add_scatter(x=last_n_data.index, 
                        y=last_n_data['upper_bound'], 
                        mode='lines', 
                        line=dict(color='yellow', dash='dash'),
                        name='Upper Bound p=95')

        fig.add_scatter(x=last_n_data.index, 
                        y=last_n_data['lower_bound'],
                        mode='lines',
                        line=dict(color='yellow', dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(255, 252, 127, 0.2)',
                        name='Lower Bound p=95')

        anomalies_95 = last_n_data[last_n_data['is_anomaly'] == 1]
        if not anomalies_95.empty:
            fig.add_scatter(x=anomalies_95.index, y=anomalies_95['displacement'], 
                            mode='markers', 
                            name='Anomalies p=95', 
                            marker=dict(color='yellow', size=10))

    if 'upper_bound_99' in last_n_data.columns and 'lower_bound_99' in last_n_data.columns:
        fig.add_scatter(x=last_n_data.index, 
                        y=last_n_data['upper_bound_99'], 
                        mode='lines', 
                        line=dict(color='red', dash='dash'),
                        name='Upper Bound p=99')

        fig.add_scatter(x=last_n_data.index, 
                        y=last_n_data['lower_bound_99'],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(254, 121, 104, 0.1)',
                        name='Lower Bound p=99')

        
        anomalies_99 = last_n_data[last_n_data['is_anomaly_99'] == 1]
        if not anomalies_99.empty:
            fig.add_scatter(x=anomalies_99.index, y=anomalies_99['displacement'], 
                            mode='markers', 
                            name='Anomalies p=99', 
                            marker=dict(color='red', size=10))

    if y_min is not None and y_max is not None:
        fig.update_yaxes(range=[y_min, y_max])

    fig.update_layout(
        xaxis_title='Date', 
        yaxis_title='Displacement LOS[mm]', 
        legend_title="Legend",
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.05)
    )

    return fig, {'display': 'block'}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
