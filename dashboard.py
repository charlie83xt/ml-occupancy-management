from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import os
import subprocess
import threading
from utils import merge_return_full_df, prepare_occupancy_df
from sklearn.metrics import root_mean_squared_error

def run_data_update():
    subprocess.Popen(["python", "main.py"])

threading.Thread(target=run_data_update).start()

# Load initial data
DATA_PATH = "data/processed/processed.csv"
pred_path = "data/processed/predictions.csv"
occupancy_path = 'data/raw_occupancy'
def load_predictions():
    try:
        if os.path.exists(pred_path):
            pred_df = pd.read_csv(pred_path)
            pred_df["Summary_Date_Local"] = pd.to_datetime(pred_df["Summary_Date_Local"], errors='coerce', format="ISO8601")
            pred_df = pred_df[pred_df['Summary_Date_Local'].dt.weekday < 5]
            return pred_df
    except pd.errors.EmptyDataError:
        print(f"Warning: {pred_path} is empty. Returning empty dataframe for predictions")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading predictions from: {pred_path}: {e}")
        return pd.DataFrame()
    return pd.DataFrame()
    

def load_data():
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            # Ensure Summary_Date_Local is parsed correctly.
            df["Summary_Date_Local"] = pd.to_datetime(df["Summary_Date_Local"], errors='coerce', format="%Y-%m-%d")
            df.dropna(subset=["Summary_Date_Local"], inplace=True)
            df = df[df['Summary_Date_Local'].dt.weekday < 5]
            return df
        except pd.errors.EmptyDataError:
            print(f"Warning: {DATA_PATH} is empty. Returning empty DataFrame for main data.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading main data from {DATA_PATH}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# Initialise app
app = Dash(__name__)
server = app.server # for deployment

# Layout
app.layout = html.Div(style={"backgroundcolor": "#f4f4f4f4", "padding": "20px"}, children=[
    html.H1('Office Occupancy Dashboard', style={'textAlign': 'center', 'color': '#333333'}),

    html.Div(id='last-updated', style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#666666'}),

    html.Div([
        html.Div([
            html.Label("Select Occupancy Metric:", style={'color': '#333333'}),
            dcc.Dropdown(
                id="occupancy-type",
                options=[
                    {"label": "Desk Occupancy (used spaces)", "value": "used_spaces"},
                    {"label": "Occupancy Rate (%)", "value": "Occupancy_Rates"}
                ],
                value="Occupancy_Rates",
                clearable=False,
                style={"color": "#000000"}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={"marginBottom": "20px"}),

    html.Div([
        html.Div([
            dcc.Graph(id="occupancy-trend")
        ], style={'width': '100%', 'marginBottom': '40px'}),

        html.Div([
            dcc.Graph(id="heatmap")
        ], style={'width': '100%', 'marginBottom': '40px'}),

        html.Div([
            dcc.Graph(id="scatter-plot")
        ], style={'width': '100%', 'marginBottom': '40px'}),

        html.Div([
            html.H4("Model Accuracy", style={"color": "#333333"}),
            html.Div(id="accuracy-card", style={
                'fontSize': '24px',
                'padding': '10px',
                'backgroundColor': '#ffffff',
                'border': '1px solid #cccccc',
                'borderRadius': '5px',
                'textAlign': 'center',
                'color': '#333333'
            })
        ], style={'width': '100%', 'marginBottom': '40px'}),

        html.Div([
            dcc.Graph(id="actual-vs-predicted")
        ], style={'width': '100%'})
    ]),

    dcc.Interval(
        id='interval-component',
        interval=5*60*1000, # 5 minutes
        n_intervals=0
    )
])

@app.callback(
    Output("occupancy-trend", "figure"),
    Output("heatmap", "figure"),
    Output("scatter-plot", "figure"),
    Output("accuracy-card", "children"),
    Output("actual-vs-predicted", "figure"),
    Output("last-updated", "children"),
    Input("occupancy-type", "value"),
    Input("interval-component", "n_intervals")
)

def updated_dashboard(selected_metric, n):
    df = load_data()
    pred_df = load_predictions()

    # Handle empty main DataFrame
    if df.empty:
        print("Main data DataFrame is empty. Cannot generate plots.")
        # Return empty figures and N/A for all outputs
        empty_fig = px.line(title="No Data Available")
        return empty_fig, empty_fig, empty_fig, "Accuracy: N/A", empty_fig, "Last updated: N/A"

    # Logic to merge both dataframes
    df["Summary_Date_Local"] = pd.to_datetime(df["Summary_Date_Local"], format="%Y-%m-%d")
    pred_df["Summary_Date_Local"] = pd.to_datetime(pred_df["Summary_Date_Local"], format="%Y-%m-%d")
    df = pd.merge(df, pred_df[["Summary_Date_Local", "Predicted_Occupancy"]], on="Summary_Date_Local", how='outer')
    # print(df.head(20)) ########

    # Last updated timestamp
    mod_time = os.path.getmtime(DATA_PATH)
    last_updated = f"Last updated: {datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')}"


    ###### Lets add here only Occupancy DataFrame #########
    full_occupancy_df = merge_return_full_df(occupancy_path) 
    pross_occupancy_df = prepare_occupancy_df(full_occupancy_df)
    ########## ############## ############# ####################
    

    # Bar plot - Last month
    latest_date = pd.to_datetime(pross_occupancy_df["Summary_Date_Local"]).max()
    last_month = latest_date - pd.DateOffset(months=1)
    pross_occupancy_df = pross_occupancy_df[pross_occupancy_df['Summary_Date_Local'].dt.weekday < 5]
    last_month_df = pross_occupancy_df[pd.to_datetime(pross_occupancy_df["Summary_Date_Local"]) >= last_month].copy()
    last_month_df['Day-date'] = pd.to_datetime(last_month_df["Summary_Date_Local"]).dt.strftime('%a%d')

    color_map = {
        'used_spaces': '#ff00b4',
        'Occupancy_Rates': '#ff790e'
    }

    fig_bar = px.bar(
        last_month_df,
        x="Day-date",
        y=selected_metric,
        title="Occupancy Trends Over Time",
        labels={selected_metric: "Occupancy"}
    )
    fig_bar.update_traces(marker_color=color_map[selected_metric])
    fig_bar.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font_color='#333333',
        title_font_size=20
    )

    # Heatmap - Last 5 weeks
    five_weeks_ago = latest_date - pd.DateOffset(weeks=6)
    last_5weeks_df = df[pd.to_datetime(df["Summary_Date_Local"]) >= five_weeks_ago].copy()
    last_5weeks_df['Week'] = pd.to_datetime(last_5weeks_df["Summary_Date_Local"]).dt.isocalendar().week
    last_5weeks_df['Week'] = last_5weeks_df['Week'].astype("int64")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    last_5weeks_df["DayOfWeek"] = pd.Categorical(last_5weeks_df["DayOfWeek"], categories=day_order, ordered=True)

    heatmap_data = last_5weeks_df.pivot_table(
        index='DayOfWeek',
        columns='Week',
        values=selected_metric,
        aggfunc='mean'
    )

    fig_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x="Week Number", y="Day of Week", color="Occupancy"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        title="Weekly Occupancy Rates"
    )
    
    fig_heatmap.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font_color='#333333',
        title_font_size=20,xaxis_title="Week Number",
        yaxis_title="Day of Week"
    )

    fig_heatmap.update_xaxes(side='top')
    fig_heatmap.update_yaxes(type='category')

    # Scatterplot
    fig_scatter = px.scatter(
        df,
        x="temperature_2m_max",
        y=selected_metric,
        color="Weather Label",
        title="Temperature vs Occupancy",
        labels={
            "temperature_2m_max": "Max Temperature (°C)",
            selected_metric: "Occupancy"
        }
    )
    
    fig_scatter.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font_color='#333333',
        title_font_size=20,
        xaxis_title="Max Temperature (°C)",
        yaxis_title="Occupancy"
    )
    fig_scatter.update_xaxes(side='top')

    # Actual vs Predicted Line plot
    if "Predicted_Occupancy" in df.columns:
        actual_vs_pred = df[["Summary_Date_Local", selected_metric, "Predicted_Occupancy"]].copy()
        actual_vs_pred["Summary_Date_Local"] = pd.to_datetime(actual_vs_pred["Summary_Date_Local"])
        actual_vs_pred = actual_vs_pred.sort_values("Summary_Date_Local")
        # last_date = actual_vs_pred["Summary_Date_Local"].max()
        start_date = latest_date - pd.Timedelta(days=25)
        week_ago = pd.to_datetime(datetime.today() - timedelta(days=10))
        plot_df = actual_vs_pred[(actual_vs_pred["Summary_Date_Local"] > start_date) & 
                                 (actual_vs_pred["Summary_Date_Local"] <= week_ago)].copy()   
        plot_df = plot_df.dropna(subset=['Occupancy_Rates', 'Predicted_Occupancy'], how='all').copy()
        

        fig_actual_pred = px.line(
        plot_df,
        x="Summary_Date_Local",
        y=[selected_metric, "Predicted_Occupancy"],
        title="Actual vs Predicted Occupancy",
        labels={"value": "Occupancy", "Summary_Date_Local": "Date", "variable": "Legend"}
    )
        fig_actual_pred.update_layout(
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font_color='#333333',
            title_font_size=20,
        )
    else:
        fig_actual_pred = px.line(title="Actual vs Predicted Occupancy (No prediction data available)")   
    
    # Accuracy card (placeholder)
    if not actual_vs_pred.empty:
        valid_rows = actual_vs_pred.dropna(subset=["Occupancy_Rates", "Predicted_Occupancy"])
        if not valid_rows.empty:
            rmse = root_mean_squared_error(valid_rows["Occupancy_Rates"], valid_rows["Predicted_Occupancy"])
            # accuracy = df["model_accuracy"].iloc[-1] if "model_accuracy" in df.columns else "N/A"
            accuracy_text = f"Accuracy: {100 - rmse:.2f}% (RMSE: {rmse:.2f})" 
        else:
            accuracy_text = "Accuracy: N/A"
    else:
        accuracy_text = "Accuracy: N/A"

    return fig_bar, fig_heatmap, fig_scatter, accuracy_text, fig_actual_pred, last_updated
    
# Run the app
if __name__ == "__main__":
    app.run(debug=True)