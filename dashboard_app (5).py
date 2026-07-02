"""
EV Battery Degradation Dashboard — Industrial IoT Project
==========================================================
Dataset  : ev_battery_degradation_v1.csv  (10,000 EV records)
Framework: Plotly Dash

HOW TO RUN
----------
1.  Run the Colab notebook and download dashboard_data.csv, OR
    just rename ev_battery_degradation_v1.csv → dashboard_data.csv
2.  Place dashboard_data.csv in the same folder as this file.
3.  Install dependencies:
        pip install dash plotly pandas numpy
4.  Run:
        python dashboard_app.py
5.  Open:  http://127.0.0.1:8050
"""

import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, callback

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

try:
    df = pd.read_csv("ev_battery_degradation_v1.csv")
    print(f"✅ Loaded {len(df):,} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print("\n❌  dashboard_data.csv not found.")
    print("    Either run the Colab notebook, or rename ev_battery_degradation_v1.csv")
    print("    to dashboard_data.csv and place it here.")
    sys.exit(1)

# ── Derived columns ──────────────────────────────────────────────────────────
REPLACE_THRESHOLD = 80.0   # SoH % below which battery should be replaced

df["health_status_calc"] = np.where(df["SoH_Percent"] >= REPLACE_THRESHOLD,
                                    "Healthy", "Replace Required")

bins   = [-15, 0, 15, 25, 35, 60]
labels = ["< 0°C", "0–15°C", "15–25°C", "25–35°C", "> 35°C"]
df["temp_band"] = pd.cut(df["Avg_Temperature_C"], bins=bins, labels=labels)

# ── Constants ────────────────────────────────────────────────────────────────
MODELS  = sorted(df["Car_Model"].unique().tolist())
STYLES  = ["Conservative", "Moderate", "Aggressive"]
BTYPES  = sorted(df["Battery_Type"].unique().tolist())

MODEL_COLORS = {
    "Tesla Model 3":       "#E31937",
    "Ford Mustang Mach-E": "#003A8C",
    "Hyundai Ioniq 5":     "#00AAD4",
    "Wuling Air EV":       "#E5002D",
    "BYD Atto 3":          "#1DB954",
}
STYLE_COLORS  = {"Aggressive": "#E74C3C", "Moderate": "#F39C12", "Conservative": "#2ECC71"}
STATUS_COLORS = {"Healthy": "#27AE60", "Replace Required": "#E74C3C"}
BTYPE_COLORS  = {"NMC": "#3498DB", "LFP": "#27AE60"}

BASE_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#f9f9f9",
    font=dict(family="Arial, sans-serif", size=12, color="#333"),
    margin=dict(l=55, r=25, t=50, b=50),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", borderwidth=0.5),
    hoverlabel=dict(bgcolor="white", font_size=12),
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FLEET-LEVEL KPIs  (computed once from full dataset)
# ─────────────────────────────────────────────────────────────────────────────

fleet_avg_soh       = df["SoH_Percent"].mean()
fleet_replace_count = int((df["Battery_Status"] == "Replace Required").sum())
fleet_replace_pct   = fleet_replace_count / len(df) * 100
fleet_avg_cycles    = df["Total_Charging_Cycles"].mean()
fleet_avg_resistance= df["Internal_Resistance_Ohm"].mean()

# ─────────────────────────────────────────────────────────────────────────────
# 3.  FIGURE BUILDERS  (all read the global df, filtered by selection)
# ─────────────────────────────────────────────────────────────────────────────

def filter_df(models, styles, btypes):
    mask = (
        df["Car_Model"].isin(models) &
        df["Driving_Style"].isin(styles) &
        df["Battery_Type"].isin(btypes)
    )
    return df[mask]


# ── Chart 1: SoH Distribution (box per model) ────────────────────────────────
def fig_soh_box(dff):
    order = dff.groupby("Car_Model")["SoH_Percent"].median().sort_values().index.tolist()
    fig = go.Figure()
    for model in order:
        sub = dff[dff.Car_Model == model]
        fig.add_trace(go.Box(
            y=sub["SoH_Percent"], name=model,
            marker_color=MODEL_COLORS.get(model, "#999"),
            boxmean="sd",
            hovertemplate=f"<b>{model}</b><br>SoH: %{{y:.1f}}%<extra></extra>",
        ))
    fig.add_hline(y=REPLACE_THRESHOLD, line_dash="dash", line_color="red",
                  annotation_text=f"Replace threshold {REPLACE_THRESHOLD}%",
                  annotation_position="bottom right")
    fig.update_layout(title="SoH Distribution by Car Model",
                      yaxis_title="State of Health (%)",
                      showlegend=False, **BASE_LAYOUT)
    return fig


# ── Chart 2: SoH vs Charging Cycles (scatter) ────────────────────────────────
def fig_soh_vs_cycles(dff):
    fig = go.Figure()
    for model in dff["Car_Model"].unique():
        sub = dff[dff.Car_Model == model]
        fig.add_trace(go.Scatter(
            x=sub["Total_Charging_Cycles"], y=sub["SoH_Percent"],
            name=model, mode="markers",
            marker=dict(color=MODEL_COLORS.get(model, "#999"), size=4, opacity=0.45),
            hovertemplate=(f"<b>{model}</b><br>"
                           "Cycles: %{x}<br>SoH: %{y:.2f}%<extra></extra>"),
        ))
    # Trend line on filtered data
    if len(dff) > 10:
        z = np.polyfit(dff["Total_Charging_Cycles"], dff["SoH_Percent"], 1)
        xs = np.linspace(dff["Total_Charging_Cycles"].min(),
                         dff["Total_Charging_Cycles"].max(), 300)
        r = dff["Total_Charging_Cycles"].corr(dff["SoH_Percent"])
        fig.add_trace(go.Scatter(
            x=xs, y=np.polyval(z, xs),
            name=f"Trend  (r={r:.3f})", mode="lines",
            line=dict(color="black", width=2, dash="dash"),
        ))
    fig.update_layout(title="SoH vs Total Charging Cycles",
                      xaxis_title="Total Charging Cycles",
                      yaxis_title="SoH (%)", **BASE_LAYOUT)
    return fig


# ── Chart 3: SoH vs Vehicle Age ──────────────────────────────────────────────
def fig_soh_vs_age(dff):
    fig = go.Figure()
    for model in dff["Car_Model"].unique():
        sub = dff[dff.Car_Model == model]
        fig.add_trace(go.Scatter(
            x=sub["Vehicle_Age_Months"], y=sub["SoH_Percent"],
            name=model, mode="markers",
            marker=dict(color=MODEL_COLORS.get(model, "#999"), size=4, opacity=0.45),
            hovertemplate=(f"<b>{model}</b><br>"
                           "Age: %{x} months<br>SoH: %{y:.2f}%<extra></extra>"),
        ))
    if len(dff) > 10:
        z = np.polyfit(dff["Vehicle_Age_Months"], dff["SoH_Percent"], 1)
        xs = np.linspace(dff["Vehicle_Age_Months"].min(),
                         dff["Vehicle_Age_Months"].max(), 300)
        r = dff["Vehicle_Age_Months"].corr(dff["SoH_Percent"])
        fig.add_trace(go.Scatter(
            x=xs, y=np.polyval(z, xs),
            name=f"Trend  (r={r:.3f})", mode="lines",
            line=dict(color="black", width=2, dash="dash"),
        ))
    fig.update_layout(title="SoH vs Vehicle Age",
                      xaxis_title="Vehicle Age (Months)",
                      yaxis_title="SoH (%)", **BASE_LAYOUT)
    return fig


# ── Chart 4: Internal Resistance vs SoH ──────────────────────────────────────
def fig_resistance_soh(dff):
    fig = go.Figure()
    for btype in dff["Battery_Type"].unique():
        sub = dff[dff.Battery_Type == btype]
        fig.add_trace(go.Scatter(
            x=sub["Internal_Resistance_Ohm"], y=sub["SoH_Percent"],
            name=btype, mode="markers",
            marker=dict(color=BTYPE_COLORS.get(btype, "#999"), size=4, opacity=0.45),
            hovertemplate=(f"<b>{btype}</b><br>"
                           "Resistance: %{x:.4f} Ω<br>SoH: %{y:.2f}%<extra></extra>"),
        ))
    if len(dff) > 10:
        z = np.polyfit(dff["Internal_Resistance_Ohm"], dff["SoH_Percent"], 1)
        xs = np.linspace(dff["Internal_Resistance_Ohm"].min(),
                         dff["Internal_Resistance_Ohm"].max(), 300)
        r = dff["Internal_Resistance_Ohm"].corr(dff["SoH_Percent"])
        fig.add_trace(go.Scatter(
            x=xs, y=np.polyval(z, xs),
            name=f"Trend  (r={r:.3f})", mode="lines",
            line=dict(color="black", width=2, dash="dash"),
        ))
    fig.update_layout(title="Internal Resistance vs SoH",
                      xaxis_title="Internal Resistance (Ω)",
                      yaxis_title="SoH (%)", **BASE_LAYOUT)
    return fig


# ── Chart 5: Temperature band bar chart ──────────────────────────────────────
def fig_temp_bar(dff):
    avg_soh = (dff.groupby("temp_band", observed=True)["SoH_Percent"]
                  .mean().reset_index())
    bar_colors = ["#3498DB", "#1ABC9C", "#2ECC71", "#F39C12", "#E74C3C"]
    fig = go.Figure(go.Bar(
        x=avg_soh["temp_band"].astype(str),
        y=avg_soh["SoH_Percent"].round(2),
        marker_color=bar_colors[:len(avg_soh)],
        text=avg_soh["SoH_Percent"].round(2).astype(str) + "%",
        textposition="inside",
        hovertemplate="Band: %{x}<br>Avg SoH: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(title="Avg SoH by Operating Temperature Band",
                      xaxis_title="Temperature Band",
                      yaxis_title="Average SoH (%)",
                      yaxis=dict(range=[
                          avg_soh["SoH_Percent"].min() - 2,
                          avg_soh["SoH_Percent"].max() + 1
                      ]),
                      **BASE_LAYOUT)
    return fig


# ── Chart 6: Fast Charge Ratio vs SoH (box by driving style) ─────────────────
def fig_fast_charge(dff):
    fig = go.Figure()
    bins_fc  = [0, 0.25, 0.5, 0.75, 1.01]
    lbl_fc   = ["0–25%", "25–50%", "50–75%", "75–100%"]
    dff = dff.copy()
    dff["fc_band"] = pd.cut(dff["Fast_Charge_Ratio"],
                            bins=bins_fc, labels=lbl_fc, include_lowest=True)
    for style in STYLES:
        sub = dff[dff.Driving_Style == style]
        avg = sub.groupby("fc_band", observed=True)["SoH_Percent"].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=avg["fc_band"].astype(str), y=avg["SoH_Percent"],
            name=style, mode="lines+markers",
            line=dict(color=STYLE_COLORS[style], width=2.5),
            marker=dict(size=8),
            hovertemplate=f"<b>{style}</b><br>Fast Charge: %{{x}}<br>Avg SoH: %{{y:.2f}}%<extra></extra>",
        ))
    fig.update_layout(title="Avg SoH by Fast Charge Ratio & Driving Style",
                      xaxis_title="Fast Charge Ratio Band",
                      yaxis_title="Average SoH (%)", **BASE_LAYOUT)
    return fig


# ── Chart 7: Battery Status donut ────────────────────────────────────────────
def fig_status_donut(dff):
    counts = dff["Battery_Status"].value_counts().reset_index()
    counts.columns = ["status", "count"]
    fig = go.Figure(go.Pie(
        labels=counts["status"], values=counts["count"],
        hole=0.55,
        marker_colors=[STATUS_COLORS.get(s, "#aaa") for s in counts["status"]],
        hovertemplate="%{label}: %{value:,} vehicles (%{percent})<extra></extra>",
    ))
    fig.update_layout(title="Fleet Battery Status",
                      **BASE_LAYOUT)
    return fig


# ── Chart 8: SoH Heatmap — Driving Style × Car Model ─────────────────────────
def fig_heatmap(dff):
    pivot = (dff.pivot_table(values="SoH_Percent",
                             index="Driving_Style",
                             columns="Car_Model",
                             aggfunc="mean")
               .reindex(["Conservative", "Moderate", "Aggressive"]))

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        zmin=92, zmax=99,
        text=np.round(pivot.values, 1),
        texttemplate="%{text}%",
        hovertemplate="Driving Style: %{y}<br>Model: %{x}<br>Avg SoH: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Avg SoH (%)"),
    ))
    fig.update_layout(title="Avg SoH Heatmap — Driving Style × Car Model",
                      xaxis_title="", yaxis_title="Driving Style",
                      **BASE_LAYOUT)
    return fig


# ── Chart 9: NMC vs LFP box per model ────────────────────────────────────────
def fig_chemistry_compare(dff):
    fig = go.Figure()
    for btype in ["NMC", "LFP"]:
        sub = dff[dff.Battery_Type == btype]
        fig.add_trace(go.Box(
            y=sub["SoH_Percent"],
            x=sub["Car_Model"],
            name=btype,
            marker_color=BTYPE_COLORS[btype],
            boxmean=True,
            hovertemplate=f"<b>{btype}</b><br>%{{x}}<br>SoH: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(title="SoH — NMC vs LFP per Car Model",
                      xaxis_title="", yaxis_title="SoH (%)",
                      boxmode="group", **BASE_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LAYOUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def kpi_card(label, value, sub_text, accent):
    return html.Div(style={
        "background": "white", "borderRadius": "8px",
        "padding": "16px 20px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
        "borderLeft": f"4px solid {accent}",
    }, children=[
        html.P(label, style={
            "margin": 0, "fontSize": "11px", "color": "#888",
            "textTransform": "uppercase", "letterSpacing": "0.06em",
        }),
        html.H2(value, style={
            "margin": "4px 0", "fontSize": "24px",
            "fontWeight": "700", "color": accent,
        }),
        html.P(sub_text, style={"margin": 0, "fontSize": "11px", "color": "#aaa"}),
    ])


def chart_card(child, span=1):
    style = {
        "background": "white", "borderRadius": "8px",
        "padding": "16px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
    }
    if span == 2:
        style["gridColumn"] = "span 2"
    return html.Div(style=style, children=[child])


NO_BAR = {"displayModeBar": False}

# ─────────────────────────────────────────────────────────────────────────────
# 5.  APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

app = Dash(__name__, title="EV Battery Degradation Dashboard")

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif",
           "background": "#eef1f5", "minHeight": "100vh"},
    children=[

        # ── Header ────────────────────────────────────────────────────────────
        html.Div(style={
            "background": "linear-gradient(135deg, #0D47A1 0%, #1976D2 100%)",
            "color": "white", "padding": "22px 32px",
        }, children=[
            html.H1("⚡ EV Battery Degradation Dashboard",
                    style={"margin": 0, "fontSize": "24px", "fontWeight": "700"}),
            html.P("Industrial IoT  ·  10,000 Vehicles  ·  "
                   "Tesla · Ford · Hyundai · Wuling · BYD  ·  NMC & LFP Chemistries",
                   style={"margin": "5px 0 0", "opacity": 0.75, "fontSize": "12px"}),
        ]),

        html.Div(style={"padding": "22px 32px"}, children=[

            # ── KPI row ───────────────────────────────────────────────────────
            html.Div(style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "14px", "marginBottom": "20px",
            }, children=[
                kpi_card("Fleet Avg SoH",
                         f"{fleet_avg_soh:.1f}%",
                         "Across all 10,000 vehicles", "#1565C0"),
                kpi_card("Replace Required",
                         f"{fleet_replace_count:,}  ({fleet_replace_pct:.1f}%)",
                         "Vehicles below 80% SoH", "#C62828"),
                kpi_card("Avg Charging Cycles",
                         f"{fleet_avg_cycles:.0f}",
                         "Mean cycles per vehicle", "#2E7D32"),
                kpi_card("Avg Internal Resistance",
                         f"{fleet_avg_resistance*1000:.1f} mΩ",
                         "Fleet mean resistance", "#6A1B9A"),
            ]),

            # ── Filters ───────────────────────────────────────────────────────
            html.Div(style={
                "background": "white", "borderRadius": "8px",
                "padding": "16px 22px", "marginBottom": "20px",
                "boxShadow": "0 1px 4px rgba(0,0,0,0.07)",
            }, children=[
                html.P("🔍 Filters — all charts update together",
                       style={"margin": "0 0 12px", "fontWeight": "600",
                              "fontSize": "13px", "color": "#555"}),
                html.Div(style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr 1fr",
                    "gap": "16px",
                }, children=[
                    html.Div([
                        html.Label("Car Model", style={"fontSize": "12px",
                                                       "color": "#777",
                                                       "marginBottom": "6px",
                                                       "display": "block"}),
                        dcc.Checklist(
                            id="filter-model",
                            options=[{"label": f"  {m}", "value": m} for m in MODELS],
                            value=MODELS,
                            inputStyle={"marginRight": "5px"},
                            labelStyle={"display": "block", "marginBottom": "4px",
                                        "fontSize": "13px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Driving Style", style={"fontSize": "12px",
                                                            "color": "#777",
                                                            "marginBottom": "6px",
                                                            "display": "block"}),
                        dcc.Checklist(
                            id="filter-style",
                            options=[{"label": f"  {s}", "value": s} for s in STYLES],
                            value=STYLES,
                            inputStyle={"marginRight": "5px"},
                            labelStyle={"display": "block", "marginBottom": "4px",
                                        "fontSize": "13px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Battery Chemistry", style={"fontSize": "12px",
                                                                "color": "#777",
                                                                "marginBottom": "6px",
                                                                "display": "block"}),
                        dcc.Checklist(
                            id="filter-btype",
                            options=[{"label": f"  {b}", "value": b} for b in BTYPES],
                            value=BTYPES,
                            inputStyle={"marginRight": "5px"},
                            labelStyle={"display": "block", "marginBottom": "4px",
                                        "fontSize": "13px"},
                        ),
                        html.Div(id="row-count",
                                 style={"marginTop": "12px", "fontSize": "12px",
                                        "color": "#888"}),
                    ]),
                ]),
            ]),

            # ── Row 1: SoH box (full width) ───────────────────────────────────
            html.Div(style={"marginBottom": "14px"},
                     children=[chart_card(
                         dcc.Graph(id="g-soh-box", config=NO_BAR),
                         span=2
                     )]),

            # ── Row 2: SoH vs Cycles | SoH vs Age ────────────────────────────
            html.Div(style={
                "display": "grid", "gridTemplateColumns": "1fr 1fr",
                "gap": "14px", "marginBottom": "14px",
            }, children=[
                chart_card(dcc.Graph(id="g-soh-cycles",  config=NO_BAR)),
                chart_card(dcc.Graph(id="g-soh-age",     config=NO_BAR)),
            ]),

            # ── Row 3: Resistance vs SoH | Temperature bar ───────────────────
            html.Div(style={
                "display": "grid", "gridTemplateColumns": "1fr 1fr",
                "gap": "14px", "marginBottom": "14px",
            }, children=[
                chart_card(dcc.Graph(id="g-resistance",  config=NO_BAR)),
                chart_card(dcc.Graph(id="g-temp-bar",    config=NO_BAR)),
            ]),

            # ── Row 4: Fast charge lines | Status donut ───────────────────────
            html.Div(style={
                "display": "grid", "gridTemplateColumns": "1fr 1fr",
                "gap": "14px", "marginBottom": "14px",
            }, children=[
                chart_card(dcc.Graph(id="g-fast-charge", config=NO_BAR)),
                chart_card(dcc.Graph(id="g-donut",       config=NO_BAR)),
            ]),

            # ── Row 5: Heatmap | NMC vs LFP ──────────────────────────────────
            html.Div(style={
                "display": "grid", "gridTemplateColumns": "1fr 1fr",
                "gap": "14px", "marginBottom": "14px",
            }, children=[
                chart_card(dcc.Graph(id="g-heatmap",      config=NO_BAR)),
                chart_card(dcc.Graph(id="g-chemistry",    config=NO_BAR)),
            ]),

            # Footer
            html.P(
                "Dataset: ev_battery_degradation_v1.csv · 10,000 EV Records · "
                "Models: Tesla Model 3, Ford Mustang Mach-E, Hyundai Ioniq 5, "
                "Wuling Air EV, BYD Atto 3 · Dashboard: Plotly Dash",
                style={"textAlign": "center", "color": "#bbb",
                       "fontSize": "11px", "paddingTop": "8px"},
            ),
        ]),
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  CALLBACK — all charts + row count update together
# ─────────────────────────────────────────────────────────────────────────────

from dash import Output, Input

@callback(
    Output("g-soh-box",    "figure"),
    Output("g-soh-cycles", "figure"),
    Output("g-soh-age",    "figure"),
    Output("g-resistance", "figure"),
    Output("g-temp-bar",   "figure"),
    Output("g-fast-charge","figure"),
    Output("g-donut",      "figure"),
    Output("g-heatmap",    "figure"),
    Output("g-chemistry",  "figure"),
    Output("row-count",    "children"),
    Input("filter-model",  "value"),
    Input("filter-style",  "value"),
    Input("filter-btype",  "value"),
    prevent_initial_call=False,
)
def update_all(models, styles, btypes):
    models = models or MODELS
    styles = styles or STYLES
    btypes = btypes or BTYPES
    dff    = filter_df(models, styles, btypes)
    n      = len(dff)
    count_text = f"Showing {n:,} / {len(df):,} vehicles"
    if n == 0:
        empty = go.Figure()
        empty.add_annotation(text="No data for current selection",
                             xref="paper", yref="paper", x=0.5, y=0.5,
                             showarrow=False, font=dict(size=14, color="gray"))
        empty.update_layout(**BASE_LAYOUT)
        return (empty,)*9 + (count_text,)
    return (
        fig_soh_box(dff),
        fig_soh_vs_cycles(dff),
        fig_soh_vs_age(dff),
        fig_resistance_soh(dff),
        fig_temp_bar(dff),
        fig_fast_charge(dff),
        fig_status_donut(dff),
        fig_heatmap(dff),
        fig_chemistry_compare(dff),
        count_text,
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
