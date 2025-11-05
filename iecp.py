# =====================================================
# IRELAND ECONOMIC POVERTY — AI-ECONOMETRIC DASHBOARD
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import statsmodels.api as sm
import openai

# -----------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------
st.set_page_config(page_title="Ireland Economic Poverty AI Dashboard", layout="wide")

openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

st.title("🇮🇪 Ireland Economic Poverty Dashboard (2016–2030)")
st.caption("Powered by Econometrics × AI × Geospatial Intelligence")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/jit/Downloads/ie_poverty_county_panel_2016_2024.csv")
    centroids = {
        "Carlow": [52.73, -6.64], "Cavan": [54.00, -7.36], "Clare": [52.84, -9.00],
        "Cork": [51.90, -8.47], "Donegal": [54.95, -7.73], "Dublin": [53.35, -6.26],
        "Galway": [53.27, -9.06], "Kerry": [52.16, -9.56], "Kildare": [53.16, -6.91],
        "Kilkenny": [52.65, -7.25], "Laois": [53.03, -7.30], "Leitrim": [54.11, -8.00],
        "Limerick": [52.66, -8.63], "Longford": [53.72, -7.80], "Louth": [53.95, -6.54],
        "Mayo": [53.95, -9.31], "Meath": [53.60, -6.65], "Monaghan": [54.25, -6.97],
        "Offaly": [53.26, -7.50], "Roscommon": [53.63, -8.19], "Sligo": [54.27, -8.47],
        "Tipperary": [52.68, -7.88], "Waterford": [52.26, -7.11], "Westmeath": [53.53, -7.36],
        "Wexford": [52.34, -6.46], "Wicklow": [52.98, -6.36]
    }
    df["lat"] = df["county"].map(lambda x: centroids[x][0])
    df["lon"] = df["county"].map(lambda x: centroids[x][1])
    return df

df = load_data()
years = sorted(df["year"].unique())

# -----------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------
st.sidebar.header("Controls")
selected_year = st.sidebar.slider("Select Year", int(min(years)), int(max(years)), int(max(years)))
show_radar = st.sidebar.checkbox("Show Radar Chart", True)
show_pie = st.sidebar.checkbox("Show Land Use Pie", True)

filtered = df[df["year"] == selected_year]

# -----------------------------------------------------
# 3D GEOSPATIAL MAP
# -----------------------------------------------------
st.subheader(f"🗺️ 3D Geospatial View — {selected_year}")

layer = pdk.Layer(
    "ColumnLayer",
    data=filtered,
    get_position=["lon", "lat"],
    get_elevation="economic_poverty_index * 3000",
    elevation_scale=50,
    radius=2000,
    get_fill_color="[255 * economic_poverty_index, 100, 180 * (1-economic_poverty_index), 200]",
    pickable=True,
    auto_highlight=True,
)
tooltip = {
    "html": "<b>{county}</b><br/>EPI: {economic_poverty_index:.3f}<br/>Income: €{disposable_income_eur}<br/>Deprivation: {deprivation_index_std:.2f}",
    "style": {"backgroundColor": "steelblue", "color": "white"},
}
view_state = pdk.ViewState(latitude=53.4, longitude=-8.2, zoom=6.5, pitch=45)
r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
st.pydeck_chart(r)

# -----------------------------------------------------
# ANALYTICAL VISUALIZATIONS
# -----------------------------------------------------
st.markdown("### 📊 Interactive Visual Analytics")
col1, col2 = st.columns(2)

with col1:
    if show_radar:
        st.markdown("**Radar Chart — County Socioeconomic Profile**")
        county_choice = st.selectbox("Select County", sorted(df["county"].unique()))
        radar_df = df[(df["county"] == county_choice) & (df["year"] == selected_year)]
        if not radar_df.empty:
            radar_data = radar_df[
                ["economic_poverty_index", "disposable_income_eur", "viirs_nl_rad_nW",
                 "cap_support_eur_per_ha", "ghg_kgco2e_per_ha"]
            ].values.flatten()
            categories = ["EPI", "Income (€)", "Nightlights", "CAP Support", "GHG Emissions"]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_data, theta=categories, fill='toself', name=county_choice))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
            st.plotly_chart(fig_radar, use_container_width=True)

with col2:
    if show_pie:
        st.markdown("**Land Use Composition (Pie Chart)**")
        pie_df = filtered.groupby("county")[["pct_urban", "pct_pasture", "pct_forest", "pct_arable"]].mean()
        county_choice_pie = st.selectbox("Select County (Pie)", sorted(filtered["county"].unique()))
        values = pie_df.loc[county_choice_pie].values
        fig_pie = px.pie(
            names=["Urban", "Pasture", "Forest", "Arable"],
            values=values,
            title=f"Land Use Composition — {county_choice_pie} ({selected_year})"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------------------------------
# MONTE CARLO FORECASTING TO 2030
# -----------------------------------------------------
st.markdown("### 🔮 Monte Carlo Forecasting — Economic Poverty Index till 2030")

n_sim = 1000
future_years = list(range(2025, 2031))
forecast_results = []
np.random.seed(42)

for county in df["county"].unique():
    sub = df[df["county"] == county].sort_values("year")
    y = sub["economic_poverty_index"].values
    years_hist = sub["year"].values
    coef = np.polyfit(years_hist, y, 1)
    trend = np.poly1d(coef)
    sigma = np.std(np.diff(y))
    sims = []
    last_val = y[-1]
    for _ in range(n_sim):
        vals, cur = [], last_val
        for yr in future_years:
            noise = np.random.normal(0, sigma)
            drift = trend(yr) - trend(yr - 1)
            cur = np.clip(cur + drift + noise, 0, 1)
            vals.append(cur)
        sims.append(vals)
    sims = np.array(sims)
    forecast_results.append(pd.DataFrame({
        "county": county,
        "year": future_years,
        "epi_mean": sims.mean(axis=0),
        "epi_lower": np.percentile(sims, 2.5, axis=0),
        "epi_upper": np.percentile(sims, 97.5, axis=0)
    }))

forecast_df = pd.concat(forecast_results)
hist_df = df[["county", "year", "economic_poverty_index"]].rename(columns={"economic_poverty_index": "epi_mean"})
hist_df["epi_lower"] = hist_df["epi_mean"]
hist_df["epi_upper"] = hist_df["epi_mean"]
combined = pd.concat([hist_df, forecast_df])

county_select = st.selectbox("Select County for Forecast", sorted(df["county"].unique()))
county_data = combined[combined["county"] == county_select]
fig = go.Figure()
fig.add_trace(go.Scatter(x=county_data["year"], y=county_data["epi_mean"], mode="lines", name="Mean"))
fig.add_trace(go.Scatter(
    x=pd.concat([county_data["year"], county_data["year"][::-1]]),
    y=pd.concat([county_data["epi_upper"], county_data["epi_lower"][::-1]]),
    fill="toself", fillcolor="rgba(65,105,225,0.2)", line=dict(color="rgba(255,255,255,0)"),
    hoverinfo="skip", showlegend=False))
fig.update_layout(title=f"EPI Monte Carlo Forecast for {county_select} (2016–2030)",
                  xaxis_title="Year", yaxis_title="Economic Poverty Index (0–1)",
                  template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# POLICY SHOCK SIMULATION + IMPULSE RESPONSE
# -----------------------------------------------------
st.markdown("### ⚡ Policy Shock Simulation & Impulse Response")
shock_year = 2024
shock_magnitude = st.slider("Policy Shock to CAP Support (%)", -20, 50, 10)
target_var = "cap_support_eur_per_ha"

panel = df.copy()
panel["lag_epi"] = panel.groupby("county")["economic_poverty_index"].shift(1)
panel.dropna(inplace=True)
X = sm.add_constant(panel[target_var])
y = panel["economic_poverty_index"]
model = sm.OLS(y, X).fit()
beta_cap = model.params[target_var]

shock_results = []
for county in df["county"].unique():
    sub = df[df["county"] == county].sort_values("year")
    base_trend = np.polyfit(sub["year"], sub["economic_poverty_index"], 1)
    trend = np.poly1d(base_trend)
    sigma = np.std(np.diff(sub["economic_poverty_index"].values))
    epi_values = [sub["economic_poverty_index"].iloc[-1]]
    for year in range(2025, 2031):
        drift = trend(year) - trend(year - 1)
        decay = np.exp(-0.5 * (year - shock_year))
        shock_effect = beta_cap * (shock_magnitude / 100) * decay
        noise = np.random.normal(0, sigma * 0.3)
        next_epi = np.clip(epi_values[-1] + drift + shock_effect + noise, 0, 1)
        epi_values.append(next_epi)
    shock_results.append(pd.DataFrame({"year": list(range(2024, 2031)), "epi": epi_values, "county": county}))
shock_df = pd.concat(shock_results)

county_irf = st.selectbox("Select County for Impulse Response", sorted(df["county"].unique()))
irf_base = combined[combined["county"] == county_irf]
irf_shock = shock_df[shock_df["county"] == county_irf]
fig_irf = go.Figure()
fig_irf.add_trace(go.Scatter(x=irf_base["year"], y=irf_base["epi_mean"], name="Baseline", line=dict(color="royalblue")))
fig_irf.add_trace(go.Scatter(x=irf_shock["year"], y=irf_shock["epi"], name="Post-Shock", line=dict(color="firebrick", dash="dot")))
fig_irf.update_layout(title=f"Impulse Response — {county_irf}", xaxis_title="Year", yaxis_title="EPI (0–1)", template="plotly_white")
st.plotly_chart(fig_irf, use_container_width=True)

# -----------------------------------------------------
# CONVERGENCE DIAGNOSTICS (σ AND β)
# -----------------------------------------------------
st.markdown("### 📈 Convergence Diagnostics — County-Level Inequality Dynamics")

sigma_trend = combined.groupby("year")["epi_mean"].std().reset_index(name="sigma_epi")
fig_sigma = px.line(sigma_trend, x="year", y="sigma_epi",
                    title="σ-Convergence: Dispersion of EPI Across Counties (2016–2030)",
                    labels={"sigma_epi": "Std Dev of EPI"})
st.plotly_chart(fig_sigma, use_container_width=True)

growth_data = combined.groupby("county").apply(
    lambda d: (d.loc[d["year"] == 2030, "epi_mean"].values[0] -
               d.loc[d["year"] == 2016, "epi_mean"].values[0]) /
              d.loc[d["year"] == 2016, "epi_mean"].values[0]
).reset_index(name="growth_rate")
growth_data["initial_epi"] = [
    combined[(combined["county"] == c) & (combined["year"] == 2016)]["epi_mean"].values[0]
    for c in growth_data["county"]
]
X_beta = sm.add_constant(growth_data["initial_epi"])
y_beta = growth_data["growth_rate"]
beta_model = sm.OLS(y_beta, X_beta).fit()
beta_coef = beta_model.params["initial_epi"]

st.write(f"**β-Convergence Coefficient:** {beta_coef:.4f} ({'Convergence' if beta_coef < 0 else 'Divergence'})")

fig_beta = px.scatter(growth_data, x="initial_epi", y="growth_rate",
                      trendline="ols",
                      title="β-Convergence: Growth vs Initial EPI (2016–2030)",
                      labels={"initial_epi": "Initial EPI (2016)", "growth_rate": "EPI Growth"})
st.plotly_chart(fig_beta, use_container_width=True)

# -----------------------------------------------------
# AGENTIC AI INTERPRETATION
# -----------------------------------------------------
st.markdown("### 🤖 Agentic AI Reasoner — Policy & Convergence Interpretation")

if st.button("Run AI Interpretation"):
    summary = (
        f"By 2030, σ-convergence dispersion is {sigma_trend.loc[sigma_trend['year']==2030,'sigma_epi'].values[0]:.3f}, "
        f"and β-convergence coefficient is {beta_coef:.4f}. "
        f"A policy shock of {shock_magnitude}% in CAP support yields elasticity β={beta_cap:.4f}. "
        f"Interpret Ireland's regional poverty convergence/divergence patterns, policy implications, "
        f"and SDG10 alignment based on these econometric results."
    )
    with st.spinner("AI reasoning..."):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an econometric AI policy analyst for Ireland's poverty and regional convergence."},
                    {"role": "user", "content": summary}
                ],
                temperature=0.55,
                max_tokens=400
            )
            answer = response.choices[0].message["content"]
            st.success("AI Interpretation:")
            st.markdown(answer)
        except Exception as e:
            st.error(f"AI request failed: {e}")

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------
st.markdown("---")
st.caption("Developed by Jit — Powered by Econometrics × AI × Policy Intelligence | Version 2.0 (2025)")
