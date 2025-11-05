# =====================================================
# IRELAND ECONOMIC POVERTY — AI-ECONOMETRIC DASHBOARD
# =====================================================

import os
import sys
import subprocess

# ---------- AUTO-INSTALL FALLBACK (for Streamlit Cloud) ----------
# This ensures missing modules like plotly, pydeck, or statsmodels are installed on runtime
def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        print(f"📦 Installing missing dependency: {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["streamlit", "pandas", "numpy", "plotly", "pydeck", "statsmodels", "openai"]:
    install_if_missing(pkg)
# -----------------------------------------------------------------

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
st.set_page_config(
    page_title="Ireland Economic Poverty AI Dashboard",
    layout="wide"
)

# OpenAI key: env var OR Streamlit secrets
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

st.title("🇮🇪 Ireland Economic Poverty Dashboard (2016–2030)")
st.caption("Upload a county-level panel CSV to unlock geospatial, econometric, and AI analysis.")

# -----------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------
st.sidebar.header("1️⃣ Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload county-level panel CSV (2016–2024)",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "⬅️ Please upload your `ie_poverty_county_panel_2016_2024.csv` "
        "or an equivalent county-level panel file to begin."
    )
    st.stop()

# Read CSV
df = pd.read_csv(uploaded_file)

# -----------------------------------------------------
# ADD COUNTY CENTROIDS FOR MAP
# -----------------------------------------------------
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

df["lat"] = df["county"].map(lambda c: centroids.get(c, [np.nan, np.nan])[0])
df["lon"] = df["county"].map(lambda c: centroids.get(c, [np.nan, np.nan])[1])

# Drop rows with missing coords if any
df = df.dropna(subset=["lat", "lon"])

years = sorted(df["year"].unique())
min_year, max_year = int(min(years)), int(max(years))

# -----------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------
st.sidebar.header("2️⃣ Controls")

selected_year = st.sidebar.slider(
    "Select Year for Map & Profiles",
    min_value=min_year,
    max_value=max_year,
    value=max_year
)
show_radar = st.sidebar.checkbox("Show Radar Chart", True)
show_pie = st.sidebar.checkbox("Show Land Use Pie", True)

filtered = df[df["year"] == selected_year]

# -----------------------------------------------------
# 🌍 OSM-BASED 3D GEOSPATIAL MAP
# -----------------------------------------------------
st.subheader(f"🗺️ OpenStreetMap 3D View — {selected_year}")

if filtered.empty:
    st.warning(f"No records found for year {selected_year}. Check your uploaded CSV.")
else:
    layer = pdk.Layer(
        "ColumnLayer",
        data=filtered,
        get_position=["lon", "lat"],
        get_elevation="economic_poverty_index * 3000",
        radius=2500,
        get_fill_color="[255 * economic_poverty_index, 120, 180 * (1 - economic_poverty_index), 200]",
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=53.4,
        longitude=-8.2,
        zoom=6.3,
        pitch=45
    )

    r = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": (
                "<b>{county}</b><br/>"
                "EPI: {economic_poverty_index:.3f}<br/>"
                "Income: €{disposable_income_eur}<br/>"
                "Deprivation: {deprivation_index_std:.2f}"
            ),
            "style": {"backgroundColor": "steelblue", "color": "white"},
        },
    )

    st.pydeck_chart(r)

# -----------------------------------------------------
# ANALYTICAL VISUALIZATIONS
# -----------------------------------------------------
st.markdown("### 📊 Interactive Visual Analytics")

col1, col2 = st.columns(2)

with col1:
    if show_radar:
        st.markdown("**Radar Chart — County Socioeconomic Profile**")
        county_choice = st.selectbox("Select County (Radar)", sorted(df["county"].unique()))
        radar_df = df[(df["county"] == county_choice) & (df["year"] == selected_year)]

        if radar_df.empty:
            st.info(f"No data for {county_choice} in {selected_year}.")
        else:
            radar_row = radar_df.iloc[0]
            radar_data = [
                radar_row["economic_poverty_index"],
                radar_row["disposable_income_eur"],
                radar_row["viirs_nl_rad_nW"],
                radar_row["cap_support_eur_per_ha"],
                radar_row["ghg_kgco2e_per_ha"],
            ]
            categories = ["EPI", "Income (€)", "Nightlights", "CAP Support", "GHG Emissions"]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_data,
                theta=categories,
                fill='toself',
                name=county_choice
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)

with col2:
    if show_pie:
        st.markdown("**Land Use Composition (Pie Chart)**")
        if filtered.empty:
            st.info("No data for selected year to plot land use.")
        else:
            pie_df = filtered.groupby("county")[["pct_urban", "pct_pasture", "pct_forest", "pct_arable"]].mean()
            county_choice_pie = st.selectbox("Select County (Pie)", sorted(pie_df.index))
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
forecast_horizon = 2030
last_hist_year = int(df["year"].max())
future_years = list(range(last_hist_year + 1, forecast_horizon + 1))

forecast_results = []
np.random.seed(42)

for county in df["county"].unique():
    sub = df[df["county"] == county].sort_values("year")
    y = sub["economic_poverty_index"].values
    years_hist = sub["year"].values

    if len(years_hist) < 2:
        continue

    coef = np.polyfit(years_hist, y, 1)
    trend = np.poly1d(coef)
    sigma = np.std(np.diff(y)) if len(y) > 1 else 0.02

    sims = []
    last_val = y[-1]
    for _ in range(n_sim):
        vals = []
        cur = last_val
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
        "epi_upper": np.percentile(sims, 97.5, axis=0),
    }))

if forecast_results:
    forecast_df = pd.concat(forecast_results)
    hist_df = df[["county", "year", "economic_poverty_index"]].rename(
        columns={"economic_poverty_index": "epi_mean"}
    )
    hist_df["epi_lower"] = hist_df["epi_mean"]
    hist_df["epi_upper"] = hist_df["epi_mean"]
    combined = pd.concat([hist_df, forecast_df])
else:
    combined = df.copy()
    combined = combined.rename(columns={"economic_poverty_index": "epi_mean"})
    combined["epi_lower"] = combined["epi_mean"]
    combined["epi_upper"] = combined["epi_mean"]

county_select = st.selectbox("Select County for Forecast", sorted(df["county"].unique()))
county_data = combined[combined["county"] == county_select].sort_values("year")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=county_data["year"],
    y=county_data["epi_mean"],
    mode="lines",
    name="Mean"
))
fig.add_trace(go.Scatter(
    x=pd.concat([county_data["year"], county_data["year"][::-1]]),
    y=pd.concat([county_data["epi_upper"], county_data["epi_lower"][::-1]]),
    fill="toself",
    fillcolor="rgba(65,105,225,0.2)",
    line=dict(color="rgba(255,255,255,0)"),
    hoverinfo="skip",
    showlegend=False
))
fig.update_layout(
    title=f"EPI Monte Carlo Forecast for {county_select} ({min_year}–2030)",
    xaxis_title="Year",
    yaxis_title="Economic Poverty Index (0–1)",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# POLICY SHOCK SIMULATION + IMPULSE RESPONSE
# -----------------------------------------------------
st.markdown("### ⚡ Policy Shock Simulation & Impulse Response")

shock_year = last_hist_year
shock_magnitude = st.slider("Policy Shock to CAP Support (%)", -20, 50, 10)
target_var = "cap_support_eur_per_ha"

panel = df.copy()
panel["lag_epi"] = panel.groupby("county")["economic_poverty_index"].shift(1)
panel = panel.dropna(subset=["lag_epi", target_var])

if panel.empty:
    st.info("Not enough data to estimate CAP–EPI elasticity.")
    beta_cap = 0.0
else:
    X = sm.add_constant(panel[target_var])
    y_epi = panel["economic_poverty_index"]
    model = sm.OLS(y_epi, X).fit()
    beta_cap = float(model.params[target_var])

st.write(f"Estimated elasticity (EPI–CAP): **{beta_cap:.4f}**")

shock_results = []
future_years_shock = list(range(shock_year, forecast_horizon + 1))

for county in df["county"].unique():
    sub = df[df["county"] == county].sort_values("year")
    base_trend = np.polyfit(sub["year"], sub["economic_poverty_index"], 1)
    trend = np.poly1d(base_trend)
    sigma = np.std(np.diff(sub["economic_poverty_index"].values)) if len(sub) > 1 else 0.02

    epi_values = [sub["economic_poverty_index"].iloc[-1]]

    for year in future_years_shock[1:]:
        drift = trend(year) - trend(year - 1)
        if year == shock_year:
            shock_effect = beta_cap * (shock_magnitude / 100)
        else:
            decay = np.exp(-0.5 * (year - shock_year))
            shock_effect = beta_cap * (shock_magnitude / 100) * decay

        noise = np.random.normal(0, sigma * 0.3)
        next_epi = np.clip(epi_values[-1] + drift + shock_effect + noise, 0, 1)
        epi_values.append(next_epi)

    shock_results.append(pd.DataFrame({
        "year": future_years_shock,
        "epi": epi_values,
        "county": county
    }))

shock_df = pd.concat(shock_results)

county_irf = st.selectbox("Select County for Impulse Response", sorted(df["county"].unique()))
irf_base = combined[combined["county"] == county_irf].sort_values("year")
irf_shock = shock_df[shock_df["county"] == county_irf].sort_values("year")

fig_irf = go.Figure()
fig_irf.add_trace(go.Scatter(
    x=irf_base["year"],
    y=irf_base["epi_mean"],
    name="Baseline Forecast",
    line=dict(color="royalblue")
))
fig_irf.add_trace(go.Scatter(
    x=irf_shock["year"],
    y=irf_shock["epi"],
    name=f"Post-Shock ({shock_magnitude}%)",
    line=dict(color="firebrick", dash="dot")
))
fig_irf.update_layout(
    title=f"Impulse Response — {county_irf} (Shock in {shock_year})",
    xaxis_title="Year",
    yaxis_title="EPI (0–1)",
    template="plotly_white"
)
st.plotly_chart(fig_irf, use_container_width=True)

# -----------------------------------------------------
# CONVERGENCE DIAGNOSTICS (σ AND β)
# -----------------------------------------------------
st.markdown("### 📈 Convergence Diagnostics — County-Level Inequality Dynamics")

sigma_trend = combined.groupby("year")["epi_mean"].std().reset_index(name="sigma_epi")
fig_sigma = px.line(
    sigma_trend,
    x="year",
    y="sigma_epi",
    title="σ-Convergence: Dispersion of EPI Across Counties (Historical + Forecast)",
    labels={"sigma_epi": "Std Dev of EPI"}
)
st.plotly_chart(fig_sigma, use_container_width=True)

base_year = int(df["year"].min())
target_year = forecast_horizon

growth_rows = []
for county in df["county"].unique():
    d = combined[combined["county"] == county]
    if base_year in d["year"].values and target_year in d["year"].values:
        epi_0 = d.loc[d["year"] == base_year, "epi_mean"].values[0]
        epi_T = d.loc[d["year"] == target_year, "epi_mean"].values[0]
        if epi_0 != 0:
            growth = (epi_T - epi_0) / epi_0
            growth_rows.append((county, epi_0, growth))

growth_data = pd.DataFrame(growth_rows, columns=["county", "initial_epi", "growth_rate"])

if not growth_data.empty:
    X_beta = sm.add_constant(growth_data["initial_epi"])
    y_beta = growth_data["growth_rate"]
    beta_model = sm.OLS(y_beta, X_beta).fit()
    beta_coef = float(beta_model.params["initial_epi"])
else:
    beta_coef = 0.0

st.write(
    f"**β-Convergence Coefficient (between {base_year} and {target_year}):** "
    f"{beta_coef:.4f} — "
    f"{'Convergence (poorer counties improving faster)' if beta_coef < 0 else 'Divergence (poorer counties improving slower)'}"
)

if not growth_data.empty:
    fig_beta = px.scatter(
        growth_data,
        x="initial_epi",
        y="growth_rate",
        trendline="ols",
        title=f"β-Convergence: Growth vs Initial EPI ({base_year}–{target_year})",
        labels={
            "initial_epi": f"Initial EPI ({base_year})",
            "growth_rate": f"EPI Growth {base_year}–{target_year}"
        }
    )
    st.plotly_chart(fig_beta, use_container_width=True)

# -----------------------------------------------------
# AGENTIC AI INTERPRETATION
# -----------------------------------------------------
st.markdown("### 🤖 Agentic AI Reasoner — Policy & Convergence Interpretation")

if not openai.api_key:
    st.info("OpenAI API key not set. Add it as OPENAI_API_KEY env var or in Streamlit secrets to enable AI reasoning.")
else:
    if st.button("Run AI Interpretation"):
        sigma_2030 = sigma_trend.loc[sigma_trend["year"] == target_year, "sigma_epi"]
        sigma_2030_val = float(sigma_2030.values[0]) if not sigma_2030.empty else float("nan")

        summary = (
            f"Using a panel of Irish counties with an Economic Poverty Index (EPI), "
            f"we forecasted poverty to {target_year} using Monte Carlo simulation and a linear trend. "
            f"The σ-convergence (standard deviation of EPI across counties) in {target_year} "
            f"is approximately {sigma_2030_val:.3f}. The β-convergence coefficient between "
            f"{base_year} and {target_year} is {beta_coef:.4f}. "
            f"A policy shock of {shock_magnitude}% in CAP support has an estimated elasticity "
            f"of β_CAP={beta_cap:.4f} on EPI, with an impulse response that decays over time. "
            f"Please interpret these findings in terms of regional poverty convergence/divergence, "
            f"policy implications for CAP and cohesion policy, and alignment with SDG10 (Reduced Inequalities)."
        )

        with st.spinner("AI reasoning..."):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an econometric AI policy analyst for Ireland, "
                                "specialized in poverty, regional convergence, and CAP policy."
                            )
                        },
                        {"role": "user", "content": summary}
                    ],
                    temperature=0.55,
                    max_tokens=450,
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
st.caption("Developed by Jit — Powered by Econometrics × AI × Policy Intelligence | Version 3.0 (2025)")
