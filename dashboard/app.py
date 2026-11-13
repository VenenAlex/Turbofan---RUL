"""Turbofan fleet dashboard: a Streamlit client of the RUL service.

The dashboard never loads the model. Every prediction, explanation and drift check
comes from the FastAPI service over HTTP, exactly as any other client (a maintenance
system, a mobile app) would get them.

Run from the project root, with the API already running::

    streamlit run dashboard/app.py
"""

import os

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from turbofan_rul.config import COLUMNS, MAINTENANCE_AT, RUL_CAP, SENSOR_NAMES, SENSORS, WINDOW
from turbofan_rul.data import load_cycles, load_test_rul, read_cycles
from turbofan_rul.evaluate import health_band

API_URL = os.getenv("API_URL", "http://localhost:8000")
LINKS = {
    "API docs": os.getenv("PUBLIC_API_URL", "http://localhost:8000") + "/docs",
    "Grafana": os.getenv("GRAFANA_URL", "http://localhost:3000"),
    "MLflow": os.getenv("MLFLOW_URL", "http://localhost:5000"),
}

# Series colours, plus status colours reserved for the three health bands.
BLUE, ORANGE, GRAY, RED = "#3987e5", "#d95926", "#898781", "#e66767"
BAND_COLORS = {"Critical": "#d03b3b", "Warning": "#fab219", "Healthy": "#0ca30c"}
BAND_ICONS = {"Critical": "🔴", "Warning": "🟠", "Healthy": "🟢"}
BADGE_COLORS = {"Critical": "red", "Warning": "orange", "Healthy": "green"}
DRIFT_COLORS = {"stable": "#0ca30c", "moderate": "#fab219", "significant": "#d03b3b"}

TABS = ["Fleet overview", "Engine inspector", "Maintenance planner", "Model card", "Drift monitor", "About"]
CHART = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

st.set_page_config(page_title="Turbofan RUL", page_icon="✈️", layout="wide")


# ---------------------------------------------------------------- service calls
def api(method: str, path: str, payload: dict | None = None) -> dict:
    """Call the RUL service.

    Parameters
    ----------
    method : str
        ``"GET"`` or ``"POST"``.
    path : str
        Endpoint path, such as ``/api/v1/predictions``.
    payload : dict, optional
        JSON body for POST requests.

    Returns
    -------
    dict
        The decoded JSON answer.

    Raises
    ------
    RuntimeError
        With a readable message if the service is down or rejects the request.
    """
    try:
        response = httpx.request(method, f"{API_URL}{path}", json=payload, timeout=120)
        response.raise_for_status()
    except httpx.HTTPStatusError as err:
        raise RuntimeError(f"The RUL service rejected the request: {err.response.text}") from err
    except httpx.HTTPError as err:
        raise RuntimeError(
            f"Cannot reach the RUL service at {API_URL} ({err}). Start it with "
            "`uvicorn turbofan_rul.api.main:app` or `docker compose up`."
        ) from err
    return response.json()


def to_readings(cycles: pd.DataFrame) -> list[dict]:
    """Convert NASA-format rows into the API's reading objects.

    Parameters
    ----------
    cycles : pd.DataFrame
        Rows with the 26 columns in ``COLUMNS``.

    Returns
    -------
    list of dict
        One ``{"unit", "cycle", "settings", "sensors"}`` object per row.
    """
    values = cycles[COLUMNS].to_numpy(dtype=float)
    return [
        {"unit": int(r[0]), "cycle": int(r[1]), "settings": r[2:5].tolist(), "sensors": r[5:].tolist()}
        for r in values
    ]


@st.cache_data(show_spinner=False)
def test_fleet() -> tuple[pd.DataFrame, pd.Series]:
    """The 100 NASA test engines and their true remaining cycles (the answer key)."""
    actual = load_test_rul()
    return load_cycles("test"), pd.Series(actual, index=range(1, len(actual) + 1), name="actual_rul")


@st.cache_data(ttl=60, show_spinner=False)
def model_card() -> dict:
    """Fetch the served model's card from ``GET /api/v1/models/production``."""
    return api("GET", "/api/v1/models/production")


@st.cache_data(ttl=600, show_spinner="Scoring 100 engines through the API...")
def score_fleet() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Score every cycle of every test engine in one API call.

    Returns
    -------
    engines : pd.DataFrame
        One row per engine: last cycle, predicted and actual RUL, error, band.
    history : pd.DataFrame
        Predicted and actual RUL at every cycle of every engine.
    contributions : pd.DataFrame
        SHAP value of each feature at each engine's last cycle.
    base_value : float
        The model's average prediction, where every SHAP breakdown starts.
    """
    cycles, actual = test_fleet()
    body = api(
        "POST",
        "/api/v1/predictions",
        {"readings": to_readings(cycles), "explain": True, "include_history": True},
    )
    rows, history, contributions = [], [], {}
    for e in body["engines"]:
        rows.append({k: e[k] for k in ("unit", "last_cycle", "predicted_rul", "health_band")})
        history += [{"unit": e["unit"], **h} for h in e["history"]]
        contributions[e["unit"]] = {c["feature"]: c["value"] for c in e["explanation"]["contributions"]}
    engines = pd.DataFrame(rows).set_index("unit").rename(columns={"health_band": "band"})
    engines["actual_rul"] = actual
    engines["error"] = engines["predicted_rul"] - engines["actual_rul"]

    history = pd.DataFrame(history)
    last = history.groupby("unit")["cycle"].transform("max")
    # The training target: true cycles left, capped at RUL_CAP.
    history["actual_rul"] = (history["unit"].map(actual) + last - history["cycle"]).clip(upper=RUL_CAP)
    base_value = body["engines"][0]["explanation"]["base_value"]
    return engines, history, pd.DataFrame(contributions).T, base_value


@st.cache_data(ttl=600, show_spinner=False)
def explain_at(unit: int, cycle: int) -> tuple[float, pd.Series, float]:
    """Ask the API to explain one engine as it looked at an earlier cycle.

    Features only look backwards, so sending the readings up to ``cycle`` gives
    exactly the prediction the model would have made on that day.

    Parameters
    ----------
    unit : int
        Engine id.
    cycle : int
        Last cycle to send.

    Returns
    -------
    tuple
        Predicted RUL, SHAP value per feature, and the base value.
    """
    cycles, _ = test_fleet()
    part = cycles[(cycles["unit"] == unit) & (cycles["cycle"] <= cycle)]
    engine = api("POST", "/api/v1/predictions", {"readings": to_readings(part), "explain": True})["engines"][
        0
    ]
    contributions = pd.Series({c["feature"]: c["value"] for c in engine["explanation"]["contributions"]})
    return engine["predicted_rul"], contributions, engine["explanation"]["base_value"]


@st.cache_data(ttl=600, show_spinner="Checking drift...")
def drift_check(cycles: pd.DataFrame) -> dict:
    """``POST /api/v1/drift-reports`` for a batch of readings."""
    return api("POST", "/api/v1/drift-reports", {"readings": to_readings(cycles)})


# ---------------------------------------------------------------- helpers
def pretty(feature: str) -> str:
    """Readable name for a model feature, such as ``s15_mean``."""
    sensor, _, stat = feature.partition("_")
    label = f"{SENSOR_NAMES[sensor][1]} ({sensor})"
    return {"": label, "mean": f"{label}, {WINDOW}-cycle mean", "std": f"{label}, {WINDOW}-cycle spread"}[
        stat
    ]


def sensor_label(sensor: str) -> str:
    """``s4 · LPT outlet temperature``."""
    return f"{sensor} · {SENSOR_NAMES[sensor][1]}"


def finish(fig: go.Figure, height: int) -> go.Figure:
    """Shared layout: fixed height, slim margins, legend above the plot."""
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig


def gauge(value: float, threshold: float) -> go.Figure:
    """Dial of predicted RUL with the three health bands and the alert threshold."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": " cycles", "valueformat": ".0f", "font": {"size": 30}},
            gauge={
                "axis": {"range": [0, 150], "tickvals": [0, 30, 60, 90, 125, 150]},
                "bar": {"color": BLUE, "thickness": 0.3},
                "steps": [
                    {"range": [0, 30], "color": "rgba(208, 59, 59, 0.35)"},
                    {"range": [30, 60], "color": "rgba(250, 178, 25, 0.35)"},
                    {"range": [60, 150], "color": "rgba(12, 163, 12, 0.25)"},
                ],
                "threshold": {"line": {"color": GRAY, "width": 3}, "thickness": 0.9, "value": threshold},
            },
        )
    )
    fig.update_layout(height=220, margin={"l": 30, "r": 30, "t": 20, "b": 0})
    return fig


def fleet_map(engines: pd.DataFrame, selected: int) -> go.Figure:
    """10 x 10 grid of engine tiles coloured by health band; click a tile to select it."""
    fig = go.Figure()
    for band, color in BAND_COLORS.items():
        d = engines[engines["band"] == band]
        units = d.index.to_numpy()
        fig.add_scatter(
            x=(units - 1) % 10,
            y=(units - 1) // 10,
            mode="markers+text",
            name=f"{BAND_ICONS[band]} {band}",
            marker={"symbol": "square", "size": 34, "color": color},
            text=units.astype(str),
            textfont={"color": "#0b0b0b" if band == "Warning" else "#ffffff", "size": 11},
            customdata=np.c_[units, d["predicted_rul"], d["actual_rul"]],
            hovertemplate="<b>Engine %{customdata[0]}</b><br>Predicted RUL %{customdata[1]:.0f} cycles"
            "<br>Actual RUL %{customdata[2]} cycles<extra></extra>",
            selected={"marker": {"opacity": 1}},
            unselected={"marker": {"opacity": 1}},
        )
    fig.add_scatter(  # ring around the engine being inspected
        x=[(selected - 1) % 10],
        y=[(selected - 1) // 10],
        mode="markers",
        marker={"symbol": "square-open", "size": 44, "color": BLUE, "line": {"width": 3}},
        hoverinfo="skip",
        showlegend=False,
    )
    fig.update_xaxes(visible=False, range=[-0.6, 9.6])
    fig.update_yaxes(visible=False, range=[9.6, -0.6])
    fig.update_layout(dragmode=False, clickmode="event+select")
    return finish(fig, 440)


def pick_engine() -> None:
    """Fleet-map click: select that engine and jump to the inspector."""
    points = st.session_state["fleet_map"]["selection"]["points"]
    if points:
        st.session_state["engine"] = int(round(points[0]["y"])) * 10 + int(round(points[0]["x"])) + 1
        st.session_state["view"] = TABS[1]


def waterfall(base: float, contributions: pd.Series, top: int = 10) -> go.Figure:
    """SHAP waterfall: from the average engine to this prediction, biggest effects first."""
    order = contributions.abs().sort_values(ascending=False).index
    shown, rest = contributions[order[:top]], contributions[order[top:]].sum()
    labels = ["Average engine", *[pretty(f) for f in shown.index], f"{len(order) - top} other features"]
    values = [base, *shown.values, rest]
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["absolute"] + ["relative"] * (len(values) - 1) + ["total"],
            y=[*labels, "Prediction"],
            x=[*values, None],
            text=[f"{base:.1f}", *[f"{v:+.1f}" for v in values[1:]], f"{sum(values):.1f}"],
            textposition="outside",
            increasing={"marker": {"color": BLUE}},
            decreasing={"marker": {"color": RED}},
            totals={"marker": {"color": GRAY}},
            connector={"line": {"color": GRAY, "width": 1}},
            hovertemplate="%{y}: %{text}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Remaining cycles")
    return finish(fig, 470)


def alert_sweep(engines: pd.DataFrame, cost_ratio: float) -> pd.DataFrame:
    """Missed failures, false alarms and cost for every alert threshold from 0 to 125."""
    due = engines["actual_rul"] <= MAINTENANCE_AT
    rows = []
    for t in range(0, RUL_CAP + 1):
        flagged = engines["predicted_rul"] <= t
        missed, false = int((due & ~flagged).sum()), int((flagged & ~due).sum())
        rows.append(
            {
                "threshold": t,
                "caught": int((due & flagged).sum()),
                "missed": missed,
                "false_alarms": false,
                "cost": missed * cost_ratio + false,
            }
        )
    return pd.DataFrame(rows).set_index("threshold")


# ---------------------------------------------------------------- data
try:
    health = api("GET", "/api/v1/health")
    card = model_card()
    engines, history, contributions, base_value = score_fleet()
except RuntimeError as err:
    st.error(str(err), icon=":material/cloud_off:")
    st.stop()
cycles, _ = test_fleet()

# ---------------------------------------------------------------- sidebar
if "engine" not in st.session_state:
    st.session_state["engine"] = int(engines["predicted_rul"].idxmin())  # most urgent engine first

with st.sidebar:
    st.markdown("## ✈️ Turbofan RUL")
    st.caption("Predictive maintenance for 100 NASA C-MAPSS turbofan engines")
    st.success(f"Service online · model v{health['model_version']}", icon=":material/check_circle:")
    engine = st.selectbox(
        "Engine",
        engines.index.tolist(),
        key="engine",
        format_func=lambda u: f"Engine {u} · {BAND_ICONS[engines.at[u, 'band']]} {engines.at[u, 'band']}",
    )
    threshold = st.slider(
        "Alert when predicted RUL ≤",
        10,
        100,
        int(card["policy"]["alert_threshold"]),
        5,
        help="Engines at or below this many predicted cycles get a maintenance alert.",
    )
    st.divider()
    st.caption("Behind the dashboard")
    for name, url in LINKS.items():
        st.link_button(name, url, width="stretch")

st.title("Turbofan engine health monitor")
st.caption(
    f"XGBoost model v{card['version']} served by FastAPI · trained on {card['data']['train_engines']} "
    f"run-to-failure engines · tracked in MLflow · monitored with Prometheus + Grafana"
)
engines["alert"] = engines["predicted_rul"] <= threshold
tabs = st.tabs(TABS, key="view", on_change="rerun")

# ---------------------------------------------------------------- fleet overview
with tabs[0]:
    counts = engines["band"].value_counts()
    k = st.columns(5)
    k[0].metric("Engines monitored", len(engines), border=True)
    k[1].metric("🔴 Critical", int(counts.get("Critical", 0)), help="Predicted RUL ≤ 30 cycles", border=True)
    k[2].metric("🟠 Warning", int(counts.get("Warning", 0)), help="Predicted RUL 31-60 cycles", border=True)
    k[3].metric("🟢 Healthy", int(counts.get("Healthy", 0)), help="Predicted RUL > 60 cycles", border=True)
    k[4].metric(
        "Maintenance alerts", int(engines["alert"].sum()), help=f"Predicted RUL ≤ {threshold}", border=True
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Fleet health map")
        st.caption("Each tile is one engine, coloured by predicted health. **Click a tile** to inspect it.")
        st.plotly_chart(
            fleet_map(engines, engine),
            key="fleet_map",
            on_select=pick_engine,
            selection_mode="points",
            config={"displayModeBar": False},
        )
    with right:
        e = engines.loc[engine]
        st.subheader(f"Engine {engine}")
        st.badge(e["band"], color=BADGE_COLORS[e["band"]])
        st.plotly_chart(
            gauge(e["predicted_rul"], threshold), key="fleet_gauge", config={"displayModeBar": False}
        )
        c = st.columns(3)
        c[0].metric("Predicted", f"{e['predicted_rul']:.0f}")
        c[1].metric("Actual", int(e["actual_rul"]), help="NASA answer key; unknown in real operation")
        c[2].metric("Error", f"{e['error']:+.0f}", help="Predicted minus actual; positive is optimistic")
        st.caption("Open **Engine inspector** for this engine's full history and the reasons behind it.")

    st.subheader("Maintenance queue")
    queue = engines.sort_values("predicted_rul").reset_index()
    queue["status"] = queue["band"].map(lambda b: f"{BAND_ICONS[b]} {b}")
    st.dataframe(
        queue[["unit", "status", "predicted_rul", "actual_rul", "error", "last_cycle", "alert"]],
        hide_index=True,
        height=330,
        column_config={
            "unit": st.column_config.NumberColumn("Engine", format="%d"),
            "status": "Health",
            "predicted_rul": st.column_config.ProgressColumn(
                "Predicted RUL", min_value=0, max_value=RUL_CAP, format="%.0f cycles"
            ),
            "actual_rul": st.column_config.NumberColumn("Actual RUL", help="NASA answer key"),
            "error": st.column_config.NumberColumn("Error", format="%+.0f", help="Predicted − actual"),
            "last_cycle": st.column_config.NumberColumn("Cycles flown"),
            "alert": st.column_config.CheckboxColumn("Alert"),
        },
    )

# ---------------------------------------------------------------- engine inspector
with tabs[1]:
    e = engines.loc[engine]
    h = history[history["unit"] == engine].set_index("cycle")
    last_cycle = int(e["last_cycle"])
    st.subheader(f"Engine {engine}: {last_cycle} cycles flown")
    cycle = st.slider(
        "Replay this engine's life: see what the model said at any past cycle",
        1,
        last_cycle,
        last_cycle,
        key=f"cycle_{engine}",
    )
    predicted, actual = h.at[cycle, "predicted_rul"], int(e["actual_rul"]) + last_cycle - cycle
    band = str(health_band([predicted])[0])

    top_left, top_right = st.columns([2, 3], gap="large")
    with top_left:
        st.plotly_chart(gauge(predicted, threshold), key="inspector_gauge", config={"displayModeBar": False})
    with top_right:
        st.badge(band, color=BADGE_COLORS[band])
        if predicted <= threshold:
            st.badge("Maintenance alert", icon=":material/notifications_active:", color="red")
        c = st.columns(3)
        c[0].metric(f"Predicted at cycle {cycle}", f"{predicted:.0f} cycles")
        c[1].metric("Actual", f"{actual} cycles", help="NASA answer key; unknown in real operation")
        c[2].metric("Error", f"{predicted - actual:+.0f}", help="Positive = optimistic (the dangerous side)")
        if actual > RUL_CAP:
            st.info(
                f"This engine still had {actual} cycles left. The model was trained with RUL capped at "
                f"{RUL_CAP}, so early in life it only says 'healthy, about {RUL_CAP}+'. That is by design.",
                icon=":material/info:",
            )
        elif predicted - actual > 30:
            st.warning(
                f"The model is {predicted - actual:.0f} cycles too optimistic here, the costly kind of miss. "
                "The breakdown below shows which sensors misled it.",
                icon=":material/warning:",
            )

    st.markdown("#### Predicted vs actual remaining life")
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=MAINTENANCE_AT, fillcolor=BAND_COLORS["Critical"], opacity=0.08, line_width=0)
    fig.add_scatter(
        x=h.index,
        y=h["actual_rul"],
        name=f"Actual RUL (capped at {RUL_CAP}, the training target)",
        line={"color": GRAY, "width": 2, "dash": "dot"},
    )
    fig.add_scatter(x=h.index, y=h["predicted_rul"], name="Predicted RUL", line={"color": BLUE, "width": 2})
    fig.add_hline(
        y=threshold,
        line={"color": GRAY, "width": 1, "dash": "dash"},
        annotation_text=f"alert ≤ {threshold}",
        annotation_position="top left",
    )
    fig.add_vline(x=cycle, line={"color": BLUE, "width": 1})
    fig.update_layout(hovermode="x unified", yaxis_title="Remaining cycles", xaxis_title="Cycle (flight)")
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(finish(fig, 380), config=CHART)

    why, sensors_col = st.columns(2, gap="large")
    with why:
        st.markdown(f"#### Why {predicted:.0f} cycles? (SHAP, at cycle {cycle})")
        try:
            pred_at, contrib_at, base_at = explain_at(engine, cycle)
        except RuntimeError as err:
            st.error(str(err))
        else:
            down, up = contrib_at.idxmin(), contrib_at.idxmax()
            st.caption(
                f"Starts from the average engine ({base_at:.0f} cycles). Blue bars add life, red bars "
                f"remove it. Biggest pull down: **{pretty(down)}** ({contrib_at[down]:+.1f}); biggest push "
                f"up: **{pretty(up)}** ({contrib_at[up]:+.1f})."
            )
            st.plotly_chart(waterfall(base_at, contrib_at), config=CHART)
    with sensors_col:
        st.markdown("#### What the sensors show")
        chosen = st.pills(
            "Sensors",
            SENSORS,
            selection_mode="multi",
            default=["s4", "s11", "s15"],
            format_func=sensor_label,
            label_visibility="collapsed",
        )
        raw = cycles[cycles["unit"] == engine]
        if not chosen:
            st.info("Pick one or more sensors.")
        else:
            fig = make_subplots(
                rows=len(chosen),
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=[sensor_label(s) for s in chosen],
            )
            for i, s in enumerate(chosen, start=1):
                fig.add_scatter(
                    x=raw["cycle"],
                    y=raw[s],
                    name="Reading",
                    line={"color": GRAY, "width": 1},
                    showlegend=i == 1,
                    row=i,
                    col=1,
                )
                fig.add_scatter(
                    x=raw["cycle"],
                    y=raw[s].rolling(WINDOW, min_periods=1).mean(),
                    name=f"{WINDOW}-cycle mean (model feature)",
                    line={"color": BLUE, "width": 2},
                    showlegend=i == 1,
                    row=i,
                    col=1,
                )
            fig.add_vline(x=cycle, line={"color": BLUE, "width": 1})
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(finish(fig, 130 * len(chosen) + 90), config=CHART)

# ---------------------------------------------------------------- maintenance planner
with tabs[2]:
    st.subheader("Turn predictions into a maintenance policy")
    st.caption(
        f"An engine truly needs service when it has ≤ {MAINTENANCE_AT} cycles left "
        f"({int((engines['actual_rul'] <= MAINTENANCE_AT).sum())} of 100 test engines). The alert fires "
        "when the *predicted* RUL is at or below the threshold in the sidebar. Alerting earlier catches "
        "more failures but grounds healthy engines."
    )
    cost_ratio = st.slider(
        "One missed failure costs as much as this many unnecessary inspections",
        1,
        50,
        10,
        help="An assumption for you to change, not a measured number.",
    )
    sweep = alert_sweep(engines, cost_ratio)
    now, best = sweep.loc[threshold], int(sweep["cost"].idxmin())
    k = st.columns(4)
    k[0].metric("Failures caught", f"{now['caught']} / {now['caught'] + now['missed']}", border=True)
    k[1].metric("Failures missed", int(now["missed"]), border=True)
    k[2].metric("False alarms", int(now["false_alarms"]), border=True)
    k[3].metric(
        "Relative cost",
        f"{now['cost']:.0f}",
        border=True,
        delta=f"{now['cost'] - sweep.at[best, 'cost']:+.0f} vs best (≤ {best})",
        delta_color="inverse",
    )

    left, right = st.columns(2, gap="large")
    with left:
        fig = go.Figure()
        fig.add_scatter(
            x=sweep.index, y=sweep["missed"], name="Missed failures", line={"color": BLUE, "width": 2}
        )
        fig.add_scatter(
            x=sweep.index, y=sweep["false_alarms"], name="False alarms", line={"color": ORANGE, "width": 2}
        )
        fig.add_vline(x=threshold, line={"color": GRAY, "width": 1}, annotation_text=f"now ≤ {threshold}")
        fig.update_layout(
            title="The trade-off",
            xaxis_title="Alert threshold (predicted RUL)",
            yaxis_title="Engines",
            hovermode="x unified",
        )
        st.plotly_chart(finish(fig, 360), config=CHART)
    with right:
        fig = go.Figure()
        fig.add_scatter(
            x=sweep.index, y=sweep["cost"], name="Relative cost", line={"color": BLUE, "width": 2}
        )
        fig.add_scatter(
            x=[best],
            y=[sweep.at[best, "cost"]],
            mode="markers+text",
            name="Lowest cost",
            marker={"color": BLUE, "size": 10},
            text=[f"best ≤ {best}"],
            textposition="top center",
            showlegend=False,
        )
        fig.add_vline(x=threshold, line={"color": GRAY, "width": 1})
        fig.update_layout(
            title=f"Cost = {cost_ratio} × missed + false alarms",
            xaxis_title="Alert threshold (predicted RUL)",
            yaxis_title="Relative cost",
            showlegend=False,
        )
        st.plotly_chart(finish(fig, 360), config=CHART)

    due = engines["actual_rul"] <= MAINTENANCE_AT
    missed = engines[due & ~engines["alert"]]
    false = engines[engines["alert"] & ~due]
    c = st.columns(2)
    c[0].markdown("**Missed: needed service, no alert**")
    c[0].dataframe(missed[["predicted_rul", "actual_rul"]].round(0), width="stretch")
    c[1].markdown("**False alarms: alert, but still had > 30 cycles**")
    c[1].dataframe(false[["predicted_rul", "actual_rul"]].round(0), width="stretch")

# ---------------------------------------------------------------- model card
with tabs[3]:
    m = card["test_metrics"]
    st.subheader(f"{card['name']} · version {card['version']}")
    st.badge("production", icon=":material/rocket_launch:", color="green")
    st.caption(
        f"MLflow run `{card['run_id']}` · trained {card['trained_at']} · data `{card['data']['dataset']}` "
        f"(sha256 `{card['data']['train_sha256'][:12]}…`)"
    )
    k = st.columns(5)
    k[0].metric(
        "CV RMSE", card["cv_rmse"], help="GroupKFold(5): each engine in exactly one fold", border=True
    )
    k[1].metric("Test RMSE", m["rmse"], help="100 unseen engines, at their last cycle", border=True)
    k[2].metric("Test MAE", m["mae"], border=True)
    k[3].metric("R²", m["r2"], border=True)
    k[4].metric("NASA score", m["nasa"], help="Asymmetric: late predictions cost more", border=True)

    left, right = st.columns(2, gap="large")
    with left:
        comp = pd.DataFrame(card["comparison"])
        fig = go.Figure()
        fig.add_bar(
            y=comp["model"],
            x=comp["cv_rmse"],
            name="Cross-validation RMSE",
            orientation="h",
            marker_color=GRAY,
        )
        fig.add_bar(
            y=comp["model"],
            x=comp["rmse"],
            name="Test RMSE",
            orientation="h",
            marker_color=BLUE,
            text=comp["rmse"],
            textposition="outside",
        )
        fig.update_layout(
            title="Model selection (lower is better)",
            barmode="group",
            bargap=0.35,
            xaxis_title="RMSE (cycles)",
        )
        st.plotly_chart(finish(fig, 340), config=CHART)
    with right:
        fig = go.Figure()
        fig.add_scatter(
            x=[0, 150],
            y=[0, 150],
            mode="lines",
            line={"color": GRAY, "width": 1, "dash": "dash"},
            name="Perfect prediction",
        )
        fig.add_scatter(
            x=engines["actual_rul"],
            y=engines["predicted_rul"],
            mode="markers",
            name="Test engine",
            marker={"color": BLUE, "size": 9},
            customdata=engines.index,
            hovertemplate="Engine %{customdata}<br>Actual %{x}<br>Predicted %{y:.0f}<extra></extra>",
        )
        fig.update_layout(
            title="Predicted vs actual RUL (100 test engines)",
            xaxis_title="Actual RUL",
            yaxis_title="Predicted RUL",
        )
        st.plotly_chart(finish(fig, 340), config=CHART)

    left, right = st.columns(2, gap="large")
    with left:
        importance = contributions.abs().mean().sort_values(ascending=False).head(10)
        fig = go.Figure(
            go.Bar(
                x=importance.values,
                y=[pretty(f) for f in importance.index],
                orientation="h",
                marker_color=BLUE,
            )
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            title="What the model looks at (mean |SHAP| across the fleet)",
            xaxis_title="Average effect on prediction (cycles)",
        )
        st.plotly_chart(finish(fig, 380), config=CHART)
    with right:
        st.markdown("**Comparison table**")
        columns = ["cv_rmse", "rmse", "mae", "r2", "nasa", "caught", "false_alarms"]
        st.dataframe(comp.set_index("model")[columns], width="stretch")
        st.markdown("**Hyperparameters** (chosen by grid search on CV RMSE only)")
        st.json(card["params"], expanded=False)

# ---------------------------------------------------------------- drift monitor
with tabs[4]:
    st.subheader("Is new data still like the training data?")
    st.caption(
        "A swapped or miscalibrated sensor shifts the readings, and the model keeps answering confidently. "
        "The service compares each engine's first 30 cycles (healthy in every engine) with the training "
        "engines using the Population Stability Index: < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant."
    )
    source = st.segmented_control(
        "Data", ["Simulate a sensor fault", "Upload a NASA-format file"], default="Simulate a sensor fault"
    )
    batch = None
    if source == "Upload a NASA-format file":
        upload = st.file_uploader("C-MAPSS text file: 26 space-separated columns, no header", type=["txt"])
        st.caption("Try `test_FD003.txt` (other fault modes) or `test_FD002.txt` (six flight conditions).")
        if upload is not None:
            batch = read_cycles(upload)
            if batch.isna().any().any():
                st.error("That file does not have 26 numeric columns per row.")
                batch = None
    else:
        c = st.columns(2)
        sensor = c[0].selectbox("Sensor", SENSORS, index=SENSORS.index("s4"), format_func=sensor_label)
        shift = c[1].slider("Calibration offset (standard deviations)", 0.0, 3.0, 1.0, 0.25)
        batch = cycles.assign(**{sensor: cycles[sensor] + shift * cycles[sensor].std()})

    if batch is not None:
        try:
            report = drift_check(batch)
        except RuntimeError as err:
            st.error(str(err))
        else:
            psi = pd.DataFrame(report["sensors"]).set_index("sensor")
            worst = psi["psi"].idxmax()
            k = st.columns(3)
            k[0].metric("Overall status", report["status"].capitalize(), border=True)
            k[1].metric(
                "Most shifted sensor",
                sensor_label(worst),
                f"PSI {psi.at[worst, 'psi']:.2f}",
                delta_color="off",
                border=True,
            )
            k[2].metric("Engines in batch", report["engines"], border=True)
            if report["warning"]:
                st.warning(report["warning"])
            fig = go.Figure(
                go.Bar(
                    x=[sensor_label(s) for s in psi.index],
                    y=psi["psi"],
                    marker_color=[DRIFT_COLORS[s] for s in psi["status"]],
                    customdata=psi["status"],
                    hovertemplate="%{x}<br>PSI %{y:.3f} (%{customdata})<extra></extra>",
                )
            )
            for level, name in [(0.1, "moderate"), (0.25, "significant")]:
                fig.add_hline(
                    y=level,
                    line={"color": GRAY, "width": 1, "dash": "dash"},
                    annotation_text=name,
                    annotation_position="top left",
                )
            fig.update_layout(title="PSI per sensor", yaxis_title="PSI")
            st.plotly_chart(finish(fig, 360), config=CHART)

        st.markdown("**Score this batch through the live service**")
        st.caption("Predictions also feed the service's live drift window, which Grafana charts.")
        c = st.columns([1, 1, 2])
        if c[0].button("Send batch to the service", icon=":material/send:", type="primary"):
            try:
                body = api("POST", "/api/v1/predictions", {"readings": to_readings(batch)})
            except RuntimeError as err:
                st.error(str(err))
            else:
                scored = pd.DataFrame(body["engines"]).set_index("unit")
                st.success(f"Scored {len(scored)} engines. Watch the drift panel in Grafana.")
                st.dataframe(scored[["last_cycle", "predicted_rul", "health_band", "alert"]], height=250)
                st.download_button(
                    "Download predictions (CSV)", scored.to_csv(), "predictions.csv", "text/csv"
                )
        if c[1].button("Restore clean fleet data", icon=":material/restart_alt:"):
            try:
                api("POST", "/api/v1/predictions", {"readings": to_readings(cycles)})
            except RuntimeError as err:
                st.error(str(err))
            else:
                st.success("Sent the original 100 engines; the live drift gauges will settle back.")

# ---------------------------------------------------------------- about
with tabs[5]:
    st.subheader("How it fits together")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR; bgcolor="transparent"; nodesep=0.4;
          node [shape=box, style="rounded,filled", fillcolor="#1c5cab", color="#3987e5",
                fontcolor="white", fontname="Helvetica", fontsize=11];
          edge [color="#898781", fontcolor="#898781", fontname="Helvetica", fontsize=9];
          data [label="NASA C-MAPSS\\nFD001"];
          train [label="Training pipeline\\nGroupKFold grid search"];
          mlflow [label="MLflow\\ntracking + registry"];
          bundle [label="Model bundle\\nmodel + metadata\\n+ drift profile"];
          api [label="FastAPI service\\n/api/v1"];
          dash [label="Streamlit\\ndashboard"];
          prom [label="Prometheus"];
          graf [label="Grafana"];
          data -> train;
          train -> mlflow [label="runs, metrics,\\nmodel versions"];
          train -> bundle [label="promote only if\\nCV RMSE ≤ production"];
          bundle -> api;
          dash -> api [label="REST"];
          api -> prom [label="/metrics"];
          prom -> graf;
        }
        """,
        width="stretch",
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            f"""
**Decisions that matter**
- **GroupKFold by engine.** Rows from one engine are near-copies; a random split leaks them
  across folds and reports 15.2 RMSE instead of the honest 18.3.
- **RUL capped at {RUL_CAP}.** Early in life an engine shows no wear, so no sensor can tell
  200 cycles from 300. Capping makes the target learnable.
- **Backward-looking features.** {WINDOW}-cycle rolling mean and spread per sensor, computed per
  engine. A test proves a prediction never uses future cycles.
- **One code path.** Training, the API and the tests share the same validation and feature code,
  and a test checks the served model reproduces the offline test RMSE.
- **Promotion gate.** Every run is registered in MLflow; only a model at least as good as
  production (on CV, never on test) replaces it.
"""
        )
    with right:
        st.markdown(
            """
**Honest limitations**
- One operating condition and one fault mode (FD001). FD002-FD004 are out of distribution,
  which the drift monitor detects rather than hides.
- The worst misses (engines 37, 79, 41) are engines whose sensors disagree; their SHAP
  breakdowns show it.
- Costs in the maintenance planner are an adjustable assumption, not airline data.
- Engines with more than 125 cycles left are predicted early on purpose (the cap).
"""
        )
