"""Visual theme: CSS injection and Plotly styling for the dashboard."""

from __future__ import annotations

import plotly.graph_objects as go

# Cohesive palette: deep slate + teal/cyan motion accent
ACCENT = "#2dd4bf"
ACCENT_DIM = "rgba(45, 212, 191, 0.15)"
MUTED = "#94a3b8"
SURFACE = "rgba(17, 24, 39, 0.85)"
BORDER = "rgba(148, 163, 184, 0.12)"

CHART_COLORS = ["#2dd4bf", "#a78bfa", "#38bdf8", "#fbbf24", "#fb7185", "#4ade80"]


def inject_styles() -> None:
    import streamlit as st

    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="st-"] {
  font-family: "DM Sans", system-ui, sans-serif !important;
}

#root .stApp {
  background:
    radial-gradient(ellipse 120% 80% at 50% -20%, rgba(45, 212, 191, 0.12), transparent 50%),
    radial-gradient(ellipse 80% 50% at 100% 50%, rgba(167, 139, 250, 0.06), transparent 45%),
    linear-gradient(180deg, #06080d 0%, #0c1118 50%, #080b10 100%);
}

[data-testid="stHeader"] {
  background: rgba(6, 8, 13, 0.85);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid """ + BORDER + """;
}

.block-container {
  padding-top: 1.25rem !important;
  padding-bottom: 3rem !important;
  max-width: 1320px !important;
}

/* Hero */
.velo-hero {
  position: relative;
  overflow: hidden;
  border-radius: 20px;
  padding: 2rem 2.25rem 2.25rem;
  margin-bottom: 1.75rem;
  background:
    linear-gradient(145deg, rgba(45, 212, 191, 0.09) 0%, rgba(17, 24, 39, 0.9) 45%, rgba(15, 23, 42, 0.95) 100%);
  border: 1px solid """ + BORDER + """;
  box-shadow:
    0 0 0 1px rgba(255,255,255,0.03) inset,
    0 24px 48px -24px rgba(0, 0, 0, 0.5);
}
.velo-hero::before {
  content: "";
  position: absolute;
  top: 0; right: 0;
  width: 55%;
  height: 100%;
  background: radial-gradient(circle at 70% 30%, """ + ACCENT_DIM + """ 0%, transparent 55%);
  pointer-events: none;
}
.velo-hero-inner { position: relative; z-index: 1; }
.velo-kicker {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: """ + ACCENT + """;
  margin-bottom: 0.65rem;
}
.velo-title {
  font-size: clamp(1.65rem, 4vw, 2.35rem);
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.15;
  color: #f8fafc;
  margin: 0 0 0.5rem 0;
}
.velo-sub {
  font-size: 1.05rem;
  color: """ + MUTED + """;
  margin: 0;
  max-width: 42rem;
  line-height: 1.55;
}
.velo-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-top: 1.35rem;
  padding-top: 1.25rem;
  border-top: 1px solid """ + BORDER + """;
}
.velo-pill {
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 0.75rem;
  color: #cbd5e1;
  background: rgba(15, 23, 42, 0.6);
  border: 1px solid """ + BORDER + """;
  border-radius: 999px;
  padding: 0.35rem 0.85rem;
}

/* Section titles inside tabs */
.velo-section-title {
  font-size: 1.05rem;
  font-weight: 600;
  color: #e2e8f0;
  margin: 0 0 1rem 0;
  letter-spacing: -0.02em;
}
.velo-section-hint {
  font-size: 0.9rem;
  color: """ + MUTED + """;
  margin: -0.5rem 0 1.25rem 0;
  line-height: 1.45;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 6px;
  background: rgba(15, 23, 42, 0.65);
  padding: 8px;
  border-radius: 14px;
  border: 1px solid """ + BORDER + """;
  flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px !important;
  padding: 0.55rem 1rem !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  color: """ + MUTED + """ !important;
  border: none !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(180deg, rgba(45, 212, 191, 0.2) 0%, rgba(45, 212, 191, 0.08) 100%) !important;
  color: #f1f5f9 !important;
  box-shadow: 0 1px 0 rgba(45, 212, 191, 0.35) inset;
}
.stTabs [data-baseweb="tab-panel"] {
  padding-top: 1.35rem;
}

/* Metrics */
[data-testid="stMetric"] {
  background: """ + SURFACE + """;
  border: 1px solid """ + BORDER + """;
  border-radius: 14px;
  padding: 1rem 1.1rem !important;
}
[data-testid="stMetricLabel"] div {
  color: """ + MUTED + """ !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
}
[data-testid="stMetricValue"] div {
  font-family: "JetBrains Mono", monospace !important;
  font-size: 1.35rem !important;
  font-weight: 500 !important;
  color: #f8fafc !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a0e14 0%, #0f141c 100%) !important;
  border-right: 1px solid """ + BORDER + """ !important;
}
[data-testid="stSidebar"] .block-container {
  padding-top: 1.5rem !important;
}
.velo-sidebar-brand {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: """ + ACCENT + """;
  margin-bottom: 0.35rem;
}
.velo-sidebar-title {
  font-size: 1.15rem;
  font-weight: 700;
  color: #f1f5f9;
  margin: 0 0 1.25rem 0;
  letter-spacing: -0.02em;
}
.velo-sidebar-rule {
  height: 1px;
  background: linear-gradient(90deg, """ + ACCENT + """, transparent);
  opacity: 0.35;
  margin: 1rem 0 1.25rem 0;
  border-radius: 1px;
}

/* Sliders & inputs */
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
  background-color: """ + ACCENT + """ !important;
}
[data-testid="stSidebar"] .stSlider label {
  color: #cbd5e1 !important;
  font-size: 0.82rem !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
  border: 1px solid """ + BORDER + """;
  border-radius: 12px;
  overflow: hidden;
}

hr.velo-divider {
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, """ + BORDER + """, transparent);
  margin: 1.5rem 0;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def style_plotly(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.45)",
        font=dict(family="DM Sans, sans-serif", color="#cbd5e1", size=12),
        title_font=dict(size=15, color="#f1f5f9"),
        margin=dict(l=48, r=24, t=48, b=48),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.5)",
            bordercolor=BORDER,
            borderwidth=1,
        ),
        xaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.12)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
        ),
        yaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.12)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
        ),
        colorway=CHART_COLORS,
    )
    return fig
