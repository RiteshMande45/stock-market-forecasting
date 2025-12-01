import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Stock Predictor", page_icon="🚀", layout="wide")

st.markdown("""
<style>
/* Main app background */
.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2d1b4e 100%);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1f3a 0%, #0f1419 100%) !important;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Sidebar title */
[data-testid="stSidebar"] h1 {
    color: #ffcc00 !important;
    text-shadow: 0 0 15px rgba(255, 204, 0, 0.8);
    font-weight: bold !important;
    font-size: 24px !important;
}

/* Sidebar navigation header */
[data-testid="stSidebar"] .stRadio > label {
    color: #00ff88 !important;
    font-size: 20px !important;
    font-weight: bold !important;
    text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
}

/* Sidebar Quick Stats header */
[data-testid="stSidebar"] h3 {
    color: #ff00ff !important;
    font-size: 20px !important;
    font-weight: bold !important;
    text-shadow: 0 0 10px rgba(255, 0, 255, 0.5);
}

/* Sidebar metrics */
[data-testid="stSidebar"] .stMetric {
    background: linear-gradient(135deg, #6a0572, #ab20fd) !important;
    padding: 15px !important;
    border-radius: 12px !important;
    border: 2px solid #ff00ff !important;
    box-shadow: 0 5px 20px rgba(255, 0, 255, 0.4) !important;
}

[data-testid="stSidebar"] .stMetric label {
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: bold !important;
}

[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
    color: #ffcc00 !important;
    font-size: 28px !important;
    font-weight: bold !important;
}

/* Main content headings */
h1 {
    color: #ffcc00 !important;
    text-shadow: 0 0 20px rgba(255, 204, 0, 0.9);
    font-weight: bold !important;
    font-size: 48px !important;
}

h2 {
    color: #00ff88 !important;
    text-shadow: 0 0 15px rgba(0, 255, 136, 0.8);
    font-weight: bold !important;
    font-size: 32px !important;
}

h3 {
    color: #00d4ff !important;
    text-shadow: 0 0 10px rgba(0, 212, 255, 0.8);
    font-weight: bold !important;
    font-size: 24px !important;
}

/* Main content metrics */
.stMetric {
    background: linear-gradient(135deg, #ff006e 0%, #8338ec 50%, #3a86ff 100%) !important;
    padding: 25px !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 30px rgba(255, 0, 110, 0.5) !important;
    border: 3px solid rgba(255, 255, 255, 0.3) !important;
}

.stMetric label {
    color: #ffffff !important;
    font-size: 20px !important;
    font-weight: bold !important;
}

.stMetric [data-testid="stMetricValue"] {
    color: #ffcc00 !important;
    font-size: 36px !important;
    font-weight: bold !important;
    text-shadow: 0 0 10px rgba(255, 204, 0, 0.8);
}

.stMetric [data-testid="stMetricDelta"] {
    color: #00ff88 !important;
    font-size: 20px !important;
    font-weight: bold !important;
}

/* Selectbox styling */
.stSelectbox label {
    color: #ffcc00 !important;
    font-size: 22px !important;
    font-weight: bold !important;
    text-shadow: 0 0 10px rgba(255, 204, 0, 0.6);
}

div[data-baseweb="select"] {
    background: linear-gradient(135deg, #6a0572, #ab20fd) !important;
    border: 2px solid #ff00ff !important;
}

div[data-baseweb="select"] > div {
    color: #ffffff !important;
    font-weight: bold !important;
}

/* All text visible */
p, span, label, div {
    color: #ffffff !important;
}

/* Radio buttons */
.stRadio > div {
    background: rgba(106, 5, 114, 0.3);
    padding: 15px;
    border-radius: 12px;
    border: 2px solid rgba(255, 0, 255, 0.3);
}

.stRadio label {
    color: #ffffff !important;
    font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try: return pd.read_csv('data/processed/stock_data_with_features.csv', index_col='Date', parse_dates=True)
    except: return None

@st.cache_data
def load_predictions():
    try: return pd.read_csv('data/processed/arima_sarima_predictions.csv', index_col=0, parse_dates=True)
    except: return None

with st.sidebar:
    st.markdown("# 🚀 AI STOCK PREDICTOR")
    st.markdown("---")
    page = st.radio("📍 Navigate To:", ["🏠 Dashboard", "🤖 AI Predictions", "⚡ Model Battle"], label_visibility="visible")
    st.markdown("---")
    df = load_data()
    if df is not None:
        st.markdown("### 📊 Quick Stats")
        st.metric("Latest Price", f"")
        st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")

if page == "🏠 Dashboard":
    st.markdown("# 🚀 AI STOCK MARKET DASHBOARD")
    st.markdown("### 💎 Real-time Stock Analysis with Machine Learning")
    
    df = load_data()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("💰 Current Price", f"", f"{((df['Close'].iloc[-1]-df['Close'].iloc[-2])/df['Close'].iloc[-2]*100):+.2f}%")
        with col2: st.metric("📈 24h High", f"")
        with col3: st.metric("📉 24h Low", f"")
        with col4: st.metric("📊 Volume", f"{df['Volume'].iloc[-1]:,.0f}")
        
        st.markdown("---")
        st.markdown("## 📈 PRICE MOVEMENT (CANDLESTICK)")
        
        fig = go.Figure(go.Candlestick(
            x=df.index[-100:], 
            open=df['Open'][-100:], 
            high=df['High'][-100:], 
            low=df['Low'][-100:], 
            close=df['Close'][-100:]
        ))
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['MA_21'][-100:], name='21-Day MA', line=dict(color='#ffcc00', width=3)))
        fig.update_layout(
            template='plotly_dark', 
            height=600,
            paper_bgcolor='rgba(10,14,39,0.95)',
            plot_bgcolor='rgba(26,31,58,0.95)',
            font=dict(color='#ffffff', size=14),
            title=dict(text='Stock Price Analysis', font=dict(size=24, color='#ffcc00'))
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("## 🌊 TRADING VOLUME")
        fig_vol = go.Figure(go.Bar(x=df.index[-50:], y=df['Volume'][-50:], marker_color='#ff006e'))
        fig_vol.update_layout(
            template='plotly_dark',
            height=400,
            paper_bgcolor='rgba(10,14,39,0.95)',
            plot_bgcolor='rgba(26,31,58,0.95)',
            font=dict(color='#ffffff', size=14)
        )
        st.plotly_chart(fig_vol, use_container_width=True)

elif page == "🤖 AI Predictions":
    st.markdown("# 🤖 AI MODEL PREDICTIONS")
    st.markdown("### 🎯 See How AI Predicts The Future")
    
    predictions = load_predictions()
    if predictions is not None:
        model = st.selectbox("🎯 Choose Your AI Model:", ["ARIMA", "SARIMA"])
        pred_col = 'ARIMA_Predictions' if model == "ARIMA" else 'SARIMA_Predictions'
        
        rmse = np.sqrt(mean_squared_error(predictions['Actual'], predictions[pred_col]))
        mae = mean_absolute_error(predictions['Actual'], predictions[pred_col])
        mape = np.mean(np.abs((predictions['Actual'] - predictions[pred_col]) / predictions['Actual'])) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("🎯 Accuracy", f"{100-mape:.2f}%")
        with col2: st.metric("📊 RMSE", f"{rmse:.2f}")
        with col3: st.metric("💎 MAE", f"{mae:.2f}")
        
        st.markdown("---")
        st.markdown(f"## 📈 {model} PERFORMANCE")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions['Actual'], name='ACTUAL PRICE', line=dict(color='#ffcc00', width=4)))
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions[pred_col], name=f'{model} PREDICTION', line=dict(color='#00ff88', width=4, dash='dash')))
        fig.update_layout(
            template='plotly_dark',
            height=600,
            paper_bgcolor='rgba(10,14,39,0.95)',
            plot_bgcolor='rgba(26,31,58,0.95)',
            font=dict(color='#ffffff', size=16),
            title=dict(text=f'{model} vs Reality', font=dict(size=24, color='#ffcc00')),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("# ⚡ MODEL BATTLE ARENA")
    st.markdown("### 🏆 Which AI Wins?")
    
    predictions = load_predictions()
    if predictions is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions['Actual'], name='ACTUAL', line=dict(color='#ffffff', width=5)))
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions['ARIMA_Predictions'], name='ARIMA', line=dict(color='#ffcc00', width=4, dash='dot')))
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions['SARIMA_Predictions'], name='SARIMA', line=dict(color='#00ff88', width=4, dash='dash')))
        fig.update_layout(
            template='plotly_dark',
            height=700,
            paper_bgcolor='rgba(10,14,39,0.95)',
            plot_bgcolor='rgba(26,31,58,0.95)',
            font=dict(color='#ffffff', size=16),
            title=dict(text='ALL MODELS vs REALITY', font=dict(size=28, color='#ffcc00')),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("## 🏆 LEADERBOARD")
        arima_rmse = np.sqrt(mean_squared_error(predictions['Actual'], predictions['ARIMA_Predictions']))
        sarima_rmse = np.sqrt(mean_squared_error(predictions['Actual'], predictions['SARIMA_Predictions']))
        
        col1, col2 = st.columns(2)
        with col1: st.metric("🥇 ARIMA RMSE", f"{arima_rmse:.2f}", "Winner!" if arima_rmse < sarima_rmse else "")
        with col2: st.metric("🥈 SARIMA RMSE", f"{sarima_rmse:.2f}", "Winner!" if sarima_rmse < arima_rmse else "")

st.markdown("---")
st.markdown("<h2 style='text-align: center; color: #ffcc00; text-shadow: 0 0 20px rgba(255, 204, 0, 0.8);'>🚀 Powered by AI & Machine Learning | Built with Streamlit 💎</h2>", unsafe_allow_html=True)
