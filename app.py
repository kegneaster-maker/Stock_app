import streamlit as st
import yfinance as yf
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- 1. የገጹን ገጽታ (Design) በ CSS ማሳመሪያ ---
def set_design():
    st.markdown("""
    <style>
    /* ዋናው ገጽ ዳራ */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    /* የሎጊን ሳጥን ዲዛይን */
    section[data-testid="stSidebar"] {
        background-image: linear-gradient(#1e1e2f, #2d3436);
        color: white;
    }
    /* የተለየ ቀለም ለርዕሶች */
    h1 {
        color: #00ffcc;
        text-shadow: 2px 2px #000000;
    }
    /* የገበያ ምልክቶች (Green/Red) */
    .market-up { color: #00ff00; font-weight: bold; }
    .market-down { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. የሎጊን ተግባር ---
def login():
    st.sidebar.markdown("## 🔐 የባለሙያ መግቢያ")
    st.sidebar.info("የአክሲዮን ገበያ ትንተና ሲስተም")
    
    username = st.sidebar.text_input("👤 የተጠቃሚ ስም (Username)")
    password = st.sidebar.text_input("🔑 የይለፍ ቃል (Password)", type="password")
    
    # ስም እና ፓስወርድ እዚህ ጋር ቀይር
    if st.sidebar.button("አሁኑኑ ግባ"):
        if username == "Aster" and password == "2024":
            st.session_state['logged_in'] = True
            st.rerun()
        else:
            st.sidebar.error("❌ ስህተት አለ! እባክዎ እንደገና ይሞክሩ።")

# --- 3. ዋናው የትንበያ ገጽ (Market Dashboard) ---
def run_prediction_app():
    st.title("📊 Enterprise Stock Market Dashboard")
    st.markdown("እንኳን ወደ <span class='market-up'>ዘመናዊ</span> የአክሲዮን <span class='market-down'>ትንበያ</span> ሲስተም በሰላም መጡ።", unsafe_allow_html=True)
    
    ticker = st.text_input("የኩባንያ ምልክት ያስገቡ (e.g. TSLA, NVDA, GOOGL)", "NVDA")
    
    if st.button("📈 ትንበያውን አሳይ"):
        with st.spinner('መረጃው ከገበያ እየተሰበሰበ ነው...'):
            data = yf.download(ticker, start="2022-01-01")
            
            # ቀላል የትንበያ ሞዴል
            df = data[['Close']].copy()
            df['MA10'] = df['Close'].rolling(window=10).mean()
            df = df.dropna()
            
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df)
            X, y = scaled[:-1], scaled[1:, 0]
            
            model = XGBRegressor()
            model.fit(X, y)
            preds = model.predict(X)
            
            # ግራፍ በፋይናንስ ቀለም
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#1e1e2f')
            ax.plot(df.index[1:], y, label="Actual Price", color='#00ffcc')
            ax.plot(df.index[1:], preds, label="Predicted", color='#ff4b4b', linestyle='--')
            ax.tick_params(colors='white')
            ax.legend()
            st.pyplot(fig)

# --- ዋና መቆጣጠሪያ ---
set_design()

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login()
    st.warning("⚠️ ይህ ሲስተም የተቆለፈ ነው። እባክዎ መጀመሪያ በጎን በኩል ይግቡ።")
else:
    if st.sidebar.button("🚪 ውጣ (Logout)"):
        st.session_state['logged_in'] = False
        st.rerun()
    run_prediction_app()