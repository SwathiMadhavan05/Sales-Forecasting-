# ===============================
# IMPORTS
# ===============================
import os
import hashlib
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="SalesNow",
    layout="wide"
)


# ===============================
# SESSION STATE
# ===============================
if "page" not in st.session_state:
    st.session_state.page = "landing"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


# ===============================
# AUTHENTICATION
# ===============================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

USERS = {
    "admin": hash_password("admin123"),
    "manager": hash_password("manager123"),
    "analyst": hash_password("analyst123"),
}


def login_page():
    logo_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets",
        "logo.png"
    )

    # --- Centered logo + SalesNow ---
    spacer1, content, spacer2 = st.columns([3, 4, 3])

    with content:
        col_logo, col_text = st.columns([1, 3])

        with col_logo:
            st.image(logo_path, width=220)

        with col_text:
            st.markdown(
                "<h1 class='animate' style='margin:0;'>SalesNow</h1>",
                unsafe_allow_html=True
            )

    # --- Centered Login text ---
    st.markdown(
        "<h2 class='animate animate-delay-1' style='text-align:center;'>Login</h2>",
        unsafe_allow_html=True
    )

    # --- Login form (unchanged) ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if username in USERS and USERS[username] == hash_password(password):
                st.session_state.authenticated = True
                st.session_state.page = "landing"
                st.rerun()
            else:
                st.error("Invalid username or password")


def logout():
    st.session_state.authenticated = False
    st.session_state.page = "login"
    st.rerun()



# ===============================
# CSS + ANIMATIONS (ONLY)
# ===============================
st.markdown("""
<style>
@import url('https://fonts.cdnfonts.com/css/satoshi');

* {
    font-family: 'Satoshi', sans-serif !important;
}

.stApp {
    background-color: #1e3a8a;
}

.block-container {
    padding: 2rem;
}

section[data-testid="stVerticalBlock"] > div {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    color: #0f172a;
}

h1, h2, h3 {
    color: white !important;
    font-weight: 700;
}

[data-testid="stSidebar"] {
    background-color: #0f172a;
    overflow-y: auto !important;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600;
    padding: 0.6rem 1rem;
}

footer {
    visibility: hidden;
}

/* ===== ANIMATIONS ONLY ===== */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.animate {
    animation: fadeIn 0.9s ease forwards;
}

.animate-delay-1 { animation-delay: 0.2s; }
.animate-delay-2 { animation-delay: 0.4s; }
.animate-delay-3 { animation-delay: 0.6s; }

/* Calendar fix */
div[data-baseweb="calendar"] {
    z-index: 100000 !important;
    position: fixed !important;
}

section.main > div {
    overflow: visible !important;
}

html, body {
    overflow-y: auto !important;
}
</style>
""", unsafe_allow_html=True)


# ===============================
# LANDING PAGE (ANIMATED)
# ===============================
def landing_page():
    logo_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets",
        "logo.png"
    )

    st.markdown("""
    <div class="animate" style="
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        padding: 80px 40px;
        border-radius: 24px;
        text-align: center;
        color: white;
    ">
    """, unsafe_allow_html=True)

    col_logo, col_text = st.columns([1, 6])

    with col_logo:
        st.image(logo_path, width=360)

    with col_text:
        st.markdown(
            "<h1 style='margin:0; color:white;'>SalesNow</h1>",
            unsafe_allow_html=True
        )

    st.markdown("""
        <p style="font-size:18px; max-width:750px; margin:auto; opacity:0.95;">
            SalesNow is an intelligent sales forecasting platform that helps
            businesses analyze historical trends, predict future demand,
            and make confident, data-driven decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Enter Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()


# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    train = pd.read_csv(os.path.join(base_dir, "data", "train.csv"))
    features = pd.read_csv(os.path.join(base_dir, "data", "features.csv"))
    stores = pd.read_csv(os.path.join(base_dir, "data", "stores.csv"))

    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores, on="Store", how="left")

    df["Date"] = pd.to_datetime(df["Date"])
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

    return df


# ===============================
# DASHBOARD (ANIMATED SECTIONS)
# ===============================
def dashboard():
    df = load_data()

    logo_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets",
        "logo.png"
    )

    st.sidebar.button("Logout", on_click=logout)

    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.image(logo_path, width=220)
    with col_title:
        st.markdown("<h1 style='margin:0;'>SalesNow Dashboard</h1>", unsafe_allow_html=True)

    st.sidebar.header("User Inputs")

    store_id = st.sidebar.selectbox("Select Store", sorted(df["Store"].unique()))
    dept_id = st.sidebar.selectbox("Select Department", sorted(df["Dept"].unique()))
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [df["Date"].min(), df["Date"].max()]
    )

    model_type = st.sidebar.radio(
        "Select Model",
        ["Linear Regression", "Support Vector Regression (SVR)"]
    )

    holiday_only = st.sidebar.checkbox("Show Holiday Sales Only")
    weeks_ahead = st.sidebar.slider("Forecast Weeks Ahead", 1, 12, 4)

    filtered_df = df[
        (df["Store"] == store_id) &
        (df["Dept"] == dept_id) &
        (df["Date"] >= pd.to_datetime(date_range[0])) &
        (df["Date"] <= pd.to_datetime(date_range[1]))
    ]

    if holiday_only:
        filtered_df = filtered_df[filtered_df["IsHoliday"]]

    if len(filtered_df) < 30:
        st.warning("Not enough data for selected inputs.")
        return

    st.markdown("<div class='animate animate-delay-1'>", unsafe_allow_html=True)
    st.subheader("Key Business Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sales", f"{filtered_df['Weekly_Sales'].sum():,.0f}")
    c2.metric("Average Weekly Sales", f"{filtered_df['Weekly_Sales'].mean():,.0f}")
    c3.metric("Maximum Weekly Sales", f"{filtered_df['Weekly_Sales'].max():,.0f}")
    st.markdown("</div>", unsafe_allow_html=True)

    X = filtered_df[
        ["Year", "Month", "Week", "IsHoliday",
         "Temperature", "Fuel_Price", "CPI",
         "Unemployment", "Size"]
    ]
    y = filtered_df["Weekly_Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    if model_type == "Linear Regression":
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = SVR(C=100, epsilon=0.1).fit(X_train, y_train)
        y_pred = model.predict(X_test)

    st.markdown("<div class='animate animate-delay-2'>", unsafe_allow_html=True)
    st.subheader("Actual vs Predicted Sales")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test.values, label="Actual")
    ax.plot(y_pred, label="Predicted")
    ax.legend()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='animate animate-delay-3'>", unsafe_allow_html=True)
    st.subheader("Filtered Data Preview")
    st.dataframe(filtered_df.head(20))
    st.markdown("</div>", unsafe_allow_html=True)


# ===============================
# ROUTER
# ===============================
if not st.session_state.authenticated:
    login_page()
elif st.session_state.page == "landing":
    landing_page()
else:
    dashboard()
