import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, confusion_matrix)
from mlxtend.frequent_patterns import apriori, association_rules

# --- CONFIG ---
st.set_page_config(page_title="CapitalBridge Advisors | Strategic OS", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv('capitalbridge_data.csv')
    except FileNotFoundError:
        return None

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("💎 CapitalBridge OS")
st.sidebar.markdown("---")
menu = st.sidebar.selectbox("Analysis Tier", 
    ["Market Overview", "Client Classification", "Strategic Association Rules", "Revenue Regression"])

if df is None:
    st.error("Data file not found. Please ensure 'capitalbridge_data.csv' is in the root directory.")
else:
    # --- TIER 1: MARKET OVERVIEW (Descriptive) ---
    if menu == "Market Overview":
        st.title("📊 Market Intelligence Dashboard")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active Pipeline", len(df))
        m2.metric("Total Funding Demand", f"${df['Funding_Needed_USD'].sum()/1e6:.1f}M")
        m3.metric("Avg Growth Rate", f"{df['Growth_Rate'].mean():.1f}%")
        m4.metric("UAE/India Split", f"{len(df[df['Region']=='UAE'])} / {len(df[df['Region']=='India'])}")

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.histogram(df, x="Industry", color="Region", barmode="group", title="Industry Distribution by Region")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.scatter(df, x="Revenue_USD", y="Funding_Needed_USD", color="Industry", size="Growth_Rate",
                             log_x=True, log_y=True, title="Revenue vs. Funding Demand (Log Scale)")
            st.plotly_chart(fig2, use_container_width=True)

    # --- TIER 2: CLIENT CLASSIFICATION (Predictive) ---
    elif menu == "Client Classification":
        st.title("🎯 Propensity to Engage Classification")
        
        # Preprocessing
        features = ['Region', 'Industry', 'Revenue_USD', 'Growth_Rate', 'Audit_Readiness', 'Independent_Board']
        X = pd.get_dummies(df[features], drop_first=True)
        y = df['Propensity_Label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Performance Metrics
        st.subheader("Model Performance")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        k2.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
        k3.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
        k4.metric("F1-Score", f"{f1_score(y_test, y_pred):.2%}")

        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC={roc_auc:.2f})',
                              labels={'x':'False Positive Rate', 'y':'True Positive Rate'})
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with chart_col2:
            # Feature Importance
            importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
            importances = importances.sort_values('importance', ascending=False).head(10)
            fig_imp = px.bar(importances, x='importance', y='feature', orientation='h', title="Key Drivers of Client Conversion")
            st.plotly_chart(fig_imp, use_container_width=True)

    # --- TIER 3: ASSOCIATION RULES (Diagnostic) ---
    elif menu == "Strategic Association Rules":
        st.title("🔗 Business Challenge Associations")
        st.write("Mining patterns between business challenges and industries.")
        
        basket = pd.get_dummies(df[['Industry', 'Challenge', 'Region']])
        frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        
        # Displaying rules with Confidence and Lift
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False), use_container_width=True)

    # --- TIER 4: REVENUE REGRESSION ---
    elif menu == "Revenue Regression":
        st.title("💰 Revenue & Advisory Fee Forecasting")
        
        X_reg = df[['Revenue_USD', 'Growth_Rate', 'Audit_Readiness']]
        y_reg = df['Funding_Needed_USD']
        
        reg_model = LinearRegression()
        reg_model.fit(X_reg, y_reg)
        
        st.subheader("Predict Funding Need for a Prospect")
        p1, p2, p3 = st.columns(3)
        in_rev = p1.number_input("Annual Revenue (USD)", value=5000000)
        in_gr = p2.slider("Growth Rate (%)", 5, 200, 30)
        in_audit = p3.slider("Audit Readiness", 1, 10, 5)
        
        pred_funding = reg_model.predict([[in_rev, in_gr, in_audit]])[0]
        st.info(f"Predicted Funding Requirement: **${pred_funding:,.2f}**")
        st.success(f"Potential CapitalBridge Success Fee (2%): **${pred_funding * 0.02:,.2f}**")
