import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. System Config & Design
# ---------------------------------------------------------
st.set_page_config(page_title="OccupyBed AI - Advanced", layout="wide", page_icon="🏥")

CURRENT_DATE = datetime(2026, 1, 8, 12, 0, 0)

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E6EDF3; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background-color: #010409; border-right: 1px solid #30363D; }
    
    @keyframes glow {
        from { text-shadow: 0 0 5px #fff, 0 0 10px #58A6FF; }
        to { text-shadow: 0 0 10px #fff, 0 0 20px #58A6FF; }
    }
    .logo-box { text-align: center; margin-bottom: 30px; margin-top: 10px; }
    .logo-main { 
        font-size: 28px; font-weight: 800; color: #FFFFFF; 
        animation: glow 2s infinite alternate; margin: 0; letter-spacing: 1px;
    }
    .logo-slogan { 
        font-size: 10px; color: #8B949E; text-transform: uppercase; 
        letter-spacing: 2px; margin-top: 5px; font-weight: 500;
    }

    .kpi-card {
        background-color: #161B22; border: 1px solid #30363D; border-radius: 6px;
        padding: 20px; text-align: center; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .kpi-label { font-size: 11px; color: #8B949E; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .kpi-val { font-size: 28px; font-weight: 700; color: #FFF; margin: 0; }
    .kpi-sub { font-size: 11px; color: #58A6FF; margin-top: 5px;}
    
    .section-header {
        font-size: 16px; font-weight: 700; color: #E6EDF3; 
        margin-top: 25px; margin-bottom: 15px; 
        border-left: 4px solid #58A6FF; padding-left: 10px;
    }

    .ai-container {
        background-color: #161B22; border: 1px solid #30363D; border-left: 5px solid #A371F7;
        border-radius: 6px; padding: 15px; height: 100%;
    }
    .ai-header { font-weight: 700; color: #A371F7; font-size: 14px; margin-bottom: 10px; text-transform: uppercase; }
    .ai-item { font-size: 13px; color: #E6EDF3; margin-bottom: 6px; border-bottom: 1px solid #21262D; padding-bottom: 4px; }

    .dept-card {
        background-color: #0D1117; border: 1px solid #30363D; border-radius: 6px;
        padding: 15px; margin-bottom: 12px;
    }
    .dept-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    .dept-title { font-size: 14px; font-weight: 700; color: #FFF; }
    
    .badge { padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; text-transform: uppercase; }
    .bg-safe { background: rgba(35, 134, 54, 0.2); color: #3FB950; border: 1px solid #238636; }
    .bg-warn { background: rgba(210, 153, 34, 0.2); color: #D29922; border: 1px solid #9E6A03; }
    .bg-crit { background: rgba(218, 54, 51, 0.2); color: #F85149; border: 1px solid #DA3633; }

    div[data-baseweb="select"] > div, input { background-color: #0D1117 !important; border-color: #30363D !important; color: white !important; }
    button[kind="primary"] { background-color: #238636 !important; border: none !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Load Models and Data
# ---------------------------------------------------------

import os
from pathlib import Path

@st.cache_resource
def load_model_components():
    """Load trained Random Forest model, scaler, and encoders"""
    try:
        from pathlib import Path
        import joblib
        import tensorflow as tf
        
        # Get the directory where app.py is located
        app_dir = Path(__file__).parent
        
        # Define file paths
        model_path = app_dir / 'best_model.pkl'
        scaler_path = app_dir / 'scaler.pkl'
        le_path = app_dir / 'label_encoders.pkl'
        cols_path = app_dir / 'feature_columns.pkl'
        
        # Check if all files exist
        print("\n" + "="*70)
        print("LOADING MODEL COMPONENTS")
        print("="*70)
        
        files_to_check = [
            ('Model', model_path),
            ('Scaler', scaler_path),
            ('Label Encoders', le_path),
            ('Feature Columns', cols_path)
        ]
        
        for name, path in files_to_check:
            if not path.exists():
                st.error(f"❌ Missing {name}: {path}")
                st.info(f"Current directory: {app_dir}")
                st.info(f"Files in directory: {list(app_dir.glob('*'))}")
                return None, None, None, None, False
            else:
                size = path.stat().st_size / 1024
                print(f"✓ Found {name:20}: {size:>8.2f} KB")
        
        # Load files
        st.info("Loading model components...")
        
        model = joblib.load(str(model_path))
        print(f"✓ Model loaded: {type(model).__name__}")
        
        scaler = joblib.load(str(scaler_path))
        print(f"✓ Scaler loaded")
        
        le_dict = joblib.load(str(le_path))
        print(f"✓ Label encoders loaded: {len(le_dict)} encoders")
        
        feature_cols = joblib.load(str(cols_path))
        print(f"✓ Feature columns loaded: {len(feature_cols)} features")
        
        st.success("✓ All components loaded successfully!")
        
        return model, scaler, le_dict, feature_cols, True
        
    except Exception as e:
        st.error(f"❌ Error loading components: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None, False

@st.cache_data
def load_eicu_data():
    """Load eICU patient data"""
    try:
        app_dir = Path(__file__).parent
        csv_path = app_dir / 'patient.csv'
        
        # Try local file first
        if csv_path.exists():
            df = pd.read_csv(str(csv_path))
        else:
            # Download if local file doesn't exist
            url = "https://physionet.org/content/eicu-crd-demo/2.0.1/patient.csv"
            df = pd.read_csv(url)
        
        df['icu_los_hours'] = df['unitdischargeoffset'] / 60.0
        df = df.dropna(subset=['icu_los_hours'])
        
        for col in ['unitadmitoffset', 'unitdischargeoffset', 'hospitaldischargeoffset']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        st.success(f"✓ Data loaded: {len(df)} rows")
        return df, True
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, False




model, scaler, le_dict, feature_cols, model_loaded = load_model_components()
df_eicu, data_loaded = load_eicu_data()

def append_patient_row(new_row: dict):
    """Append a new patient record to patient.csv and return updated df"""
    app_dir = Path(__file__).parent
    csv_path = app_dir / 'patient.csv'
    
    # Load current data (from cache function to keep consistency)
    df, ok = load_eicu_data()
    if not ok or df is None:
        df = pd.DataFrame()

    # Ensure we respect existing columns
    if not df.empty:
        base_cols = df.columns.tolist()
        # keep only known columns, add missing with NaN
        row_df = pd.DataFrame([new_row])
        for col in base_cols:
            if col not in row_df.columns:
                row_df[col] = pd.NA
        row_df = row_df[base_cols]
        df_updated = pd.concat([df, row_df], ignore_index=True)
    else:
        df_updated = pd.DataFrame([new_row])

    # Persist to CSV
    df_updated.to_csv(csv_path, index=False)

    # Clear cache so load_eicu_data() sees the change
    st.cache_data.clear()

    return df_updated
# ---------------------------------------------------------
# 3. Prediction Functions
# ---------------------------------------------------------

def predict_los_for_patient(patient_data, model, scaler, le_dict, feature_cols):
    """Predict ICU LOS for a single patient using Random Forest"""
    try:
        # 1. Convert dictionary to DataFrame
        new_patient = pd.DataFrame([patient_data])
        
        # 2. Fill missing features with 0 to match training schema
        for col in feature_cols:
            if col not in new_patient.columns:
                new_patient[col] = 0
        
        # 3. Apply Label Encoding
        for col, le in le_dict.items():
            if col in new_patient.columns:
                try:
                    new_patient[col] = le.transform(new_patient[col].astype(str))
                except:
                    new_patient[col] = 0 # Fallback for unknown categories
        
        # 4. Strictly align column order
        new_patient = new_patient[feature_cols]
        
        # 5. Scale and RECONSTRUCT DataFrame with names (Fixes the UserWarning)
        new_patient_scaled_array = scaler.transform(new_patient)
        new_patient_scaled_df = pd.DataFrame(new_patient_scaled_array, columns=feature_cols)
        
        # 6. Predict using the fitted model
        predicted_los = float(model.predict(new_patient_scaled_df)[0])
        return max(predicted_los, 1.0) 
        
    except Exception as e:
        return None

def generate_bed_forecast(active_patients, forecast_hours=24, total_capacity=50):
    """Generate bed occupancy forecast"""
    current_time = CURRENT_DATE
    timeline = []
    
    for hour in range(forecast_hours):
        forecast_time = current_time + timedelta(hours=hour)
        occupied = 0
        
        for _, patient in active_patients.iterrows():
            if patient['admission_time'] <= forecast_time <= patient['predicted_discharge_time']:
                occupied += 1
        
        available = max(0, total_capacity - occupied)
        timeline.append({
            'time': forecast_time,
            'occupied': occupied,
            'available': available,
            'occupancy_rate': (occupied / total_capacity) * 100
        })
    
    return pd.DataFrame(timeline)

# Build mapping from data
if data_loaded and df_eicu is not None:
    # adjust column names if needed
    unit_map = (
        df_eicu[['unittype', 'unitdischargelocation']]
        .dropna()
        .drop_duplicates()
        .set_index('unittype')['unitdischargelocation']
        .to_dict()
    )
else:
    unit_map = {}
def get_unit_type_name(unit_type):
    """Map unit type codes to names from eICU data"""
    try:
        u = int(unit_type)
    except (TypeError, ValueError):
        return str(unit_type)
    return unit_map.get(u, f"Unit {u}")
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
# ---------------------------------------------------------
# 4. Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    # Try to load the logo
    try:
        logo_base64 = get_base64_image("logo.png")
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="300" style="margin-bottom: 10px; border-radius: 8px;">'
    except Exception:
        logo_html = "" # Fallback if file is missing

    st.markdown(f"""
    <div class="logo-box">
        {logo_html}
        <div class="logo-main">OccupyBed AI</div>
        <div class="logo-slogan">ML-Powered Bed Management</div>
    </div>
    """, unsafe_allow_html=True)

        
    # Patient Search (from design.py)
    st.markdown("### Patient Search")
    search_q = st.text_input("Enter PIN", placeholder="e.g. PIN-2005")
    if search_q:
        res = df[(df['PIN'] == search_q) & (df['Actual_Discharge'].isna())]
        if not res.empty:
            r = res.iloc[0]
            st.success(f"Found: {r['Department']}")
            st.info(f"Bed: {r['Bed']}")
        else:
            st.warning("Not Active / Not Found")

    st.markdown("---")
    
    # --- Your Navigation Radio ---
    menu = st.radio(
        "NAVIGATION", 
        ["Overview", "Patient Analytics", "Bed Forecast", "Discharge Planning", "Live Admissions", "Settings"], 
        label_visibility="collapsed"
    )    
    # Model Status
    if model_loaded and data_loaded:
        st.success("✓ System Ready")
    else:
        st.error("⚠ System Incomplete")
        if not model_loaded:
            st.caption("Model files missing")
        if not data_loaded:
            st.caption("Data loading failed")
   
    
    st.markdown("---")
    st.caption(f"Reference: {CURRENT_DATE.strftime('%Y-%m-%d %H:%M')}")

# ---------------------------------------------------------
# 5. OVERVIEW 
# ---------------------------------------------------------
if menu == "Overview":
    st.title("🏥 Hospital Command Center")
    
    if not model_loaded or not data_loaded:
        st.error("❌ Cannot proceed: Model or data not loaded. Check Settings.")
        st.stop()
    
    fc_hours = st.selectbox("Forecast Window", [0.25, 0.5, 1, 2, 3, 4, 5, 6, 12, 24, 48, 72], index=2, format_func=lambda x: f"{x} Hours")
    
    # Prepare data
    df_active = df_eicu.sample(min(100, len(df_eicu))).copy()
    
    df_active['predicted_los_hours'] = df_active.apply(
        lambda row: predict_los_for_patient(row.to_dict(), model, scaler, le_dict, feature_cols),
        axis=1
    )
    
    np.random.seed(42)
    df_active['admission_time'] = CURRENT_DATE - pd.to_timedelta(
        np.random.uniform(1, 30, len(df_active)), unit='d'
    )
    df_active['predicted_discharge_time'] = (
        df_active['admission_time'] + 
        pd.to_timedelta(df_active['predicted_los_hours'], unit='h')
    )
    
    active_patients = df_active[df_active['predicted_discharge_time'] > CURRENT_DATE].copy()
    
    total_capacity = 100
    occ_count = len(active_patients)
    avail_count = total_capacity - occ_count
    ready_count = len(active_patients[
        active_patients['predicted_discharge_time'] <= CURRENT_DATE + timedelta(hours=fc_hours)
    ])
    
    # KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1: 
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total Beds</div><div class="kpi-val" style="color:#58A6FF">{total_capacity}</div></div>""", unsafe_allow_html=True)
    with k2: 
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Occupied</div><div class="kpi-val" style="color:#D29922">{occ_count}</div></div>""", unsafe_allow_html=True)
    with k3: 
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Available Now</div><div class="kpi-val" style="color:#3FB950">{avail_count}</div></div>""", unsafe_allow_html=True)
    with k4: 
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Will Free ({fc_hours}h)</div><div class="kpi-val" style="color:#A371F7">{ready_count}</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gauge & AI Recommendations
    g_col, ai_col = st.columns([1, 2])
    
    with g_col:
        occ_rate = (occ_count / total_capacity) * 100 if total_capacity > 0 else 0
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=occ_rate,
            title={'text': "Hospital Pressure"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#58A6FF"},
                'steps': [
                    {'range': [0, 70], 'color': "#161B22"},
                    {'range': [70, 85], 'color': "#451a03"},
                    {'range': [85, 100], 'color': "#450a0a"}],
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=10,r=10,t=0,b=0), 
                               paper_bgcolor="#0E1117", font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with ai_col:
        st.markdown("""<div class="ai-container"><div class="ai-header">🤖 AI Operational Insights</div>""", unsafe_allow_html=True)
        
        if occ_rate >= 85:
            st.markdown("""<div class="ai-item"><span style="color:#F85149"><b>CRITICAL:</b></span> Hospital at critical capacity. Activate surge protocol.</div>""", unsafe_allow_html=True)
        elif occ_rate >= 70:
            st.markdown("""<div class="ai-item"><span style="color:#D29922"><b>WARNING:</b></span> High occupancy detected. Prioritize pending discharges.</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="ai-item" style="color:#3FB950"><b>OPTIMAL:</b> Hospital capacity is healthy.</div>""", unsafe_allow_html=True)
        
        avg_los = active_patients['predicted_los_hours'].mean()
        st.markdown(f"""<div class="ai-item"><b>Predicted LOS:</b> {avg_los:.1f} hours ({avg_los/24:.1f} days)</div>""", unsafe_allow_html=True)
        
        next_discharge = active_patients['predicted_discharge_time'].min()
        hours_until = (next_discharge - CURRENT_DATE).total_seconds() / 3600
        st.markdown(f"""<div class="ai-item"><b>Next Predicted Bed Free:</b> {max(0, hours_until):.1f} hours</div>""", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
# ---------------------------------------------------------
    # ICU Units - Live Status (Visualizing ALL Departments)
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">ICU Units - Live Status</div>', unsafe_allow_html=True)
    
    # 1. Define the FULL list of departments and their bed capacities
    # This ensures all units are visualized even if occupancy is 0.
    UNIT_CAPS = {
        "MICU": 25,
        "SICU": 20,
        "Med-Surg ICU": 30,
        "CCU-CTICU": 15,
        "CSICU": 12,
        "Neuro ICU": 10,
        "Burn ICU": 8,      # Added even if empty
        "Trauma ICU": 12    # Added even if empty
    }

    # 2. Create columns (3 per row)
    all_units = list(UNIT_CAPS.keys())
    unit_cols = st.columns(3)
    
    for i, unit_id in enumerate(all_units):
        # Filter active patients for this specific unit
        unit_df = active_patients[active_patients['unittype'] == unit_id]
        
        # Calculate Metrics
        occ = len(unit_df)
        cap = UNIT_CAPS[unit_id]
        avail = max(0, cap - occ)
        
        # Forecast: Patients expected to be discharged in the forecast window
        future_limit = CURRENT_DATE + timedelta(hours=fc_hours)
        ready = len(unit_df[unit_df['predicted_discharge_time'] <= future_limit])
        
        # Occupancy Percentage
        pct = (occ / cap) * 100 if cap > 0 else 0
        
        # 3. Determine Status and Visual Colors
        if occ == 0:
            status, cls, bar_color = "EMPTY", "bg-safe", "#30363D" # Gray bar for empty
        elif pct < 70: 
            status, cls, bar_color = "SAFE", "bg-safe", "#3FB950"
        elif 70 <= pct <= 85: 
            status, cls, bar_color = "WARNING", "bg-warn", "#D29922"
        else: 
            status, cls, bar_color = "CRITICAL", "bg-crit", "#F85149"
        
        # Using the helper name function if it exists, otherwise just the ID
        try:
            unit_display_name = get_unit_type_name(unit_id)
        except:
            unit_display_name = unit_id
            
        # 4. Render the Card in the correct column
        with unit_cols[i % 3]:
            st.markdown(f"""
            <div class="dept-card">
                <div class="dept-header">
                    <span class="dept-title">{unit_display_name}</span>
                    <span class="badge {cls}">{status}</span>
                </div>
                <div style="font-size:12px; color:#8B949E; display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span>Cap: {cap}</span>
                    <span>Occ: <b style="color:#E6EDF3">{occ}</b></span>
                    <span>Avail: {avail}</span>
                </div>
                <div style="font-size:12px; display:flex; justify-content:space-between;">
                    <span style="color:#A371F7; font-weight:bold;">Forecast Free ({fc_hours}h): {ready}</span>
                </div>
                <div style="background:#21262D; height:6px; border-radius:3px; margin-top:10px; overflow:hidden;">
                    <div style="width:{min(pct, 100)}%; background:{bar_color}; height:100%; transition: width 0.5s ease-in-out;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
# ---------------------------------------------------------
# 6. PATIENT ANALYTICS
# ---------------------------------------------------------
elif menu == "Patient Analytics":
    st.title("📊 Patient Analytics Dashboard")
    
    if not model_loaded or not data_loaded:
        st.error("❌ Model or data not loaded.")
        st.stop()
    
    st.markdown('<div class="section-header">LOS Predictions Distribution</div>', unsafe_allow_html=True)
    
    df_sample = df_eicu.sample(min(200, len(df_eicu))).copy()
    df_sample['predicted_los_hours'] = df_sample.apply(
        lambda row: predict_los_for_patient(row.to_dict(), model, scaler, le_dict, feature_cols),
        axis=1
    )
    df_sample = df_sample.dropna(subset=['predicted_los_hours'])
    
    # Histogram of LOS predictions
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df_sample['predicted_los_hours'],
        nbinsx=30,
        name='Predicted LOS',
        marker=dict(color='#58A6FF'),
        opacity=0.7
    ))
    fig_hist.update_layout(
        title="Distribution of Predicted Length of Stay",
        xaxis_title="Hours",
        yaxis_title="Number of Patients",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean LOS", f"{df_sample['predicted_los_hours'].mean():.1f}h", 
                 f"{df_sample['predicted_los_hours'].mean()/24:.1f} days")
    with col2:
        st.metric("Median LOS", f"{df_sample['predicted_los_hours'].median():.1f}h")
    with col3:
        st.metric("Min LOS", f"{df_sample['predicted_los_hours'].min():.1f}h")
    with col4:
        st.metric("Max LOS", f"{df_sample['predicted_los_hours'].max():.1f}h")
    
    st.markdown("---")
    st.markdown('<div class="section-header">Patient Details (Sample)</div>', unsafe_allow_html=True)
    
    display_cols = ['patientunitstayid', 'gender', 'age', 'icu_los_hours', 'predicted_los_hours']
    display_df = df_sample[[c for c in display_cols if c in df_sample.columns]].head(20)
    
    st.dataframe(
        display_df.rename(columns={
            'patientunitstayid': 'Patient ID',
            'gender': 'Gender',
            'age': 'Age',
            'icu_los_hours': 'Actual LOS (h)',
            'predicted_los_hours': 'Predicted LOS (h)'
        }),
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------------
# 7. BED FORECAST (COMPLETE)
# ---------------------------------------------------------
elif menu == "Bed Forecast":
    st.title("🛏️ Bed Occupancy Forecast")
    
    if not model_loaded or not data_loaded:
        st.error("❌ Model or data not loaded.")
        st.stop()
    
    forecast_hours = st.slider("Forecast Period (hours)", 6, 168, 24, step=6)
    total_capacity = st.slider("Total ICU Capacity", 20, 100, 50)
    
    # Generate forecast
    df_forecast = df_eicu.sample(min(80, len(df_eicu))).copy()
    df_forecast['predicted_los_hours'] = df_forecast.apply(
        lambda row: predict_los_for_patient(row.to_dict(), model, scaler, le_dict, feature_cols),
        axis=1
    )
    
    np.random.seed(42)
    df_forecast['admission_time'] = CURRENT_DATE - pd.to_timedelta(
        np.random.uniform(1, 20, len(df_forecast)), unit='d'
    )
    df_forecast['predicted_discharge_time'] = (
        df_forecast['admission_time'] + 
        pd.to_timedelta(df_forecast['predicted_los_hours'], unit='h')
    )
    
    active = df_forecast[df_forecast['predicted_discharge_time'] > CURRENT_DATE].copy()
    
    # Generate timeline
    timeline_data = generate_bed_forecast(active, forecast_hours, total_capacity)
    
    st.markdown('<div class="section-header">Bed Occupancy Timeline</div>', unsafe_allow_html=True)
    
    # Occupancy line chart
    fig_occupancy = go.Figure()
    fig_occupancy.add_trace(go.Scatter(
        x=timeline_data['time'],
        y=timeline_data['occupied'],
        name='Occupied Beds',
        mode='lines',
        line=dict(color='#FF6B6B', width=3),
        fill='tozeroy'
    ))
    fig_occupancy.add_trace(go.Scatter(
        x=timeline_data['time'],
        y=timeline_data['available'],
        name='Available Beds',
        mode='lines',
        line=dict(color='#51CF66', width=3),
        fill='tozeroy'
    ))
    fig_occupancy.add_hline(y=total_capacity * 0.8, line_dash="dash", line_color="orange",
                           annotation_text="80% Alert Level", annotation_position="right")
    fig_occupancy.update_layout(
        title=f"Bed Occupancy Forecast ({forecast_hours} hours)",
        xaxis_title="Time",
        yaxis_title="Number of Beds",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color='white'),
        hovermode='x unified',
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_occupancy, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">Occupancy Rate Forecast</div>', unsafe_allow_html=True)
    
    fig_rate = go.Figure()
    fig_rate.add_trace(go.Scatter(
        x=timeline_data['time'],
        y=timeline_data['occupancy_rate'],
        name='Occupancy Rate',
        mode='lines+markers',
        line=dict(color='#58A6FF', width=2),
        marker=dict(size=6)
    ))
    fig_rate.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Warning (70%)")
    fig_rate.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Critical (85%)")
    fig_rate.update_layout(
        title="Hospital Occupancy Rate",
        xaxis_title="Time",
        yaxis_title="Occupancy Rate (%)",
        yaxis=dict(range=[0, 100]),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_rate, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">Critical Periods & Alerts</div>', unsafe_allow_html=True)
    
    # Find critical periods
    critical = timeline_data[timeline_data['occupancy_rate'] > 85]
    warning = timeline_data[timeline_data['occupancy_rate'] > 70]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if len(critical) > 0:
            st.warning(f"🚨 CRITICAL PERIODS: {len(critical)} hours")
            st.caption(f"Peak: {critical['time'].max().strftime('%H:%M')}")
        else:
            st.success("✓ No critical periods detected")
    
    with col2:
        if len(warning) > 0:
            st.info(f"⚠️ WARNING PERIODS: {len(warning)} hours")
        else:
            st.success("✓ No warning periods detected")
    
    with col3:
        min_available = timeline_data['available'].min()
        st.metric("Minimum Available Beds", int(min_available))
    
    st.markdown("---")
    st.markdown('<div class="section-header">Hourly Forecast Table</div>', unsafe_allow_html=True)
    
    display_df = timeline_data.copy()
    display_df['time'] = display_df['time'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['occupancy_rate'] = display_df['occupancy_rate'].round(1)
    
    st.dataframe(
        display_df.rename(columns={
            'time': 'Time',
            'occupied': 'Occupied Beds',
            'available': 'Available Beds',
            'occupancy_rate': 'Rate (%)'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Download forecast
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast (CSV)",
        data=csv,
        file_name=f"bed_forecast_{CURRENT_DATE.strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# 8. DISCHARGE PLANNING 
# ---------------------------------------------------------
elif menu == "Discharge Planning":
    st.title("📋 Discharge Planning & Predictions")
    
    if not model_loaded or not data_loaded:
        st.error("❌ Model or data not loaded.")
        st.stop()
    
    st.markdown('<div class="section-header">Patient Discharge Schedule</div>', unsafe_allow_html=True)
    
    # Prepare data
    df_discharge = df_eicu.sample(min(150, len(df_eicu))).copy()
    df_discharge['predicted_los_hours'] = df_discharge.apply(
        lambda row: predict_los_for_patient(row.to_dict(), model, scaler, le_dict, feature_cols),
        axis=1
    )
    
    np.random.seed(42)
    df_discharge['admission_time'] = CURRENT_DATE - pd.to_timedelta(
        np.random.uniform(1, 25, len(df_discharge)), unit='d'
    )
    df_discharge['predicted_discharge_time'] = (
        df_discharge['admission_time'] + 
        pd.to_timedelta(df_discharge['predicted_los_hours'], unit='h')
    )
    
    # Filter active patients
    active_discharge = df_discharge[df_discharge['predicted_discharge_time'] > CURRENT_DATE].copy()
    active_discharge['hours_remaining'] = (
        (active_discharge['predicted_discharge_time'] - CURRENT_DATE).dt.total_seconds() / 3600
    )
    
    # Group by discharge window
    col1, col2, col3 = st.columns(3)
    
    next_24h = len(active_discharge[active_discharge['hours_remaining'] <= 24])
    next_48h = len(active_discharge[active_discharge['hours_remaining'] <= 48])
    next_7d = len(active_discharge[active_discharge['hours_remaining'] <= 168])
    
    with col1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Next 24 Hours</div><div class="kpi-val" style="color:#FF6B6B">{next_24h}</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Next 48 Hours</div><div class="kpi-val" style="color:#FFD43B">{next_48h}</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Next 7 Days</div><div class="kpi-val" style="color:#51CF66">{next_7d}</div></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">🚨 URGENT: Discharging in Next 24 Hours</div>', unsafe_allow_html=True)
    
    urgent = active_discharge[active_discharge['hours_remaining'] <= 24].sort_values('hours_remaining')
    
    if not urgent.empty:
        for idx, (_, patient) in enumerate(urgent.head(10).iterrows(), 1):
            hours = patient['hours_remaining']
            discharge_time = patient['predicted_discharge_time'].strftime('%H:%M')
            
            if hours < 4:
                badge_color = "bg-crit"
                status = "IMMINENT"
            elif hours < 12:
                badge_color = "bg-warn"
                status = "SOON"
            else:
                badge_color = "bg-safe"
                status = "PLANNED"
            
            st.markdown(f"""
            <div class="dept-card">
                <div class="dept-header">
                    <span class="dept-title">Patient #{idx}</span>
                    <span class="badge {badge_color}">{status}</span>
                </div>
                <div style="font-size:12px; color:#8B949E; display:grid; grid-template-columns:1fr 1fr; gap:10px;">
                    <div><b>Discharge:</b> {discharge_time}</div>
                    <div><b>Hours Left:</b> {hours:.1f}h</div>
                    <div><b>Predicted LOS:</b> {patient['predicted_los_hours']:.1f}h</div>
                    <div><b>Admitted:</b> {patient['admission_time'].strftime('%H:%M')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if len(urgent) > 10:
            st.caption(f"... and {len(urgent) - 10} more patients")
    else:
        st.info("✓ No urgent discharges in next 24 hours")
    
    st.markdown("---")
    st.markdown('<div class="section-header">⏰ Next 48 Hours Schedule</div>', unsafe_allow_html=True)
    
    next_48 = active_discharge[
        (active_discharge['hours_remaining'] > 24) & 
        (active_discharge['hours_remaining'] <= 48)
    ].sort_values('hours_remaining')
    
    if not next_48.empty:
        display_cols = ['predicted_los_hours', 'hours_remaining']
        for col in ['age', 'gender', 'patientunitstayid']:
            if col in next_48.columns:
                display_cols.insert(0, col)
        
        display_df = next_48[display_cols].head(15).copy()
        display_df['hours_remaining'] = display_df['hours_remaining'].round(1)
        display_df['predicted_los_hours'] = display_df['predicted_los_hours'].round(1)
        
        st.dataframe(
            display_df.rename(columns={
                'patientunitstayid': 'Patient ID',
                'age': 'Age',
                'gender': 'Gender',
                'predicted_los_hours': 'LOS (h)',
                'hours_remaining': 'Hours Left'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.caption(f"Total patients in 24-48h window: {len(next_48)}")
    else:
        st.info("No patients scheduled for discharge in 24-48 hours")
    
    st.markdown("---")
    st.markdown('<div class="section-header">📅 Discharge Timeline (Next 7 Days)</div>', unsafe_allow_html=True)
    
    # Create daily discharge schedule
    active_discharge['discharge_date'] = active_discharge['predicted_discharge_time'].dt.date
    daily_discharge = active_discharge.groupby('discharge_date').size().reset_index(name='discharges')
    daily_discharge = daily_discharge.sort_values('discharge_date')
    
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(
        x=daily_discharge['discharge_date'].astype(str),
        y=daily_discharge['discharges'],
        name='Expected Discharges',
        marker=dict(color='#51CF66'),
        text=daily_discharge['discharges'],
        textposition='auto'
    ))
    fig_daily.update_layout(
        title="Daily Expected Discharges (Next 7 Days)",
        xaxis_title="Date",
        yaxis_title="Number of Discharges",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='#30363D'),
        yaxis=dict(showgrid=True, gridcolor='#30363D'),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_daily, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Discharge Distribution by Hour</div>', unsafe_allow_html=True)
    
    active_discharge['discharge_hour'] = active_discharge['predicted_discharge_time'].dt.hour
    hourly_discharge = active_discharge.groupby('discharge_hour').size().reset_index(name='count')
    hourly_discharge = hourly_discharge.sort_values('discharge_hour')
    
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Bar(
        x=hourly_discharge['discharge_hour'].astype(str),
        y=hourly_discharge['count'],
        name='Discharges',
        marker=dict(color='#58A6FF'),
        text=hourly_discharge['count'],
        textposition='auto'
    ))
    fig_hourly.update_layout(
        title="Expected Discharges by Hour of Day",
        xaxis_title="Hour",
        yaxis_title="Number of Discharges",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_hourly, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">🏥 Discharges by Unit Type</div>', unsafe_allow_html=True)
    
    if 'unittype' in active_discharge.columns:
        unit_discharge = active_discharge.groupby('unittype').agg({
            'hours_remaining': ['count', 'mean']
        }).round(1)
        unit_discharge.columns = ['total_patients', 'avg_hours']
        unit_discharge = unit_discharge.reset_index()
        unit_discharge['unit_name'] = unit_discharge['unittype'].apply(get_unit_type_name)
        
        fig_unit = go.Figure()
        fig_unit.add_trace(go.Bar(
            x=unit_discharge['unit_name'],
            y=unit_discharge['total_patients'],
            name='Patients',
            marker=dict(color='#FFD43B'),
            text=unit_discharge['total_patients'],
            textposition='auto'
        ))
        fig_unit.update_layout(
            title="Expected Discharges by Unit Type (Next 7 Days)",
            xaxis_title="Unit Type",
            yaxis_title="Number of Patients",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color='white'),
            xaxis=dict(tickangle=-45),
            margin=dict(l=0, r=0, t=30, b=80)
        )
        st.plotly_chart(fig_unit, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">💾 Export Discharge Plan</div>', unsafe_allow_html=True)
    
    export_df = active_discharge[[col for col in [
        'patientunitstayid', 'age', 'gender', 'predicted_los_hours', 
        'admission_time', 'predicted_discharge_time', 'hours_remaining'
    ] if col in active_discharge.columns]].copy()
    
    export_df['admission_time'] = export_df['admission_time'].dt.strftime('%Y-%m-%d %H:%M')
    export_df['predicted_discharge_time'] = export_df['predicted_discharge_time'].dt.strftime('%Y-%m-%d %H:%M')
    
    csv = export_df.to_csv(index=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download Discharge Plan (CSV)",
            data=csv,
            file_name=f"discharge_plan_{CURRENT_DATE.strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.metric("Total Patients in Plan", len(active_discharge))
# ---------------------------------------------------------
# 5. Live Admissions (Real-time Prediction)
# ---------------------------------------------------------
elif menu == "Live Admissions":
    st.title("🏥 New Patient Intake")
    st.markdown("Modify the clinical fields below to see the **AI Predicted LOS** update in real-time.")

    if not data_loaded or df_eicu is None:
        st.error("Critical: Base eICU data not loaded. Check Settings.")
        st.stop()

    # --- INPUT SECTION ---
    st.markdown('<div class="section-header">Demographic & Unit Assignment</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        patientunitstayid = st.number_input("Patient Unit Stay ID", min_value=1, step=1, value=12345)
        hospitalid = st.number_input("Hospital ID", min_value=1, step=1, value=1)
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
        age = st.text_input("Age (e.g., '60' or '> 89')", value="60")
    with c3:
        unittype_options = sorted(df_eicu["unittype"].dropna().astype(str).unique().tolist())
        unittype = st.selectbox("Assigned Unit Type", unittype_options)

    st.markdown('<div class="section-header">Timing Metrics</div>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        unitadmitoffset = st.number_input("Unit Admit Offset (min)", value=0)
    with c5:
        unitdischargeoffset = st.number_input("Unit Discharge Offset (min)", value=1440)
    with c6:
        unitdischargelocation = st.selectbox("Discharge Destination", ["Home", "Skilled Nursing Facility", "Rehabilitation", "Other"])

    # --- REAL-TIME PREDICTION LOGIC ---
    current_patient_data = {
        "patientunitstayid": int(patientunitstayid),
        "hospitalid": int(hospitalid),
        "age": age,
        "gender": gender,
        "unittype": unittype,
        "unitadmitoffset": unitadmitoffset,
        "unitdischargeoffset": unitdischargeoffset,
        "hospitaldischargeoffset": unitdischargeoffset + 60,
        "unitdischargelocation": unitdischargelocation,
    }

    st.markdown("---")
    
    # Calculate LOS immediately as inputs change
    if model_loaded:
        with st.spinner("AI is calculating LOS forecast..."):
            rt_pred_los = predict_los_for_patient(current_patient_data, model, scaler, le_dict, feature_cols)
    else:
        rt_pred_los = None

    # --- DISPLAY PREDICTION BEFORE BUTTON ---
    if rt_pred_los:
        # Calculate Days and Remaining Hours
        days = int(rt_pred_los // 24)
        remaining_hours = rt_pred_los % 24
        
        # Create a formatted string for days/hours
        day_str = f"{days} Day{'s' if days != 1 else ''}" if days > 0 else ""
        hour_str = f"{remaining_hours:.1f} Hour{'s' if remaining_hours != 1 else ''}"
        full_duration_str = f"{day_str}, {hour_str}".strip(", ")

        st.markdown(f"""
        <div class="ai-container">
            <div class="ai-header">🤖 AI PREDICTION PREVIEW</div>
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <div style="font-size: 24px; font-weight: bold; color: #A371F7;">
                        {rt_pred_los:.1f} Total Hours
                    </div>
                    <div style="font-size: 16px; color: #E6EDF3; margin-top: 4px;">
                        ⏱️ {full_duration_str}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 13px; color: #8B949E;">Estimated Discharge:</div>
                    <div style="font-size: 14px; font-weight: 600; color: #58A6FF;">
                        {(CURRENT_DATE + timedelta(hours=rt_pred_los)).strftime('%Y-%m-%d %H:%M')}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("AI Prediction unavailable. Please check model settings.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- FINAL SUBMISSION ---
    if st.button("➕ Register Patient & Save to eICU File", type="primary", use_container_width=True):
        try:
            # Add the predicted value to the row before saving
            current_patient_data["icu_los_hours"] = rt_pred_los if rt_pred_los else (unitdischargeoffset / 60.0)
            
            df_updated = append_patient_row(current_patient_data)
            st.success(f"Patient registered. Database now has {len(df_updated)} records.")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Error saving patient: {e}")
# ---------------------------------------------------------
# 9. SETTINGS
# ---------------------------------------------------------
elif menu == "Settings":
    st.title("⚙️ System Settings & Configuration")
    
    st.markdown('<div class="section-header">Model Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if model_loaded:
            st.success("✓ Machine Learning Model Loaded")
            st.caption("best_model.pkl")
        else:
            st.error("❌ Model Not Loaded")
            st.caption("Place best_model.pkl in working directory")
    
    with col2:
        if data_loaded:
            st.success(f"✓ eICU Data Loaded ({len(df_eicu)} patients)")
        else:
            st.error("❌ Data Not Loaded")
    
    st.markdown("---")
    st.markdown('<div class="section-header">Data Components Status</div>', unsafe_allow_html=True)
    
    components_status = {
        'best_model.pkl': model_loaded,
        'scaler.pkl': model_loaded,
        'label_encoders.pkl': model_loaded,
        'feature_columns.pkl': model_loaded
    }
