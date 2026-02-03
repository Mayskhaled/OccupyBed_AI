import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import time
import os
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import base64

# ---------------------------------------------------------
# 1. System Config & Design
# ---------------------------------------------------------
st.set_page_config(page_title="OccupyBed AI Pro", layout="wide", page_icon="🏥")

# --- GLOBAL TIME FREEZE ---
CURRENT_DATE = datetime(2026, 1, 8, 12, 0, 0)

# Custom CSS for the combined application
st.markdown("""
<style>
    /* Global Settings */
    .stApp { background-color: #0E1117; color: #E6EDF3; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background-color: #010409; border-right: 1px solid #30363D; }
    
    /* --- GLOWING LOGO --- */
    @keyframes glow {
        from { text-shadow: 0 0 5px #fff, 0 0 10px #58A6FF; }
        to { text-shadow: 0 0 10px #fff, 0 0 20px #58A6FF; }
    }
    .logo-box { text-align: center; margin-bottom: 30px; margin-top: 10px; }
    .logo-main { 
        font-size: 28px; 
        font-weight: 800; 
        color: #FFFFFF; 
        animation: glow 2s infinite alternate; 
        margin: 0; 
        letter-spacing: 1px;
    }
    .logo-slogan { 
        font-size: 10px; 
        color: #8B949E; 
        text-transform: uppercase; 
        letter-spacing: 2px; 
        margin-top: 5px; 
        font-weight: 500;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: #161B22; border: 1px solid #30363D; border-radius: 6px;
        padding: 20px; text-align: center; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .kpi-label { font-size: 11px; color: #8B949E; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .kpi-val { font-size: 28px; font-weight: 700; color: #FFF; margin: 0; }
    .kpi-sub { font-size: 11px; color: #58A6FF; margin-top: 5px;}
    
    /* Section Headers */
    .section-header {
        font-size: 16px; font-weight: 700; color: #E6EDF3; 
        margin-top: 25px; margin-bottom: 15px; 
        border-left: 4px solid #58A6FF; padding-left: 10px;
    }

    /* AI Board */
    .ai-container {
        background-color: #161B22; border: 1px solid #30363D; border-left: 5px solid #A371F7;
        border-radius: 6px; padding: 15px; height: 100%;
    }
    .ai-header { font-weight: 700; color: #A371F7; font-size: 14px; margin-bottom: 10px; text-transform: uppercase; }
    .ai-item { font-size: 13px; color: #E6EDF3; margin-bottom: 6px; border-bottom: 1px solid #21262D; padding-bottom: 4px; }

    /* Department Cards */
    .dept-card {
        background-color: #0D1117; border: 1px solid #30363D; border-radius: 6px;
        padding: 15px; margin-bottom: 12px;
    }
    .dept-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    .dept-title { font-size: 14px; font-weight: 700; color: #FFF; }
    
    /* Status Badges */
    .badge { padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; text-transform: uppercase; }
    .bg-safe { background: rgba(35, 134, 54, 0.2); color: #3FB950; border: 1px solid #238636; }
    .bg-warn { background: rgba(210, 153, 34, 0.2); color: #D29922; border: 1px solid #9E6A03; }
    .bg-crit { background: rgba(218, 54, 51, 0.2); color: #F85149; border: 1px solid #DA3633; }
    .bg-empty { background: rgba(48, 54, 61, 0.2); color: #8B949E; border: 1px solid #30363D; }

    /* Inputs */
    div[data-baseweb="select"] > div, input { background-color: #0D1117 !important; border-color: #30363D !important; color: white !important; }
    button[kind="primary"] { background-color: #238636 !important; border: none !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Load Models and Data
# ---------------------------------------------------------
@st.cache_resource
def load_model_components():
    """Load trained Random Forest model, scaler, and encoders"""
    try:
        app_dir = Path(__file__).parent
        
        # Define file paths
        model_path = app_dir / 'best_model.pkl'
        scaler_path = app_dir / 'scaler.pkl'
        le_path = app_dir / 'label_encoders.pkl'
        cols_path = app_dir / 'feature_columns.pkl'
        
        # Check if all files exist
        files_to_check = [
            ('Model', model_path),
            ('Scaler', scaler_path),
            ('Label Encoders', le_path),
            ('Feature Columns', cols_path)
        ]
        
        for name, path in files_to_check:
            if not path.exists():
                st.error(f"❌ Missing {name}: {path}")
                return None, None, None, None, False
        
        # Load files
        model = joblib.load(str(model_path))
        scaler = joblib.load(str(scaler_path))
        le_dict = joblib.load(str(le_path))
        feature_cols = joblib.load(str(cols_path))
        
        return model, scaler, le_dict, feature_cols, True
        
    except Exception as e:
        st.error(f"❌ Error loading components: {str(e)}")
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
        
        return df, True
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, False

def append_patient_row(new_row: dict):
    """Append a new patient record to patient.csv and return updated df"""
    app_dir = Path(__file__).parent
    csv_path = app_dir / 'patient.csv'
    
    # Load current data
    df, ok = load_eicu_data()
    if not ok or df is None:
        df = pd.DataFrame()

    # Ensure we respect existing columns
    if not df.empty:
        base_cols = df.columns.tolist()
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

def get_base64_image(image_path):
    """Convert image to base64 for HTML display"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# ---------------------------------------------------------
# 3. Traditional Hospital System Data (from app design.py)
# ---------------------------------------------------------
DEPARTMENTS = {
    "Medical Male": {"cap": 50, "gen": "Male", "overflow": "Surgical Male"},
    "Medical Female": {"cap": 50, "gen": "Female", "overflow": "Surgical Female"},
    "Surgical Male": {"cap": 40, "gen": "Male", "overflow": "Medical Male"},
    "Surgical Female": {"cap": 40, "gen": "Female", "overflow": "Medical Female"},
    "ICU": {"cap": 16, "gen": "Mixed", "overflow": "HDU"},
    "Pediatric": {"cap": 30, "gen": "Mixed", "overflow": "None"},
    "Obstetrics": {"cap": 24, "gen": "Female", "overflow": "Gynae"},
}

PATIENT_DB = {f"PIN-{1000+i}": ("Male" if i % 2 == 0 else "Female") for i in range(3000)}

def init_system():
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=[
            "PIN", "Gender", "Department", "Bed", 
            "Admit_Date", "Exp_Discharge", "Actual_Discharge", "Source"
        ])
        
        # Generate Data Relative to CURRENT_DATE
        data = []
        for dept, info in DEPARTMENTS.items():
            count = int(info['cap'] * np.random.uniform(0.4, 0.65))
            for i in range(count):
                bed_n = f"{dept[:3].upper()}-{i+1:03d}"
                
                days_ago = np.random.randint(1, 10)
                adm = CURRENT_DATE - timedelta(days=days_ago, hours=np.random.randint(1, 12))
                
                exp = adm + timedelta(days=np.random.randint(3, 8))
                
                actual_dis = pd.NaT
                if np.random.random() < 0.3 and days_ago > 4:
                    actual_dis = adm + timedelta(days=np.random.randint(2, 4))
                
                data.append({
                    "PIN": f"PIN-{np.random.randint(2000, 9999)}",
                    "Gender": "Female" if "Female" in dept else ("Male" if "Male" in dept else np.random.choice(["Male", "Female"])),
                    "Department": dept,
                    "Bed": bed_n,
                    "Admit_Date": adm,
                    "Exp_Discharge": exp,
                    "Actual_Discharge": actual_dis,
                    "Source": "Emergency"
                })
        st.session_state.df = pd.DataFrame(data)

init_system()
df = st.session_state.df

# Enforce Date Types
for col in ['Admit_Date', 'Exp_Discharge', 'Actual_Discharge']:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# ---------------------------------------------------------
# 4. ML Functions (from app functionality.py)
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
        
        # 5. Scale and RECONSTRUCT DataFrame with names
        new_patient_scaled_array = scaler.transform(new_patient)
        new_patient_scaled_df = pd.DataFrame(new_patient_scaled_array, columns=feature_cols)
        
        # 6. Predict using the fitted model
        predicted_los = float(model.predict(new_patient_scaled_df)[0])
        return max(predicted_los, 1.0) 
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
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

def get_unit_type_name(unit_type):
    """Map unit type codes to names from eICU data"""
    try:
        return str(unit_type)
    except:
        return f"Unit {unit_type}"

# ---------------------------------------------------------
# 5. Sidebar Navigation (Combined)
# ---------------------------------------------------------
with st.sidebar:
    # Logo Section
    try:
        logo_base64 = get_base64_image("logo.png")
        if logo_base64:
            logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="300" style="margin-bottom: 10px; border-radius: 8px;">'
        else:
            logo_html = ""
    except:
        logo_html = ""
    
    if not logo_html and os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    
    st.markdown(f"""
    <div class="logo-box">
        {logo_html}
        <div class="logo-main">OccupyBed AI Pro</div>
        <div class="logo-slogan">Intelligent Hospital Bed Management</div>
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
    
    # Combined Navigation Menu
    menu = st.radio("NAVIGATION", [
        "Overview", 
        "Traditional Dashboard", 
        "ML Dashboard", 
        "Patient Analytics", 
        "Bed Forecast", 
        "Discharge Planning", 
        "Live Admissions", 
        "Operational Analytics", 
        "Settings"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    
    # Model Status
    model, scaler, le_dict, feature_cols, model_loaded = load_model_components()
    df_eicu, data_loaded = load_eicu_data()
    
    if model_loaded and data_loaded:
        st.success("✓ System Ready")
    else:
        st.error("⚠ System Incomplete")
        if not model_loaded:
            st.caption("Model files missing")
        if not data_loaded:
            st.caption("Data loading failed")
    
    st.caption(f"Reference: {CURRENT_DATE.strftime('%Y-%m-%d %H:%M')}")

# ---------------------------------------------------------
# 6. OVERVIEW (Combined Dashboard)
# ---------------------------------------------------------
if menu == "Overview":
    st.title("🏥 Hospital Command Center")
    
    fc_hours = st.selectbox("Forecast Window", [6, 12, 24, 48, 72], index=2, format_func=lambda x: f"{x} Hours")
    
    # Traditional Metrics
    active_df = df[df['Actual_Discharge'].isna()]
    future_limit = CURRENT_DATE + timedelta(hours=fc_hours)
    
    total_cap = sum(d['cap'] for d in DEPARTMENTS.values())
    occ_count = len(active_df)
    avail_count = total_cap - occ_count
    ready_count = len(active_df[active_df['Exp_Discharge'] <= future_limit])
    
    # ML Metrics (if loaded)
    ml_ready = False
    if model_loaded and data_loaded:
        ml_ready = True
        df_sample = df_eicu.sample(min(100, len(df_eicu))).copy()
        df_sample['predicted_los_hours'] = df_sample.apply(
            lambda row: predict_los_for_patient(row.to_dict(), model, scaler, le_dict, feature_cols),
            axis=1
        )
        np.random.seed(42)
        df_sample['admission_time'] = CURRENT_DATE - pd.to_timedelta(
            np.random.uniform(1, 30, len(df_sample)), unit='d'
        )
        df_sample['predicted_discharge_time'] = (
            df_sample['admission_time'] + 
            pd.to_timedelta(df_sample['predicted_los_hours'], unit='h')
        )
        ml_active = df_sample[df_sample['predicted_discharge_time'] > CURRENT_DATE].copy()
        ml_occ = len(ml_active)
        ml_ready_count = len(ml_active[ml_active['predicted_discharge_time'] <= future_limit])
    
    # Top Row: Combined KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1: 
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total Beds</div><div class="kpi-val" style="color:#58A6FF">{total_cap}</div><div class="kpi-sub">Traditional System</div></div>""", unsafe_allow_html=True)
    with k2: 
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Occupied</div><div class="kpi-val" style="color:#D29922">{occ_count}</div><div class="kpi-sub">Currently Active</div></div>""", unsafe_allow_html=True)
    with k3: 
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Available Now</div><div class="kpi-val" style="color:#3FB950">{avail_count}</div><div class="kpi-sub">Ready for Admission</div></div>""", unsafe_allow_html=True)
    with k4: 
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Will Free ({fc_hours}h)</div><div class="kpi-val" style="color:#A371F7">{ready_count}</div><div class="kpi-sub">Expected Discharges</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gauge & AI Recommendations
    g_col, ai_col = st.columns([1, 2])
    
    with g_col:
        occ_rate = (occ_count / total_cap) * 100 if total_cap > 0 else 0
        fig = go.Figure(go.Indicator(
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
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=0,b=0), 
                         paper_bgcolor="#0E1117", font={'color': "white"})
        st.plotly_chart(fig, use_container_width=True)
    
    with ai_col:
        st.markdown("""<div class="ai-container"><div class="ai-header">🤖 AI Operational Insights</div>""", unsafe_allow_html=True)
        
        ai_triggered = False
        for dept, info in DEPARTMENTS.items():
            d_pats = active_df[active_df['Department'] == dept]
            pct = (len(d_pats) / info['cap']) * 100
            
            if pct >= 85:
                st.markdown(f"""<div class="ai-item"><span style="color:#F85149"><b>{dept}:</b></span> Critical load ({int(pct)}%). Activate surge protocol.</div>""", unsafe_allow_html=True)
                ai_triggered = True
            elif pct >= 70:
                st.markdown(f"""<div class="ai-item"><span style="color:#D29922"><b>{dept}:</b></span> High Load ({int(pct)}%). Prioritize pending discharges.</div>""", unsafe_allow_html=True)
                ai_triggered = True
        
        if ml_ready:
            avg_los = ml_active['predicted_los_hours'].mean() if len(ml_active) > 0 else 0
            st.markdown(f"""<div class="ai-item"><b>ML Predicted Avg LOS:</b> {avg_los:.1f} hours</div>""", unsafe_allow_html=True)
            
            if len(ml_active) > 0:
                next_discharge = ml_active['predicted_discharge_time'].min()
                hours_until = (next_discharge - CURRENT_DATE).total_seconds() / 3600
                st.markdown(f"""<div class="ai-item"><b>Next ML Predicted Bed Free:</b> {max(0, hours_until):.1f} hours</div>""", unsafe_allow_html=True)
        
        if not ai_triggered:
            st.markdown("""<div class="ai-item" style="color:#3FB950">Hospital capacity is optimal. No bottlenecks detected.</div>""", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Department Status (Traditional)
    st.markdown("### Department Live Status")
    d_cols = st.columns(3)
    for i, (dept, info) in enumerate(DEPARTMENTS.items()):
        d_df = active_df[active_df['Department'] == dept]
        occ = len(d_df)
        avail = info['cap'] - occ
        ready = len(d_df[d_df['Exp_Discharge'] <= future_limit])
        pct = (occ / info['cap']) * 100
        
        if pct < 70: status, cls, bar = "SAFE", "bg-safe", "#3FB950"
        elif 70 <= pct <= 84: status, cls, bar = "WARNING", "bg-warn", "#D29922"
        else: status, cls, bar = "CRITICAL", "bg-crit", "#F85149"
        
        with d_cols[i % 3]:
            st.markdown(f"""
            <div class="dept-card">
                <div class="dept-header">
                    <span class="dept-title">{dept}</span>
                    <span class="badge {cls}">{status}</span>
                </div>
                <div style="font-size:12px; color:#8B949E; display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span>Cap: {info['cap']}</span>
                    <span>Occ: <b style="color:#E6EDF3">{occ}</b></span>
                    <span>Avail: {avail}</span>
                </div>
                <div style="font-size:12px; display:flex; justify-content:space-between;">
                    <span style="color:#A371F7; font-weight:bold;">Forecast Free ({fc_hours}h): {ready}</span>
                </div>
                <div style="background:#21262D; height:5px; border-radius:3px; margin-top:8px;">
                    <div style="width:{min(pct, 100)}%; background:{bar}; height:100%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 7. TRADITIONAL DASHBOARD (from app design.py)
# ---------------------------------------------------------
elif menu == "Traditional Dashboard":
    c1, c2 = st.columns([3, 1])
    with c1: st.title("Traditional Hospital Dashboard")
    with c2: 
        fc_hours = st.selectbox("Forecast Window", [6, 12, 24, 48, 72], index=2, format_func=lambda x: f"{x} Hours")

    # Metrics
    active_df = df[df['Actual_Discharge'].isna()]
    future_limit = CURRENT_DATE + timedelta(hours=fc_hours)
    
    total_cap = sum(d['cap'] for d in DEPARTMENTS.values())
    occ_count = len(active_df)
    avail_count = total_cap - occ_count
    ready_count = len(active_df[active_df['Exp_Discharge'] <= future_limit])

    # KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total Licensed Beds</div><div class="kpi-val" style="color:#58A6FF">{total_cap}</div></div>""", unsafe_allow_html=True)
    with k2: st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Occupied Beds</div><div class="kpi-val" style="color:#D29922">{occ_count}</div></div>""", unsafe_allow_html=True)
    with k3: st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Available Now</div><div class="kpi-val" style="color:#3FB950">{avail_count}</div></div>""", unsafe_allow_html=True)
    with k4: st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Expected Free ({fc_hours}h)</div><div class="kpi-val" style="color:#A371F7">{ready_count}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Department Status
    st.markdown("### Department Live Status")
    d_cols = st.columns(3)
    for i, (dept, info) in enumerate(DEPARTMENTS.items()):
        d_df = active_df[active_df['Department'] == dept]
        occ = len(d_df)
        avail = info['cap'] - occ
        ready = len(d_df[d_df['Exp_Discharge'] <= future_limit])
        pct = (occ / info['cap']) * 100
        
        if pct < 70: status, cls, bar = "SAFE", "bg-safe", "#3FB950"
        elif 70 <= pct <= 84: status, cls, bar = "WARNING", "bg-warn", "#D29922"
        else: status, cls, bar = "CRITICAL", "bg-crit", "#F85149"
        
        with d_cols[i % 3]:
            st.markdown(f"""
            <div class="dept-card">
                <div class="dept-header">
                    <span class="dept-title">{dept}</span>
                    <span class="badge {cls}">{status}</span>
                </div>
                <div style="font-size:12px; color:#8B949E; display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span>Cap: {info['cap']}</span>
                    <span>Occ: <b style="color:#E6EDF3">{occ}</b></span>
                    <span>Avail: {avail}</span>
                </div>
                <div style="font-size:12px; display:flex; justify-content:space-between;">
                    <span style="color:#A371F7; font-weight:bold;">Forecast Free ({fc_hours}h): {ready}</span>
                </div>
                <div style="background:#21262D; height:5px; border-radius:3px; margin-top:8px;">
                    <div style="width:{min(pct, 100)}%; background:{bar}; height:100%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 8. ML DASHBOARD (from app functionality.py)
# ---------------------------------------------------------
elif menu == "ML Dashboard":
    st.title("🤖 ML-Powered Dashboard")
    
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
            title={'text': "ML Hospital Pressure"},
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
    
    # ICU Units - Live Status
    st.markdown('<div class="section-header">ICU Units - Live Status</div>', unsafe_allow_html=True)
    
    UNIT_CAPS = {
        "MICU": 25,
        "SICU": 20,
        "Med-Surg ICU": 30,
        "CCU-CTICU": 15,
        "CSICU": 12,
        "Neuro ICU": 10,
        "Burn ICU": 8,
        "Trauma ICU": 12
    }

    all_units = list(UNIT_CAPS.keys())
    unit_cols = st.columns(3)
    
    for i, unit_id in enumerate(all_units):
        unit_df = active_patients[active_patients['unittype'] == unit_id]
        
        occ = len(unit_df)
        cap = UNIT_CAPS[unit_id]
        avail = max(0, cap - occ)
        
        future_limit = CURRENT_DATE + timedelta(hours=fc_hours)
        ready = len(unit_df[unit_df['predicted_discharge_time'] <= future_limit])
        
        pct = (occ / cap) * 100 if cap > 0 else 0
        
        if occ == 0:
            status, cls, bar_color = "EMPTY", "bg-empty", "#30363D"
        elif pct < 70: 
            status, cls, bar_color = "SAFE", "bg-safe", "#3FB950"
        elif 70 <= pct <= 85: 
            status, cls, bar_color = "WARNING", "bg-warn", "#D29922"
        else: 
            status, cls, bar_color = "CRITICAL", "bg-crit", "#F85149"
        
        unit_display_name = get_unit_type_name(unit_id)
            
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
# 9. PATIENT ANALYTICS (from app functionality.py)
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
# 10. BED FORECAST (from app functionality.py)
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

# ---------------------------------------------------------
# 11. DISCHARGE PLANNING (from app functionality.py)
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

# ---------------------------------------------------------
# 12. LIVE ADMISSIONS (Combined)
# ---------------------------------------------------------
elif menu == "Live Admissions":
    st.title("Patient Admission & Discharge Center")
    
    tab1, tab2 = st.tabs(["Traditional Admissions", "ML-Powered ICU Admissions"])
    
    with tab1:
        # 1. Data Management
        with st.expander("Data Operations (Import / Export)", expanded=False):
            c_dl, c_ul = st.columns(2)
            with c_dl:
                st.download_button("Download Database (CSV)", df.to_csv(index=False).encode('utf-8'), "hospital_db.csv", "text/csv")
            with c_ul:
                up_file = st.file_uploader("Upload Data (CSV)", type=['csv'])
                if up_file:
                    try:
                        new_df = pd.read_csv(up_file)
                        for col in ['Admit_Date', 'Exp_Discharge', 'Actual_Discharge']: 
                            new_df[col] = pd.to_datetime(new_df[col], errors='coerce')
                        st.session_state.df = new_df
                        st.success("Data Loaded.")
                        st.rerun()
                    except: st.error("Invalid File")

        # 2. Admission Form
        st.subheader("New Admission")
        c1, c2 = st.columns(2)
        with c1:
            # Filter: Only show PINs that are NOT currently admitted
            active_pins = df[df['Actual_Discharge'].isna()]['PIN'].tolist()
            all_pins = list(PATIENT_DB.keys())
            valid_pins = [p for p in all_pins if p not in active_pins]
            
            pin = st.selectbox("Select Patient PIN", ["Select..."] + valid_pins)
            gender = "Unknown"
            if pin != "Select...":
                gender = PATIENT_DB.get(pin, "Unknown")
                st.info(f"Gender: **{gender}**")
            
            dept = st.selectbox("Assign Department", ["Select..."] + list(DEPARTMENTS.keys()))
            
            bed_opts = ["Select Dept"]
            if dept != "Select...":
                occ_beds = df[(df['Department'] == dept) & (df['Actual_Discharge'].isna())]['Bed'].tolist()
                all_beds = [f"{dept[:3].upper()}-{i+1:03d}" for i in range(DEPARTMENTS[dept]['cap'])]
                free_beds = [b for b in all_beds if b not in occ_beds]
                bed_opts = free_beds if free_beds else ["NO BEDS AVAILABLE"]
            bed = st.selectbox("Assign Bed", bed_opts)

        with c2:
            d1, t1 = st.columns(2)
            adm_d = d1.date_input("Admit Date", CURRENT_DATE)
            adm_t = t1.time_input("Admit Time", CURRENT_DATE.time())
            d2, t2 = st.columns(2)
            exp_d = d2.date_input("Exp Discharge Date", CURRENT_DATE + timedelta(days=3))
            exp_t = t2.time_input("Exp Discharge Time", CURRENT_DATE.time())
            src = st.selectbox("Source", ["Emergency", "Elective", "Transfer"])

        if st.button("Confirm Admission", type="primary"):
            if pin == "Select..." or dept == "Select..." or bed in ["Select Dept", "NO BEDS AVAILABLE"]:
                st.warning("Please complete all fields.")
            elif DEPARTMENTS[dept]['gen'] != "Mixed" and DEPARTMENTS[dept]['gen'] != gender:
                st.error(f"Error: Gender Mismatch. {dept} is for {DEPARTMENTS[dept]['gen']} only.")
            else:
                new_rec = {
                    "PIN": pin, "Gender": gender, "Department": dept, "Bed": bed,
                    "Admit_Date": datetime.combine(adm_d, adm_t),
                    "Exp_Discharge": datetime.combine(exp_d, exp_t),
                    "Actual_Discharge": pd.NaT,
                    "Source": src
                }
                st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_rec])], ignore_index=True)
                st.success("Admitted Successfully.")
                time.sleep(0.5)
                st.rerun()

        st.markdown("---")

        # 3. Patient Management
        st.subheader("Patient Management (Update / Discharge)")
        active_df = df[df['Actual_Discharge'].isna()].sort_values(by="Admit_Date", ascending=False)
        
        if not active_df.empty:
            target = st.selectbox("Select Patient to Manage", ["Select..."] + active_df['PIN'].tolist())
            
            if target != "Select...":
                p_idx = df[(df['PIN'] == target) & (df['Actual_Discharge'].isna())].index[0]
                p_row = df.loc[p_idx]
                
                st.info(f"Managing: **{target}** | Dept: **{p_row['Department']}** | Bed: **{p_row['Bed']}**")
                
                tab_up, tab_dis = st.tabs(["Edit Expected Discharge", "Discharge Patient"])
                
                with tab_up:
                    c_up1, c_up2 = st.columns(2)
                    new_exp_d = c_up1.date_input("New Expected Date", p_row['Exp_Discharge'])
                    new_exp_t = c_up2.time_input("New Expected Time", p_row['Exp_Discharge'].time())
                    if st.button("Update Information"):
                        st.session_state.df.at[p_idx, 'Exp_Discharge'] = datetime.combine(new_exp_d, new_exp_t)
                        st.success("Record Updated.")
                        st.rerun()
                
                with tab_dis:
                    c_d1, c_d2 = st.columns(2)
                    act_d = c_d1.date_input("Actual Discharge Date", CURRENT_DATE)
                    act_t = c_d2.time_input("Actual Discharge Time", CURRENT_DATE.time())
                    if st.button("Confirm Discharge", type="primary"):
                        st.session_state.df.at[p_idx, 'Actual_Discharge'] = datetime.combine(act_d, act_t)
                        st.success(f"Patient {target} Discharged.")
                        time.sleep(0.5)
                        st.rerun()
                        
            st.dataframe(active_df[['PIN', 'Department', 'Bed', 'Admit_Date', 'Exp_Discharge']], use_container_width=True)
        else:
            st.info("No active patients.")
    
    with tab2:
        st.markdown("### 🏥 New ICU Patient Intake")
        
        if not data_loaded or df_eicu is None:
            st.error("Critical: Base eICU data not loaded. Check Settings.")
            st.stop()

        # Input Section
        st.markdown('<div class="section-header">Demographic & Unit Assignment</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            patientunitstayid = st.number_input("Patient Unit Stay ID", min_value=1, step=1, value=12345, key="ml_id")
            hospitalid = st.number_input("Hospital ID", min_value=1, step=1, value=1, key="ml_hosp")
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female", "Unknown"], key="ml_gender")
            age = st.text_input("Age (e.g., '60' or '> 89')", value="60", key="ml_age")
        with c3:
            if data_loaded:
                unittype_options = sorted(df_eicu["unittype"].dropna().astype(str).unique().tolist())
                unittype = st.selectbox("Assigned Unit Type", unittype_options, key="ml_unit")
            else:
                unittype = st.text_input("Unit Type", value="ICU")

        st.markdown('<div class="section-header">Timing Metrics</div>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            unitadmitoffset = st.number_input("Unit Admit Offset (min)", value=0, key="ml_admit")
        with c5:
            unitdischargeoffset = st.number_input("Unit Discharge Offset (min)", value=1440, key="ml_discharge")
        with c6:
            unitdischargelocation = st.selectbox("Discharge Destination", ["Home", "Skilled Nursing Facility", "Rehabilitation", "Other"], key="ml_dest")

        # Real-time Prediction
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

        # Display Prediction
        if rt_pred_los:
            days = int(rt_pred_los // 24)
            remaining_hours = rt_pred_los % 24
            
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

        # Final Submission
        if st.button("➕ Register ICU Patient & Save to Database", type="primary", use_container_width=True):
            try:
                current_patient_data["icu_los_hours"] = rt_pred_los if rt_pred_los else (unitdischargeoffset / 60.0)
                
                df_updated = append_patient_row(current_patient_data)
                st.success(f"Patient registered. Database now has {len(df_updated)} records.")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error saving patient: {e}")

# ---------------------------------------------------------
# 13. OPERATIONAL ANALYTICS (from app design.py)
# ---------------------------------------------------------
elif menu == "Operational Analytics":
    st.title("Operational Analytics")
    calc = df.copy()
    
    # --- 1. CALCULATE KPIs (Hospital Level) ---
    if not calc.empty:
        min_date = calc['Admit_Date'].min()
        max_date = CURRENT_DATE
        days_range = (max_date - min_date).days
        if days_range < 1: days_range = 1
    else:
        days_range = 1
        
    total_dis = len(calc[calc['Actual_Discharge'].notnull()])
    total_cap = sum(d['cap'] for d in DEPARTMENTS.values())
    
    # BOR
    active = calc[calc['Actual_Discharge'].isna()]
    h_bor = (len(active) / total_cap) * 100
    
    # ALOS
    discharged_only = calc[calc['Actual_Discharge'].notnull()]
    h_alos = 0
    if not discharged_only.empty:
        h_alos = (discharged_only['Actual_Discharge'] - discharged_only['Admit_Date']).dt.total_seconds().mean() / 86400
        
    # BTR & BTI
    h_btr = total_dis / total_cap
    
    calc['Discharge_Calc'] = calc['Actual_Discharge'].fillna(CURRENT_DATE)
    calc['Patient_Days'] = (calc['Discharge_Calc'] - calc['Admit_Date']).dt.total_seconds() / 86400
    total_pat_days = calc['Patient_Days'].sum()
    avail_bed_days = (total_cap * days_range) - total_pat_days
    h_bti = avail_bed_days / total_dis if total_dis > 0 else 0

    # --- 2. DISPLAY KPIs ---
    st.markdown('<div class="section-header">Operational KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    def kpi_box(lbl, val, unit):
        return f"""<div class="kpi-card"><div class="kpi-label">{lbl}</div><div class="kpi-val" style="font-size:24px;">{val}</div><div class="kpi-sub">{unit}</div></div>"""
        
    k1.markdown(kpi_box("BOR", f"{h_bor:.1f}", "%"), unsafe_allow_html=True)
    k2.markdown(kpi_box("ALOS", f"{h_alos:.1f}", "Days"), unsafe_allow_html=True)
    k3.markdown(kpi_box("BTR", f"{h_btr:.2f}", "Times / Month"), unsafe_allow_html=True)
    k4.markdown(kpi_box("BTI", f"{h_bti:.1f}", "Days"), unsafe_allow_html=True)

    st.markdown("---")
    
    # --- 3. TREND CHART (Line Chart with Markers) ---
    st.markdown('<div class="section-header">Admissions vs Discharges (Operational Trend)</div>', unsafe_allow_html=True)
    
    # Aggregate data by Date
    daily_adm = calc.groupby(calc['Admit_Date'].dt.date).size().reset_index(name='Admissions')
    
    if not discharged_only.empty:
        daily_dis = discharged_only.groupby(discharged_only['Actual_Discharge'].dt.date).size().reset_index(name='Discharges')
    else:
        daily_dis = pd.DataFrame(columns=['Actual_Discharge', 'Discharges'])

    # Merge and fill NaN with 0
    if not daily_adm.empty:
        trend = pd.merge(daily_adm, daily_dis, left_on='Admit_Date', right_on='Actual_Discharge', how='outer')
        trend['Admit_Date'] = trend['Admit_Date'].fillna(trend['Actual_Discharge'])
        trend = trend.drop(columns=['Actual_Discharge']).fillna(0).sort_values('Admit_Date')
        
        # Filter for the last 7-10 days leading up to CURRENT_DATE
        mask = (trend['Admit_Date'] <= CURRENT_DATE.date()) & (trend['Admit_Date'] >= (CURRENT_DATE - timedelta(days=10)).date())
        trend = trend.loc[mask]

        # Plotly Line Chart
        fig_trend = go.Figure()
        
        # Line 1: Admissions (Blue)
        fig_trend.add_trace(go.Scatter(
            x=trend['Admit_Date'], y=trend['Admissions'],
            mode='lines+markers',
            name='Admissions',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ))
        
        # Line 2: Discharges (Orange)
        fig_trend.add_trace(go.Scatter(
            x=trend['Admit_Date'], y=trend['Discharges'],
            mode='lines+markers',
            name='Discharges',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=8)
        ))
        
        fig_trend.update_layout(
            paper_bgcolor="#0E1117", 
            plot_bgcolor="#0E1117", 
            font={'color': "white"},
            xaxis_title="Date", 
            yaxis_title="Number of Patients",
            xaxis=dict(showgrid=True, gridcolor='#30363D'),
            yaxis=dict(showgrid=True, gridcolor='#30363D'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No data available for trend analysis.")

    st.markdown("---")

    # --- 4. Detailed Department Table ---
    st.markdown('<div class="section-header">Hospital Details Performance</div>', unsafe_allow_html=True)
    
    dept_rows = []
    for dept, info in DEPARTMENTS.items():
        d_df = calc[calc['Department'] == dept]
        
        d_active = len(d_df[d_df['Actual_Discharge'].isna()])
        d_bor = (d_active / info['cap']) * 100
        
        d_dis = d_df[d_df['Actual_Discharge'].notnull()]
        d_alos = (d_dis['Actual_Discharge'] - d_dis['Admit_Date']).dt.total_seconds().mean() / 86400 if not d_dis.empty else 0
        
        d_btr = len(d_dis) / info['cap']
        
        d_pat_days = ((d_df['Discharge_Calc'] - d_df['Admit_Date']).dt.total_seconds() / 86400).sum()
        d_avail_days = (info['cap'] * days_range) - d_pat_days
        d_bti = d_avail_days / len(d_dis) if len(d_dis) > 0 else 0
        
        dept_rows.append({
            "Department": dept,
            "BOR (%)": round(d_bor, 1),
            "ALOS (Days)": round(d_alos, 1),
            "BTR (Times)": round(d_btr, 2),
            "BTI (Days)": round(d_bti, 1)
        })
    
    # Progress Bar for BOR
    st.dataframe(
        pd.DataFrame(dept_rows),
        column_config={
            "BOR (%)": st.column_config.ProgressColumn(
                "BOR (%)",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------------
# 14. SETTINGS (Combined)
# ---------------------------------------------------------
elif menu == "Settings":
    st.title("⚙️ System Settings & Configuration")
    
    st.markdown('<div class="section-header">System Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Traditional System**")
        st.caption(f"Departments: {len(DEPARTMENTS)}")
        st.caption(f"Total Beds: {sum(d['cap'] for d in DEPARTMENTS.values())}")
        st.caption(f"Active Patients: {len(df[df['Actual_Discharge'].isna()])}")
    
    with col2:
        st.info("**ML System**")
        if model_loaded:
            st.success("✓ Machine Learning Model Loaded")
            st.caption("best_model.pkl")
        else:
            st.error("❌ Model Not Loaded")
        
        if data_loaded:
            st.success(f"✓ eICU Data Loaded ({len(df_eicu)} patients)")
        else:
            st.error("❌ Data Not Loaded")
    
    st.markdown("---")
    st.markdown('<div class="section-header">Data Management</div>', unsafe_allow_html=True)
    
    # Traditional System Reset
    st.warning("Factory Reset: This will wipe all traditional system data.")
    if st.button("FACTORY RESET (Traditional System)", type="primary"):
        del st.session_state.df
        st.success("Traditional System Reset Successfully.")
        time.sleep(1)
        st.rerun()
    
    st.markdown("---")
    st.markdown('<div class="section-header">Model Components Status</div>', unsafe_allow_html=True)
    
    components_status = {
        'best_model.pkl': model_loaded,
        'scaler.pkl': model_loaded,
        'label_encoders.pkl': model_loaded,
        'feature_columns.pkl': model_loaded,
        'patient.csv': data_loaded
    }
    
    for component, loaded in components_status.items():
        col_stat, col_icon = st.columns([3, 1])
        with col_stat:
            st.write(f"**{component}**")
        with col_icon:
            if loaded:
                st.success("✓")
            else:
                st.error("✗")
    
    st.markdown("---")
    st.markdown('<div class="section-header">System Configuration</div>', unsafe_allow_html=True)
    
    # Configuration options
    st.checkbox("Enable Real-time Predictions", value=True)
    st.checkbox("Show ML Recommendations", value=True)
    st.checkbox("Enable Automated Alerts", value=True)
    
    alert_threshold = st.slider("Occupancy Alert Threshold (%)", 50, 95, 85)
    forecast_hours = st.slider("Default Forecast Window (hours)", 1, 168, 24)
    
    if st.button("Save Configuration", type="primary"):
        st.success("Configuration saved successfully!")