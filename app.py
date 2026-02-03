import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import time
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. System Config & Design (OccupyBed AI MVP)
# ---------------------------------------------------------
st.set_page_config(page_title="OccupyBed AI MVP", layout="wide", page_icon="🏥")

# --- GLOBAL TIME FREEZE ---
# Setting "Now" to Jan 8, 2026 as requested
CURRENT_DATE = datetime(2026, 1, 8, 12, 0, 0)

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

    /* Inputs */
    div[data-baseweb="select"] > div, input { background-color: #0D1117 !important; border-color: #30363D !important; color: white !important; }
    button[kind="primary"] { background-color: #238636 !important; border: none !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Logic & Data
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
        
        # --- Generate Data Relative to CURRENT_DATE (Jan 8, 2026) ---
        data = []
        for dept, info in DEPARTMENTS.items():
            count = int(info['cap'] * np.random.uniform(0.4, 0.65))
            for i in range(count):
                bed_n = f"{dept[:3].upper()}-{i+1:03d}"
                
                # Admit date is between 1 and 10 days BEFORE Current Date
                days_ago = np.random.randint(1, 10)
                adm = CURRENT_DATE - timedelta(days=days_ago, hours=np.random.randint(1, 12))
                
                # Exp discharge is in future relative to admit
                exp = adm + timedelta(days=np.random.randint(3, 8))
                
                # Determine if active or discharged based on simulation
                # Some patients admitted 8 days ago might have been discharged 2 days ago
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
# 3. Sidebar (Search & Nav)
# ---------------------------------------------------------
with st.sidebar:
    # --- GLOWING LOGO SECTION ---
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    
    st.markdown("""
    <div class="logo-box">
        <div class="logo-main">OccupyBed AI</div>
        <div class="logo-slogan">intelligent hospital bed management</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Patient Search
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
    menu = st.radio("NAVIGATION", ["Overview", "Live Admissions", "Operational Analytics", "Settings"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("System Status: Online")

# ---------------------------------------------------------
# 4. OVERVIEW
# ---------------------------------------------------------
if menu == "Overview":
    c1, c2 = st.columns([3, 1])
    with c1: st.title("Hospital Command Center")
    with c2: 
        fc_hours = st.selectbox("Forecast Window", [6, 12, 24, 48, 72], index=2, format_func=lambda x: f"{x} Hours")

    # Metrics
    active_df = df[df['Actual_Discharge'].isna()]
    future_limit = CURRENT_DATE + timedelta(hours=fc_hours)
    
    total_cap = sum(d['cap'] for d in DEPARTMENTS.values())
    occ_count = len(active_df)
    avail_count = total_cap - occ_count
    ready_count = len(active_df[active_df['Exp_Discharge'] <= future_limit])

    # 1. Top Row: KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total Licensed Beds</div><div class="kpi-val" style="color:#58A6FF">{total_cap}</div></div>""", unsafe_allow_html=True)
    with k2: st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Occupied Beds</div><div class="kpi-val" style="color:#D29922">{occ_count}</div></div>""", unsafe_allow_html=True)
    with k3: st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Available Now</div><div class="kpi-val" style="color:#3FB950">{avail_count}</div></div>""", unsafe_allow_html=True)
    with k4: st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Expected Free ({fc_hours}h)</div><div class="kpi-val" style="color:#A371F7">{ready_count}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Middle Row: Gauge + AI
    g_col, ai_col = st.columns([1, 2])
    with g_col:
        # Gauge Chart
        occ_rate = (occ_count / total_cap) * 100 if total_cap > 0 else 0
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = occ_rate,
            title = {'text': "Hospital Pressure"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#58A6FF"},
                'steps': [
                    {'range': [0, 70], 'color': "#161B22"},
                    {'range': [70, 85], 'color': "#451a03"},
                    {'range': [85, 100], 'color': "#450a0a"}],
            }
        ))
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=0,b=0), paper_bgcolor="#0E1117", font={'color': "white"})
        st.plotly_chart(fig, use_container_width=True)

    with ai_col:
        st.markdown(f"""<div class="ai-container"><div class="ai-header">AI Operational Recommendations</div>""", unsafe_allow_html=True)
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
                
        if not ai_triggered:
            st.markdown("""<div class="ai-item" style="color:#3FB950">Hospital capacity is optimal. No bottlenecks detected.</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # 3. Bottom Row: Department Status
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
# 5. Live Admissions
# ---------------------------------------------------------
elif menu == "Live Admissions":
    st.title("Patient Admission & Discharge Center")
    
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

# ---------------------------------------------------------
# 6. Operational Analytics (V25 - Exact Chart Match)
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
    
    # --- 3. TREND CHART (Line Chart with Markers) - EXACT MATCH ---
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

        # Plotly Line Chart to mimic the reference image
        fig_trend = go.Figure()
        
        # Line 1: Admissions (Blue)
        fig_trend.add_trace(go.Scatter(
            x=trend['Admit_Date'], y=trend['Admissions'],
            mode='lines+markers',
            name='Admissions',
            line=dict(color='#1f77b4', width=2), # Standard Blue
            marker=dict(size=8)
        ))
        
        # Line 2: Discharges (Orange)
        fig_trend.add_trace(go.Scatter(
            x=trend['Admit_Date'], y=trend['Discharges'],
            mode='lines+markers',
            name='Discharges',
            line=dict(color='#ff7f0e', width=2), # Standard Orange
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
# 7. Settings
# ---------------------------------------------------------
elif menu == "Settings":
    st.title("System Settings")
    st.warning("Factory Reset: This will wipe all data. Use it to fix Ghost Data issues.")
    if st.button("FACTORY RESET (Clean System)", type="primary"):
        del st.session_state.df
        st.success("System Reset Successfully.")
        time.sleep(1)
        st.rerun()

