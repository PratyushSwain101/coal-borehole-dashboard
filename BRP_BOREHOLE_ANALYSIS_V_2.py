import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.colors as pcolors
from scipy.interpolate import Rbf  # Radial Basis Function for Extrapolation
from matplotlib.path import Path   # For Polygon Clipping
from scipy.spatial.distance import cdist
from shapely.geometry import Point, LineString, Polygon
from scipy.interpolate import interp1d
from functools import lru_cache
from scipy.interpolate import griddata 
from scipy.spatial import Delaunay # <--- for Triangulation
from scipy.interpolate import LinearNDInterpolator # <--- Add this

# --- NEW IMPORTS  FOR REPORT GENERATION---
# --- UPDATED IMPORTS FOR PATTERNS ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, A3 # Added A3 for wider side-by-side logs
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import Color
import io
import random

# --- CONFIGURATION ---

# List of LCODEs that represent Coal Seams (Dark Color) - MANDATORY SEQUENCE ORDER
COAL_SEAM_LCODES = [
    'PAR', 'LAJ4', 'L4B', 'LAJ3', 'L2T3', 'L2T2', 'L2T1', 'L2T1T', 'L2T1B',
    'L2B', 'LAJ1', 'LL1', 'R5', 'R5T', 'R5B', 'R4', 'R3T', 'R3B', 'R12',
    'IBT', 'IBB'
]

# Lithology Description Mapping (Based on your image)
LITHO_DESC_MAP = {
    'SOIL': 'ALLUVIUM AND SOIL',
    'CAS': 'CARBONACEOUS SHALE',
    'SST': 'SANDSTONE',
    'INT-SST & SHALE': 'INTERCALATION',
    'SHALE': 'SHALE',
    'COAL': 'COAL',
    'SHALY COAL': 'SHALY COAL',
    'CLAY': 'CLAY',
    'LOST CORE': 'LOST CORE',
    'CHERT RED': 'VERY FINE GRAINED SST',
    'SAND': 'SAND',
    'OB': 'OVERBURDEN'
}

# Define the Parent-Daughter Mapping for Linked Correlation
SEAM_SYSTEMS = {
    'L2T1': ['L2T1T', 'L2T1B'],
    'R5': ['R5T', 'R5B']
}
ALL_DAUGHTER_SEAMS = [d for sublist in SEAM_SYSTEMS.values() for d in sublist]
ALL_PARENT_SEAMS = list(SEAM_SYSTEMS.keys())
ALL_SYSTEM_SEAMS = ALL_PARENT_SEAMS + ALL_DAUGHTER_SEAMS

# Standard Quality Parameters
QUALITY_PARAMETERS = {
    'THICKNESS': 'Thickness (m)',
    'ASH_PERC': 'Ash Content (%)',
    'VM_PERC': 'Volatile Matter (%)',
    'FC_PERC': 'Fixed Carbon (%)',
    'GCV_KCAL': 'Gross Calorific Value (Kcal/kg)',
    'M_PERC': 'Moisture Content (%)',
    'S_PERC': 'Sulphur (%)',
    'C_PERC': 'Carbon (%)',
    'H_PERC': 'Hydrogen (%)',
    'N_PERC': 'Nitrogen (%)',
    'O_PERC(DIFF)': 'Oxygen (diff) (%)', 
    'PHOS_PERC': 'Phosphorus (%)',
    'CO2_PERC': 'CO2 (%)',
    'HGI': 'HGI'
}

# Base colors
PLOT_TEXT_COLOR = 'black' 
NON_COAL_COLOR = '#ADD8E6' 
NON_COAL_BORDER = 'black'

# --- FIXED COLOR MAP ---
SEAM_COLOR_MAP = {
    'PAR':   '#A020F0', # Purple/Violet
    'LAJ4':  '#1E90FF', # DodgerBlue
    'L4B':   '#808000', # Olive/Brown
    'LAJ3':  '#8A2BE2', # BlueViolet
    'L2T3':  '#696969', # DimGray
    'L2T2':  '#2E8B57', # SeaGreen
    'L2T1':  '#00FF00', # Lime (Bright Green)
    'L2T1T': '#F0E68C', # Khaki/Cream
    'L2T1B': '#D3D3D3', # LightGray
    'L2B':   '#32CD32', # LimeGreen
    'LAJ1':  '#A0522D', # Sienna/Rust
    'LL1':   '#E0B0FF', # Mauve
    'R5':    '#FF00FF', # Magenta
    'R5T':   '#4682B4', # SteelBlue
    'R5B':   '#FFA500', # Orange
    'R4':    '#FA8072', # Salmon/Pink
    'R3T':   '#9ACD32', # YellowGreen
    'R3B':   '#FF0000', # Red
    'R12':   '#00CED1', # DarkTurquoise
    'IBT':   '#00BFFF', # DeepSkyBlue
    'IBB':   '#8B008B'  # DarkMagenta
}

# --- Explicitly Define Default Color ---
DEFAULT_SEAM_COLOR = 'gray' 

def get_litho_color(lcode):
    """Returns the fixed color for a seam, or default for non-coal."""
    return SEAM_COLOR_MAP.get(lcode, NON_COAL_COLOR)

# --- STREAMLIT APP SETUP ---

st.set_page_config(layout="wide", page_title="BRP Coal Project_V3.2.1")
st.title("BRP Coal Project_V3.2.1")

# Initialize Session State
if 'df_bh' not in st.session_state: st.session_state['df_bh'] = None
if 'df_boundary' not in st.session_state: st.session_state['df_boundary'] = None
if 'df_litho' not in st.session_state: st.session_state['df_litho'] = None
if 'df_quality' not in st.session_state: st.session_state['df_quality'] = None
if 'df_master' not in st.session_state: st.session_state['df_master'] = None # <--- STORES OPTIMIZED MASTER DATA
if 'show_avg_all' not in st.session_state: st.session_state['show_avg_all'] = False 
if 'show_coal_only' not in st.session_state: st.session_state['show_coal_only'] = False
if 'corr_bhid_select' not in st.session_state: st.session_state['corr_bhid_select'] = []


# --- FILE PROCESSING FUNCTIONS ---

@st.cache_data(show_spinner=False)
def process_bh_data(uploaded_file):
    if uploaded_file:
        try:
            df_bh = pd.read_csv(uploaded_file)
            df_bh.columns = [col.upper().strip() for col in df_bh.columns]
            required_bh_cols = ['BHID', 'X', 'Y', 'RL', 'DEPTH']
            if not all(col in df_bh.columns for col in required_bh_cols):
                st.error(f"Borehole data must contain columns: {', '.join(required_bh_cols)}")
                return None
            df_bh['X'] = pd.to_numeric(df_bh['X'], errors='coerce')
            df_bh['Y'] = pd.to_numeric(df_bh['Y'], errors='coerce')
            df_bh['RL'] = pd.to_numeric(df_bh['RL'], errors='coerce')
            df_bh['DEPTH'] = pd.to_numeric(df_bh['DEPTH'], errors='coerce')
            df_bh.dropna(subset=['X', 'Y', 'RL', 'BHID', 'DEPTH'], inplace=True)
            return df_bh
        except Exception as e:
            st.error(f"Error processing Borehole file: {e}")
            return None
    return None

def process_boundary_data(uploaded_file):
    if uploaded_file:
        try:
            df_boundary = pd.read_csv(uploaded_file)
            df_boundary.columns = [col.upper().strip() for col in df_boundary.columns]
            df_boundary.dropna(subset=['X', 'Y'], inplace=True)
            return df_boundary
        except Exception as e:
            st.error(f"Error processing Boundary file: {e}")
            return None
    return None

def process_litho_data(uploaded_file):
    if uploaded_file:
        try:
            df_litho = pd.read_csv(uploaded_file)
            df_litho.columns = [col.upper().strip() for col in df_litho.columns]
            required_litho_cols = ['BHID', 'FROM', 'TO', 'LCODE', 'DETAILED LITHOLOGY']
            if not all(col in df_litho.columns for col in required_litho_cols):
                st.error(f"Lithology data must contain columns: {', '.join(required_litho_cols)}")
                return None
            df_litho['FROM'] = pd.to_numeric(df_litho['FROM'], errors='coerce')
            df_litho['TO'] = pd.to_numeric(df_litho['TO'], errors='coerce')
            df_litho['LCODE'] = df_litho['LCODE'].astype(str).str.upper().str.strip()
            df_litho['WIDTH'] = df_litho['TO'] - df_litho['FROM']
            df_litho['DETAILED LITHOLOGY'] = df_litho['DETAILED LITHOLOGY'].astype(str).str.strip()
            df_litho.dropna(subset=['BHID', 'FROM', 'TO', 'LCODE', 'WIDTH', 'DETAILED LITHOLOGY'], inplace=True)
            return df_litho
        except Exception as e:
            st.error(f"Error processing Lithology file: {e}")
            return None
    return None

def process_quality_data(uploaded_file):
    if uploaded_file:
        try:
            df_quality = pd.read_csv(uploaded_file)
            df_quality.columns = [col.upper().strip() for col in df_quality.columns]
            if 'LCODE TYPE OF SAMPLES' in df_quality.columns:
                df_quality.rename(columns={'LCODE TYPE OF SAMPLES': 'LCODE'}, inplace=True)
            required_quality_cols = ['BHID', 'FROM', 'TO', 'LCODE', 'SAMPLE_TYPE'] 
            if not all(col in df_quality.columns for col in required_quality_cols):
                st.error(f"Quality data must contain columns: BHID, FROM, TO, LCODE, and SAMPLE_TYPE.")
                return None
            df_quality['SAMPLE_TYPE'] = df_quality['SAMPLE_TYPE'].astype(str).str.upper().str.strip()
            df_quality['FROM'] = pd.to_numeric(df_quality['FROM'], errors='coerce')
            df_quality['TO'] = pd.to_numeric(df_quality['TO'], errors='coerce')
            df_quality['LCODE'] = df_quality['LCODE'].astype(str).str.upper().str.strip()
            df_quality['INTERVAL'] = df_quality['TO'] - df_quality['FROM']
            for col_key in QUALITY_PARAMETERS:
                if col_key in df_quality.columns:
                    df_quality[col_key] = pd.to_numeric(df_quality[col_key], errors='coerce')
            df_quality.dropna(subset=['BHID', 'FROM', 'TO', 'LCODE', 'SAMPLE_TYPE'], inplace=True)
            return df_quality
        except Exception as e:
            st.error(f"Error processing Quality file: {e}")
            return None
    return None









@st.cache_data(show_spinner=False)
def create_master_composite(df_bh, df_litho):
    """
    Creates a single, optimized master dataframe containing all spatial and lithological data.
    """
    if df_bh is None or df_litho is None:
        return None

    # Merge Litho with Collar info (Inner join keeps only valid boreholes)
    df_master = pd.merge(df_litho, df_bh[['BHID', 'X', 'Y', 'RL', 'DEPTH']], on='BHID', how='inner')
    
    # Calculate Absolute Elevations immediately
    df_master['Z_FROM'] = df_master['RL'] - df_master['FROM']
    df_master['Z_TO'] = df_master['RL'] - df_master['TO']
    df_master['THICKNESS'] = df_master['WIDTH']
    
    # Pre-calculate Colors
    df_master['COLOR'] = df_master['LCODE'].apply(get_litho_color)

    return df_master


# --- PDF EXPORT FUNCTIONS ---
def draw_pattern_rect(c, x, y, w, h, lcode):
    """
    Draws a rectangle filled with a custom geological pattern.
    """
    c.saveState()
    
    # Clip path
    path = c.beginPath()
    path.rect(x, y, w, h)
    c.clipPath(path, stroke=0)
    
    # Default Base Color
    base_color = colors.white
    lcode_u = lcode.upper()
    
    if lcode in COAL_SEAM_LCODES: base_color = colors.black
    elif 'SHALE' in lcode_u or 'CAS' in lcode_u: base_color = colors.Color(0.95, 0.95, 0.95)
    elif 'SST' in lcode_u or 'SAND' in lcode_u: base_color = colors.Color(1, 1, 0.9) # Light Yellow
    
    c.setFillColor(base_color)
    c.rect(x, y, w, h, fill=1, stroke=1) # Draw background

    # --- PATTERN LOGIC ---
    if lcode in COAL_SEAM_LCODES:
        c.setFillColor(colors.black)
        c.rect(x, y, w, h, fill=1, stroke=0)
        
    elif 'SST' in lcode_u or 'SAND' in lcode_u:
        c.setFillColor(colors.black)
        # Dots
        rows = int(h / (2*mm)) 
        cols = int(w / (2*mm))
        for r in range(rows + 2):
            for col in range(cols + 2):
                dot_x = x + (col * 2*mm) + random.uniform(-0.5, 0.5)
                dot_y = y + (r * 2*mm) + random.uniform(-0.5, 0.5)
                c.circle(dot_x, dot_y, 0.2, fill=1, stroke=0)

    elif 'SHALE' in lcode_u:
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.5)
        c.setDash([3, 2]) 
        step = 2*mm
        curr_y = y
        while curr_y < y + h:
            c.line(x, curr_y, x + w, curr_y)
            curr_y += step
        c.setDash([])

    elif 'INT' in lcode_u or 'INTER' in lcode_u:
        # Alternating lines
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.5)
        step = 1.5*mm
        curr_y = y
        while curr_y < y + h:
            c.line(x, curr_y, x + w, curr_y)
            curr_y += step

    elif 'SOIL' in lcode_u or 'ALLUVIUM' in lcode_u:
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.5)
        
        # Tighter spacing constants (2.5mm instead of 5mm)
        step_y = 2.5 * mm
        step_x = 2.5 * mm
        
        rows = int(h / step_y)
        cols = int(w / step_x)
        
        for r in range(rows + 2):
            for col in range(cols + 2):
                # Grid position with random jitter for organic look
                sx = x + (col * step_x) + random.uniform(-1, 1)
                sy = y + (r * step_y) + random.uniform(-1, 1)
                
                # Draw 'v' shape
                p = c.beginPath()
                p.moveTo(sx, sy)
                p.lineTo(sx + 1.2, sy - 2) 
                p.lineTo(sx + 2.4, sy)
                c.drawPath(p, stroke=1, fill=0)


    elif 'CLAY' in lcode_u:
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.5)
        
        # Tighter grid spacing 
        step_y = 2* mm  # Vertical gap (was 3mm)
        step_x = 3.5 * mm  # Horizontal gap
        
        rows = int(h / step_y)
        cols = int(w / step_x)
        
        for r in range(rows + 2):
            for col in range(cols + 2):
                # Grid position with random jitter
                sx = x + (col * step_x) + random.uniform(-1, 1)
                sy = y + (r * step_y) + random.uniform(-0.5, 0.5)
                
                # Draw short horizontal dash (length 2mm)
                c.line(sx, sy, sx + 2*mm, sy)
            
    elif 'CAS' in lcode_u: # Carbonaceous Shale (Cross hatch or dense lines)
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.5)
        step = 1.5*mm
        curr_y = y
        while curr_y < y + h:
            c.line(x, curr_y, x + w, curr_y)
            curr_y += step
        # Vertical hatch for CAS
        curr_x = x
        while curr_x < x + w:
            c.line(curr_x, y, curr_x, y+h)
            curr_x += 3*mm
    c.restoreState()



def generate_graphic_log_pdf(df_master, selected_bhids):
    buffer = io.BytesIO()
    
    # PAGE CONFIG
    SCALE_FACTOR = 1 * cm  # 1m = 1cm
    MARGIN_TOP = 4 * cm
    MARGIN_BOTTOM = 5 * cm 
    MARGIN_LEFT = 2 * cm
    COL_WIDTH = 2.5 * cm 
    SPACING = 6 * cm       
    
    # 1. Determine Page Dimensions & Unique Lithologies for Legend
    max_depth = 0
    unique_lcodes_found = set()
    
    for bhid in selected_bhids:
        subset = df_master[df_master['BHID'] == bhid]
        if not subset.empty:
            # Use actual max depth if available, else max TO
            d_row = df_master[(df_master['BHID'] == bhid) & (df_master['DEPTH'].notna())]
            d = d_row.iloc[0]['DEPTH'] if not d_row.empty else subset['TO'].max()
            
            if d > max_depth: max_depth = d
            unique_lcodes_found.update(subset['LCODE'].unique())
            
    total_content_height = (max_depth * SCALE_FACTOR) + MARGIN_TOP + MARGIN_BOTTOM
    # Calculate width based on number of boreholes
    logs_width = MARGIN_LEFT + (len(selected_bhids) * (COL_WIDTH + SPACING))
    # Add space for Legend on the right (at least 8cm)
    total_content_width = logs_width + 10*cm 
    
    c = canvas.Canvas(buffer, pagesize=(total_content_width, total_content_height))
    Y_REF = total_content_height - MARGIN_TOP

    # --- DRAW LEGEND (Aligned to Right Margin) ---
    # Position legend 1cm from the right edge
    legend_x = total_content_width - 9*cm 
    legend_y = total_content_height - 2*cm
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(legend_x, legend_y + 0.5*cm, "LITHOLOGY LEGEND")
    
    # --- LEGEND LOGIC FIX: Aggressive Coal Consolidation ---
    legend_items = []
    has_coal = False
    
    for lcode in unique_lcodes_found:
        # Check 1: Is it in the official seam list?
        # Check 2: Does the name itself say "COAL"? (Fixes duplicate issue)
        if lcode in COAL_SEAM_LCODES or 'COAL' in lcode.upper():
            has_coal = True
        else:
            legend_items.append(lcode)
    
    # Sort non-coal items alphabetically
    legend_items.sort()
    
    # Add single Consolidated Entry
    if has_coal:
        legend_items.insert(0, 'COAL')
        
    # Draw Legend Box Grid
    l_box_h = 0.6*cm
    curr_ly = legend_y
    
    for item in legend_items:
        # Draw Pattern Box
        # If item is 'COAL', we pass a known coal code to the drawer to get the black box
        draw_code = COAL_SEAM_LCODES[0] if item == 'COAL' else item
        
        draw_pattern_rect(c, legend_x, curr_ly - l_box_h, 1*cm, l_box_h, draw_code)
        
        # Draw Description
        if item == 'COAL':
            desc = "COAL SEAM"
        else:
            desc = LITHO_DESC_MAP.get(item.upper(), item)
            
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 8)
        c.drawString(legend_x + 1.2*cm, curr_ly - 0.4*cm, desc)
        
        curr_ly -= (l_box_h + 0.2*cm)

    # --- DRAW LOGS ---
    for i, bhid in enumerate(selected_bhids):
        x_origin = MARGIN_LEFT + (i * (COL_WIDTH + SPACING))
        
        bh_data = df_master[df_master['BHID'] == bhid].sort_values('FROM')
        if bh_data.empty: continue
        
        collar_rl = bh_data.iloc[0]['RL']
        bh_td_row = df_master[(df_master['BHID'] == bhid) & (df_master['DEPTH'].notna())]
        total_depth = bh_td_row.iloc[0]['DEPTH'] if not bh_td_row.empty else bh_data['TO'].max()
        
        # 1. Header
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(x_origin + COL_WIDTH/2, Y_REF + 2.5*cm, bhid)
        c.setFont("Helvetica", 9)
        c.drawCentredString(x_origin + COL_WIDTH/2, Y_REF + 2.0*cm, f"RL: {collar_rl:.2f}m")
        c.drawCentredString(x_origin + COL_WIDTH/2, Y_REF + 1.6*cm, f"TD: {total_depth:.2f}m")
        
        # 2. Depth Scale
        c.setLineWidth(1)
        c.line(x_origin, Y_REF, x_origin, Y_REF - (total_depth * SCALE_FACTOR))
        
        for d in range(0, int(total_depth) + 1):
            y_pos = Y_REF - (d * SCALE_FACTOR)
            if d % 5 == 0:
                c.line(x_origin - 3*mm, y_pos, x_origin, y_pos)
                c.setFont("Helvetica", 8)
                c.drawRightString(x_origin - 4*mm, y_pos - 2, f"{d}")
            else:
                c.line(x_origin - 1.5*mm, y_pos, x_origin, y_pos)

        # 3. Lithology Column & Labels
        for _, row in bh_data.iterrows():
            top = row['FROM']
            bot = row['TO']
            lcode = row['LCODE']
            thick = bot - top
            
            if thick <= 0: continue
            
            rect_y = Y_REF - (bot * SCALE_FACTOR)
            rect_h = thick * SCALE_FACTOR
            
            # Pattern
            draw_pattern_rect(c, x_origin, rect_y, COL_WIDTH, rect_h, lcode)
            
            # Center Point for Text
            center_x = x_origin + (COL_WIDTH / 2)
            center_y = rect_y + (rect_h / 2) - 2 
            
            # A. COAL SEAMS (Bracket + Bold Depth)
            if lcode in COAL_SEAM_LCODES:
                # Bracket
                bracket_x = x_origin + COL_WIDTH + 2*mm
                bracket_top = Y_REF - (top * SCALE_FACTOR)
                bracket_bot = Y_REF - (bot * SCALE_FACTOR)
                
                c.setLineWidth(1)
                c.line(bracket_x, bracket_top, bracket_x + 2*mm, bracket_top) 
                c.line(bracket_x + 2*mm, bracket_top, bracket_x + 2*mm, bracket_bot)
                c.line(bracket_x, bracket_bot, bracket_x + 2*mm, bracket_bot)
                
                # Seam Name
                c.setFont("Helvetica-Bold", 10)
                c.drawString(bracket_x + 4*mm, (bracket_top + bracket_bot)/2 - 3, lcode)
                
                # Bold Depth Labels
                c.setFont("Helvetica-Bold", 8)
                c.drawRightString(x_origin - 8*mm, bracket_top - 2, f"{top:.2f}")
                c.drawRightString(x_origin - 8*mm, bracket_bot + 2, f"{bot:.2f}")

            # B. NON-COAL LAYERS (Text in Middle Center)
            else:
                if rect_h > 0.5*cm:
                    c.setFillColor(colors.black)
                    c.setFont("Helvetica", 7)
                    c.drawCentredString(center_x, center_y, lcode)

        # 4. "Closed At" Footer
        close_y = Y_REF - (total_depth * SCALE_FACTOR) - 1*cm
        c.setLineWidth(1.5)
        c.line(x_origin, close_y + 0.8*cm, x_origin + COL_WIDTH, close_y + 0.8*cm)
        
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(x_origin + COL_WIDTH/2, close_y, f"Closed At: {total_depth:.2f} m")

    c.save()
    buffer.seek(0)
    return buffer
# --- CORE PLOTTING FUNCTIONS (2D) ---

def plot_seam_stats(df_stats, title, y_axis_title, parameter, plot_type, selected_seams_d):
    
    if plot_type == 'Bar Chart':
        df_plot = df_stats[df_stats['LCODE'].isin(COAL_SEAM_LCODES)].copy()
        if df_plot.empty: return go.Figure().add_annotation(text="No data found for the selected criteria.", showarrow=False).update_layout(title_text=title, height=400), pd.DataFrame()
        plot_col = 'AVERAGE_THICKNESS_M' if 'AVERAGE_THICKNESS_M' in df_plot.columns else parameter
        df_plot = df_plot.rename(columns={'LCODE': 'COAL_SEAM', plot_col: 'VALUE'})
        present_seams = df_plot['COAL_SEAM'].unique().tolist()
        seam_plot_order = [seam for seam in COAL_SEAM_LCODES if seam in present_seams]
        fig = px.bar(df_plot, x='COAL_SEAM', y='VALUE', title=title, labels={'VALUE': y_axis_title, 'COAL_SEAM': 'Coal Seam LCODE'}, color='COAL_SEAM', color_discrete_map=SEAM_COLOR_MAP, category_orders={"COAL_SEAM": seam_plot_order}, text='VALUE')
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        y_max = df_plot['VALUE'].max() * 1.1 if not df_plot.empty and df_plot['VALUE'].max() > 0 else 10
        fig.update_layout(xaxis={'categoryorder':'array'}, yaxis=dict(range=[0, y_max]), font=dict(color=PLOT_TEXT_COLOR), height=500, legend=dict(font=dict(size=10)))
        df_summary = df_plot.rename(columns={'VALUE': y_axis_title})
        df_summary['COAL_SEAM'] = pd.Categorical(df_summary['COAL_SEAM'], categories=COAL_SEAM_LCODES, ordered=True)
        df_summary = df_summary.sort_values('COAL_SEAM')
        df_summary['SEAM NAME'] = df_summary['COAL_SEAM']
        df_summary = df_summary[['SEAM NAME', y_axis_title]]
        df_summary[y_axis_title] = df_summary[y_axis_title].round(2)
        return fig, df_summary
    elif plot_type == 'Box Plot':
        # NOTE: This uses RAW data (df_stats should contain LCODE, parameter, BHID)
        df_plot_raw = df_stats[df_stats['LCODE'].isin(selected_seams_d)].copy()
        if df_plot_raw.empty or parameter not in df_plot_raw.columns: return go.Figure().add_annotation(text=f"No raw data available for {parameter}.", showarrow=False).update_layout(title_text=title, height=500), pd.DataFrame()
        df_plot_raw = df_plot_raw.dropna(subset=[parameter])
        if df_plot_raw.empty: return go.Figure().add_annotation(text=f"No valid {parameter} samples found.", showarrow=False).update_layout(title_text=title, height=500), pd.DataFrame()
        present_seams = df_plot_raw['LCODE'].unique().tolist()
        seam_plot_order = [seam for seam in COAL_SEAM_LCODES if seam in present_seams]
        df_plot_raw['COAL_SEAM'] = pd.Categorical(df_plot_raw['LCODE'], categories=seam_plot_order, ordered=True)
        df_plot_raw = df_plot_raw.sort_values('COAL_SEAM')
        df_summary_data = df_plot_raw.groupby('LCODE')[parameter].agg(['count', 'mean', 'median', 'min', 'max']).reset_index()
        fig = px.box(df_plot_raw, x='COAL_SEAM', y=parameter, title=title, labels={parameter: y_axis_title, 'COAL_SEAM': 'Coal Seam LCODE'}, color='COAL_SEAM', category_orders={"COAL_SEAM": seam_plot_order}, color_discrete_map=SEAM_COLOR_MAP)
        fig.update_traces(marker_size=5, line=dict(width=1))
        for i, seam in enumerate(seam_plot_order):
            stats = df_summary_data[df_summary_data['LCODE'] == seam]
            if not stats.empty:
                stats = stats.iloc[0]
                fig.add_annotation(x=i, y=stats['median'], text=f"{stats['median']:.2f}", showarrow=False, textangle=-90, font=dict(size=10, color=PLOT_TEXT_COLOR), yshift=15, xanchor='center', yanchor='middle', bgcolor="rgba(255,255,255,0.8)", bordercolor='black', borderwidth=0.5)
        y_data_min, y_data_max = df_summary_data['min'].min(), df_summary_data['max'].max()
        y_range = y_data_max - y_data_min if y_data_max > y_data_min else 1
        y_min_adj, y_max_adj = y_data_min - y_range * 0.1, y_data_max + y_range * 0.1  
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color=PLOT_TEXT_COLOR), height=600, yaxis=dict(range=[y_min_adj, y_max_adj]), xaxis=dict(tickangle=-45, categoryorder='array', categoryarray=seam_plot_order), legend=dict(font=dict(size=10)))
        df_full_stats = df_plot_raw.groupby('LCODE')[parameter].agg(['count', 'mean', 'median', 'std', 'min', 'max']).reset_index()
        df_full_stats.columns = ['SEAM NAME', 'Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
        for col in ['Mean', 'Median', 'Std Dev', 'Min', 'Max']: df_full_stats[col] = df_full_stats[col].round(2)
        df_full_stats['SEAM NAME'] = pd.Categorical(df_full_stats['SEAM NAME'], categories=seam_plot_order, ordered=True)
        df_full_stats = df_full_stats.sort_values('SEAM NAME').dropna(subset=['SEAM NAME'])
        return fig, df_full_stats
    return go.Figure().add_annotation(text="Invalid Plot Type Selected.", showarrow=False), pd.DataFrame()


def plot_plan_view(df_bh, df_boundary, selected_bhids=None):
    selected_bhids = selected_bhids if isinstance(selected_bhids, list) else ([selected_bhids] if selected_bhids else [])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_boundary['X'], y=df_boundary['Y'], mode='lines', line=dict(color='red', width=1, dash='dash'),
        name='Block Boundary', hovertemplate='Boundary Point<extra></extra>', showlegend=True
    ))

    # <<< Hover template >>>
    hover_template = (
        '<b>BHID:</b> %{customdata[0]}<br>' +
        '<b>RL:</b> %{customdata[1]:.2f}<br>' +
        '<b>TD:</b> %{customdata[2]:.2f}<br>' +
        '<b>X:</b> %{x:.2f}<br>' +
        '<b>Y:</b> %{y:.2f}<extra></extra>'
    )
    
    df_unselected = df_bh[~df_bh['BHID'].isin(selected_bhids)]
    
    # <<< Unselected boreholes trace >>>
    fig.add_trace(go.Scatter(
        x=df_unselected['X'], y=df_unselected['Y'], mode='markers', 
        marker=dict(size=8, color=NON_COAL_COLOR, line=dict(width=1, color=NON_COAL_BORDER)),
        name='Boreholes',
        hovertemplate=hover_template,
        customdata=df_unselected[['BHID', 'RL', 'DEPTH']],
        showlegend=True, 
        legendgroup='boreholes'
    ))

    bh_label_y_offset = -(df_bh['Y'].max() - df_bh['Y'].min()) * 0.03
    fig.add_trace(go.Scatter(
        x=df_bh['X'], y=df_bh['Y'] + bh_label_y_offset, mode='text', text=df_bh['BHID'], textposition="bottom center",
        textfont=dict(size=8), showlegend=False, legendgroup='boreholes', hoverinfo='skip'
    ))
    if selected_bhids:
        df_selected = df_bh[df_bh['BHID'].isin(selected_bhids)]
        if not df_selected.empty:
            # <<< Selected boreholes trace >>>
            fig.add_trace(go.Scatter(
                x=df_selected['X'], y=df_selected['Y'], mode='markers', 
                marker=dict(size=13, color='red', symbol='circle', line=dict(width=2, color=PLOT_TEXT_COLOR)),
                name=f'Selected BH ({len(selected_bhids)})', 
                hovertemplate=hover_template,
                customdata=df_selected[['BHID', 'RL', 'DEPTH']],
                showlegend=True, 
                legendgroup='selected_bhids'
            ))
        df_labels = df_bh[df_bh['BHID'].isin(selected_bhids)].copy()
        fig.add_trace(go.Scatter(
            x=df_labels['X'], y=df_labels['Y'] + bh_label_y_offset, mode='text', text=df_labels['BHID'], textposition="bottom center",
            textfont=dict(size=8, color='red'), showlegend=False, hoverinfo='skip', legendgroup='selected_bhids'
        ))
    if selected_bhids and len(selected_bhids) > 1:
        df_polyline = df_bh[df_bh['BHID'].isin(selected_bhids)].set_index('BHID').loc[selected_bhids]
        fig.add_trace(go.Scatter(
            x=df_polyline['X'], y=df_polyline['Y'], mode='text+lines', text=[str(i+1) for i in range(len(selected_bhids))],
            textposition="middle center", line=dict(color='blue', width=2, dash='dot'), textfont=dict(size=10, color="White"),
            name='Correlation Line', hoverinfo='text', hovertext=[f'Order: {i+1} / {bhid}' for i, bhid in enumerate(selected_bhids)], showlegend=True
        ))
    
    fig.update_layout(
        xaxis_title="Easting (X) - UTM", yaxis_title="Northing (Y) - UTM", dragmode='pan',
        yaxis=dict(scaleanchor="x", scaleratio=1), title_text="Borehole Locations & Block Boundary (Plan View)",
        title_font=dict(color=PLOT_TEXT_COLOR), 
        font=dict(color=PLOT_TEXT_COLOR), 
        # plot_bgcolor='white', # Optimized for Light Theme
        # paper_bgcolor='white', # Optimized for Light Theme
        hovermode="closest", height=700,
        legend=dict(title = "Legend",font=dict(size=10)),
        margin=dict(l=0, r=0, t=50, b=50)
    )
    return fig

def plot_litho_correlation(df_bh, df_litho, selected_bhids, selected_seams, filter_mode, reference_seam=None):
    scale_multiplier, BAR_WIDTH_VISUAL, HEADER_HEIGHT_OFFSET = 1.0, 15, 15
    excluded_bhids = []
    
    df_combined = pd.merge(df_litho, df_bh[['BHID', 'RL', 'DEPTH', 'X', 'Y']], on='BHID', how='left').dropna(subset=['RL', 'DEPTH', 'X', 'Y'])
    df_combined['FROM RL'] = df_combined['RL'] - df_combined['FROM']
    df_combined['TO RL'] = df_combined['RL'] - df_combined['TO']
    df_combined['RL_WIDTH'] = df_combined['FROM RL'] - df_combined['TO RL']

    is_flattened_mode = reference_seam and reference_seam != 'None'
    if is_flattened_mode:
        bh_offsets = {}
        
        # --- DATUM LOGIC: Use only the exact reference seam for the floor datum ---
        ref_target_lcodes = [reference_seam]
        
        # NOTE: If a parent is selected as datum, use ALL components for composite datum floor
        if reference_seam in ALL_PARENT_SEAMS:
            ref_target_lcodes.extend(SEAM_SYSTEMS[reference_seam])
                    
        for bhid in selected_bhids:
            # Filter data for the reference seam system (only the exact seam, or composite if parent)
            ref_seam_data = df_combined[(df_combined['BHID'] == bhid) & (df_combined['LCODE'].isin(ref_target_lcodes))]
            
            # The reference point is the *absolute floor* (min TO RL) of the system
            min_to_rl = ref_seam_data['TO RL'].min()
            
            if not ref_seam_data.empty and not pd.isna(min_to_rl): 
                bh_offsets[bhid] = -min_to_rl
            else: 
                excluded_bhids.append(bhid)
                
            plottable_bhids = [bhid for bhid in selected_bhids if bhid not in excluded_bhids]
    else:
        bh_offsets = {bhid: 0 for bhid in selected_bhids}
        plottable_bhids = selected_bhids

    if not plottable_bhids:
        return go.Figure().add_annotation(text="No boreholes selected.", showarrow=False), excluded_bhids

    df_selected_bh = df_bh[df_bh['BHID'].isin(plottable_bhids)].set_index('BHID').loc[plottable_bhids].reset_index()
    bh_x_positions = [0.0]
    for i in range(1, len(df_selected_bh)):
        x1, y1 = df_selected_bh.loc[i-1, ['X', 'Y']]; x2, y2 = df_selected_bh.loc[i, ['X', 'Y']]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        bh_x_positions.append(bh_x_positions[-1] + distance * scale_multiplier)
    df_selected_bh['CUM_DISTANCE'] = bh_x_positions

    # --- FILTERING DATA BASED ON 'filter_mode' ---
    if filter_mode == 'Coal Seams Only':
        df_plot_data = df_combined[df_combined['LCODE'].isin(COAL_SEAM_LCODES)].copy()
    else:
        df_plot_data = df_combined.copy()

    all_y_values, max_header_y, collar_y_coords = [], 0, []
    for _, row in df_selected_bh.iterrows():
        bhid, y_offset = row['BHID'], bh_offsets.get(row['BHID'], 0)
        collar_y_coords.append((row['CUM_DISTANCE'], row['RL'] + y_offset))
        max_header_y = max(max_header_y, row['RL'] + y_offset)
        bh_data = df_plot_data[df_plot_data['BHID'] == bhid]
        if not bh_data.empty:
            all_y_values.extend((bh_data['FROM RL'] + y_offset).tolist()); all_y_values.extend((bh_data['TO RL'] + y_offset).tolist())
    
    min_y_range = min(all_y_values) - 10 if all_y_values else -50
    max_y_range = max(all_y_values) + HEADER_HEIGHT_OFFSET + 20 if all_y_values else 50
    max_y_range = max(max_header_y + HEADER_HEIGHT_OFFSET + 10, max_y_range)
        
    fig = go.Figure()
    


    # 1. Plot Lithology Bars and Annotations
    for i, row in df_selected_bh.iterrows():
        bhid, rl, final_depth, x_pos, y_offset = row['BHID'], row['RL'], row['DEPTH'], row['CUM_DISTANCE'], bh_offsets.get(row['BHID'], 0)
        df_litho_bh = df_plot_data[df_plot_data['BHID'] == bhid]
        
        # Total Depth Line (Hidden Bar)
        fig.add_trace(go.Bar(x=[x_pos], y=[final_depth], base=[rl - final_depth + y_offset], marker=dict(color='rgba(0,0,0,0)', line=dict(color='black', width=1.0)), orientation='v', width=BAR_WIDTH_VISUAL, hoverinfo='skip', showlegend=False))
        
        # Lithology Intervals (Stacked Bars)
        if not df_litho_bh.empty:
            df_litho_bh['COLOR'] = df_litho_bh['LCODE'].apply(get_litho_color)

            hover_text_series =('<b>BHID:</b> ' + bhid + '<br>' + 
            '<b>RL:</b> ' + df_litho_bh['FROM RL'].round(2).astype(str) + ' to ' + df_litho_bh['TO RL'].round(2).astype(str) + ' m<br>'+ 
            '<b>Depth:</b> ' + df_litho_bh['FROM'].round(2).astype(str) + ' to ' + df_litho_bh['TO'].round(2).astype(str) + ' m<br>' + '<b>Thickness:</b> ' + df_litho_bh['WIDTH'].round(2).astype(str) + ' m<br>' + 
            '<b>LCODE:</b> ' + df_litho_bh['LCODE'] + '<br>' + 
            '<b>Lithology:</b> ' + df_litho_bh['DETAILED LITHOLOGY'])
            
            # RECTIFIED TRACE: Removed the invalid `legendgroup` assignment
            fig.add_trace(go.Bar(
                x=[x_pos] * len(df_litho_bh), 
                y=df_litho_bh['RL_WIDTH'], 
                base=df_litho_bh['TO RL'] + y_offset, 
                marker=dict(color=df_litho_bh['COLOR'], line=dict(color='black', width=1.0)), 
                text=df_litho_bh['LCODE'], 
                textposition='inside', 
                textfont=dict(color=PLOT_TEXT_COLOR, size=9), 
                orientation='v', 
                width=BAR_WIDTH_VISUAL, 
                hoverinfo='text', 
                hovertext=hover_text_series, 
                showlegend=False, # We use the dummy traces for the legend
            ))
        
        # Borehole Header Annotation
        fig.add_annotation(x=x_pos, y=max_header_y + HEADER_HEIGHT_OFFSET, text=f"<b>{bhid}</b><br>RL: {rl:.1f}<br>TD: {final_depth:.1f} m", showarrow=False, font=dict(color=PLOT_TEXT_COLOR, size=10), xanchor='center', yanchor='bottom')
        
        # Depth Ticks (Right Side of Log)
        seam_boundaries_rl = df_combined[df_combined['LCODE'].isin(COAL_SEAM_LCODES) & (df_combined['BHID'] == bhid)][['FROM RL', 'TO RL']].stack().unique().tolist()
        seam_boundaries_rl.append(rl - final_depth)
        unique_rls_to_label = sorted(list(set([r for r in seam_boundaries_rl if r <= rl + 0.1])), reverse=True)
        for rl_tick in unique_rls_to_label:
            fig.add_annotation(x=x_pos + BAR_WIDTH_VISUAL / 2 + 5, y=rl_tick + y_offset, text=f"{(rl - rl_tick):.2f} m", showarrow=False, font=dict(color=PLOT_TEXT_COLOR, size=8), xanchor='left', yanchor='middle')

    # 2. Implement LINKED Seam Correlation Line Logic (Visual Grouping & Composite Envelope)
    
    seams_to_plot_lines = set()
    systems_to_plot_composite = set()
    
    # 2a. Determine the final set of LCODEs to plot based on user selection
    for s_seam in selected_seams:
        # 1. Plot the individual selected seam
        seams_to_plot_lines.add(s_seam)
        
        # 2. If part of a split system is selected, mark the parent for COMPOSITE tracing
        if s_seam in ALL_PARENT_SEAMS:
            systems_to_plot_composite.add(s_seam)
            for daughter in SEAM_SYSTEMS[s_seam]:
                seams_to_plot_lines.add(daughter)
        elif s_seam in ALL_DAUGHTER_SEAMS:
            for parent, daughters in SEAM_SYSTEMS.items():
                if s_seam in daughters:
                    systems_to_plot_composite.add(parent)
                    # Add all other components for tracing
                    seams_to_plot_lines.add(parent)
                    for daughter in daughters:
                          seams_to_plot_lines.add(daughter)
                    break 
                
    
    # Use a dictionary to store traces to ensure consistent color mapping
    plot_traces = {}
    
    # 2b. Plot COMPOSITE ENVELOPES first (Solid line)
    for parent_seam in sorted(list(systems_to_plot_composite)): 
        line_color = SEAM_COLOR_MAP.get(parent_seam, 'black') # Use the parent seam's color
        x_coords, y_top_composite, y_bottom_composite = [], [], []
        
        # Target all components of the system
        target_lcodes = [parent_seam] + SEAM_SYSTEMS[parent_seam]
        
        for _, bh_row in df_selected_bh.iterrows():
            bhid, dist, offset = bh_row['BHID'], bh_row['CUM_DISTANCE'], bh_offsets.get(bh_row['BHID'], 0)
            
            seam_data = df_combined[(df_combined['BHID'] == bhid) & (df_combined['LCODE'].isin(target_lcodes))]
            
            if not seam_data.empty: 
                # COMPOSITE ROOF/FLOOR: Max FROM RL and Min TO RL of all components
                y_top_composite.append(seam_data['FROM RL'].max() + offset); 
                y_bottom_composite.append(seam_data['TO RL'].min() + offset)
                x_coords.append(dist)
        
        if len(x_coords) > 1:
            plot_traces[f'{parent_seam}_COMPOSITE_TOP'] = go.Scatter(
                x=x_coords, y=y_top_composite, mode='lines+markers', 
                line=dict(color=line_color, width=3, dash='solid'), 
                name=f'{parent_seam} Composite Roof', showlegend=True,
                legendgroup=parent_seam
            )
            plot_traces[f'{parent_seam}_COMPOSITE_BOTTOM'] = go.Scatter(
                x=x_coords, y=y_bottom_composite, mode='lines+markers', 
                line=dict(color=line_color, width=3, dash='dashdot'), # Slightly different dash for floor
                name=f'{parent_seam} Composite Floor', showlegend=True,
                legendgroup=parent_seam
            )
            
    # 2c. Plot INDIVIDUAL SEAM LINES (Daughter or Non-split Seams)
    for s_seam in sorted(list(seams_to_plot_lines)): 
        if s_seam in systems_to_plot_composite:
            # Skip plotting the individual parent LCODE if its composite is already plotted (avoids duplication)
            continue
            
        if s_seam == 'None' or s_seam not in COAL_SEAM_LCODES: continue
        
        line_color = SEAM_COLOR_MAP.get(s_seam, 'black')
        x_coords, y_top, y_bottom = [], [], []
        
        # Target LCODE for visualization is now ALWAYS just the seam itself
        target_lcodes = [s_seam]
        
        dash_style = 'dot' if s_seam in ALL_DAUGHTER_SEAMS else 'solid'
        legend_name = f'{s_seam} Top/Bottom'
        
        for _, bh_row in df_selected_bh.iterrows():
            bhid, dist, offset = bh_row['BHID'], bh_row['CUM_DISTANCE'], bh_offsets.get(bh_row['BHID'], 0)
            
            seam_data = df_combined[(df_combined['BHID'] == bhid) & (df_combined['LCODE'].isin(target_lcodes))]
            
            if not seam_data.empty: 
                # Individual Roof/Floor: Max FROM RL and Min TO RL of the single seam
                y_top.append(seam_data['FROM RL'].max() + offset); 
                y_bottom.append(seam_data['TO RL'].min() + offset)
                x_coords.append(dist)
                
        # Plot the lines
        if len(x_coords) > 1:
            plot_traces[f'{s_seam}_TOP'] = go.Scatter(x=x_coords, y=y_top, mode='lines+markers', line=dict(color=line_color, width=1, dash=dash_style), name=legend_name, showlegend=True, legendgroup=s_seam)
            plot_traces[f'{s_seam}_BOTTOM'] = go.Scatter(x=x_coords, y=y_bottom, mode='lines+markers', line=dict(color=line_color, width=1, dash=dash_style), name=legend_name, showlegend=False, legendgroup=s_seam)


    # Add all calculated traces to the figure
    for trace in plot_traces.values():
        fig.add_trace(trace)
            
    # 3. Plot Surface Profile (if not flattened)
    if not is_flattened_mode and len(collar_y_coords) > 1:
        x_c, y_c = zip(*collar_y_coords); fig.add_trace(go.Scatter(x=list(x_c), y=list(y_c), mode='lines', line=dict(color='blue', width=2, dash='dash'), name='Surface Profile', showlegend=True))

    # 4. Final Layout Configuration

    # --- START FIX: DUMMY TRACES FOR LITHOLOGY LEGEND ---
    # Determine the unique LCODEs present in the current plot data (Coal or All Lithology)
    if not df_plot_data.empty:
        # Get all unique LCODEs that actually appear in the plotted bars
        unique_plotted_lcodes = df_plot_data['LCODE'].unique().tolist()
    else:
        unique_plotted_lcodes = []
    
    # Sort Coal Seams by COAL_SEAM_LCODES order
    unique_coal_lcodes = [lcode for lcode in COAL_SEAM_LCODES if lcode in unique_plotted_lcodes]
    
    # Non-Coal LCODEs that are currently plotted (only relevant if filter_mode != 'Coal Seams Only')
    unique_non_coal_lcodes = [lcode for lcode in unique_plotted_lcodes if lcode not in COAL_SEAM_LCODES]

    # Add dummy trace for COAL SEAMS (Mandatory for Legend)
    for lcode in unique_coal_lcodes:
        # Use go.Bar with opacity 0 to ensure the color box is correct and toggleable
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            marker=dict(color=get_litho_color(lcode), line=dict(color='black', width=1)),
            name=lcode,
            showlegend=True,
            legendgroup='Lithology_Coal', # Use consistent group name
            visible='legendonly' # Hide the trace itself, but keep the legend item
        ))

    # Add dummy trace for NON-COAL SEAMS (Only if filter is 'All Lithology')
    if filter_mode != 'Coal Seams Only':
        for lcode in unique_non_coal_lcodes:
             fig.add_trace(go.Bar(
                 x=[None], y=[None],
                 marker=dict(color=NON_COAL_COLOR, line=dict(color='black', width=1)),
                 name=lcode,
                 showlegend=True,
                 legendgroup='Lithology_NonCoal', # Use consistent group name
                 visible='legendonly'
             ))

    # --- END FIX: DUMMY TRACES FOR LITHOLOGY LEGEND ---

    # 5. Final Layout Configuration (Cont.)
    if is_flattened_mode:
        title, yaxis_title = f"Seam Correlation (Datum: Floor of '{reference_seam}')", "Relative Elevation from Datum (m)"
        yaxis_config = dict(title=yaxis_title, showticklabels=True, showgrid=True, zeroline=True, zerolinecolor='red', zerolinewidth=2, range=[min_y_range, max_y_range])
    else:
        title, yaxis_title = "Geological Correlation Plot", "Elevation (RL) above MSL (m)"
        yaxis_config = dict(title=yaxis_title, showgrid=True, zeroline=True, zerolinecolor=PLOT_TEXT_COLOR, range=[min_y_range, max_y_range])
    
    # We remove the old custom lithology legend logic at the end since we use the dummy traces now
    
    fig.update_layout(
        title_text=title, title_font=dict(size=16),
        xaxis=dict(title="Cumulative Distance along Section (m)", tickvals=df_selected_bh['CUM_DISTANCE'], ticktext=[f'{d:.0f} m' for d in df_selected_bh['CUM_DISTANCE']], showgrid=False, zeroline=False),
        yaxis=yaxis_config, height=700, barmode='stack', 
        # plot_bgcolor='white', 
        # paper_bgcolor='white', 
        font=dict(color=PLOT_TEXT_COLOR),
        legend=dict(title="Legend",font=dict(size=10), x=1.02, y=1,  borderwidth=1),
        margin=dict(l=50, r=100, t=100, b=50)
    )
    return fig, excluded_bhids

def plot_quality_crossplot(df_quality, selected_seam, selected_sample_type, x_param, y_param):
    if df_quality is None or x_param not in df_quality.columns or y_param not in df_quality.columns: 
        return go.Figure().add_annotation(text="Missing data for cross-plot.", showarrow=False)

    # NEW LOGIC: Include daughters if parent is selected
    target_lcodes = [selected_seam]
    if selected_seam in SEAM_SYSTEMS:
        target_lcodes.extend(SEAM_SYSTEMS[selected_seam])

    df_plot = df_quality[df_quality['LCODE'].isin(target_lcodes)].copy()
    
    if selected_sample_type != 'All Samples': 
        df_plot = df_plot[df_plot['SAMPLE_TYPE'] == selected_sample_type].copy()
    
    df_plot = df_plot.dropna(subset=[x_param, y_param])
    if df_plot.empty: return go.Figure().add_annotation(text=f"No combined data found for {x_param} vs {y_param} in Seam {selected_seam} for Sample Type {selected_sample_type}.", showarrow=False)
    x_display, y_display = QUALITY_PARAMETERS.get(x_param, x_param), QUALITY_PARAMETERS.get(y_param, y_param)
    fig = px.scatter(df_plot, x=x_param, y=y_param, color='INTERVAL', size='INTERVAL', color_continuous_scale=px.colors.sequential.Viridis, title=f"Cross-Plot: {x_display} vs. {y_display} for Seam {selected_seam} ({selected_sample_type})", labels={x_param: x_display, y_param: y_display, 'INTERVAL': 'Sample Interval (m)'}, hover_data=['BHID', 'FROM', 'TO'])
    X, Y = df_plot[x_param].values, df_plot[y_param].values
    if len(X) >= 2 and np.std(X) > 0:
        try:
            coeffs = np.polyfit(X, Y, 1); slope, intercept = coeffs[0], coeffs[1]
            r_sq = 1 - (np.sum((Y - (intercept + slope * X))**2) / np.sum((Y - np.mean(Y))**2))
            x_fit, y_fit = np.array([X.min(), X.max()]), intercept + slope * np.array([X.min(), X.max()])
            fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', line=dict(color='red', width=2, dash='dash'), name=f'Regression Line (R²={r_sq:.3f})', hovertemplate=f'Predicted {y_param}: %{{y:.2f}}<extra>R²={r_sq:.3f}</extra>'))
            sign = "+" if intercept >= 0 else "-"; equation = f"{y_param} = {slope:.2f} * {x_param} {sign} {abs(intercept):.2f}"
            fig.add_annotation(x=df_plot[x_param].max(), y=df_plot[y_param].min(), text=f"<b>Eq:</b> {equation}<br><b>R²:</b> {r_sq:.3f}", showarrow=False, xref="x", yref="y", xanchor='right', yanchor='bottom', bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="black", borderwidth=1, font=dict(size=10))
        except Exception as e: st.caption(f"Note: Could not calculate regression line. Error: {e}")
    
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color=PLOT_TEXT_COLOR), height=600, hovermode='closest', coloraxis_colorbar=dict(title="Sample Interval (m)"), legend=dict(font=dict(size=10)))
    return fig


def preprocess_quality_data(df_litho, df_quality, df_bh, selected_seam, selected_sample_type):
    """
    Prepares data for the Quality Map.
    FIX: Strictly filters for the single selected seam (LCODE) only.
    Does NOT aggregate daughter seams into the parent.
    """
    
    # --- FIX START: Strict Selection ---
    # We only look for the exact selected seam. 
    # We do NOT extend the list with daughter seams anymore.
    target_lcodes = [selected_seam]
    # --- FIX END ---
        
    # Calculate Thickness for ONLY the selected seam
    df_thickness = df_litho[df_litho['LCODE'].isin(target_lcodes)].groupby('BHID')['WIDTH'].sum().reset_index()
    df_thickness.columns = ['BHID', 'THICKNESS']
    
    # Merge with borehole coordinates
    df_stats = pd.merge(df_bh[['BHID', 'X', 'Y', 'RL']].copy(), df_thickness, on='BHID', how='left')
    df_stats['THICKNESS'] = df_stats['THICKNESS'].fillna(0)
    
    if df_quality is not None:
        def calculate_wavg_for_seam(df, parameter):
            # Mask out NaNs to ensure accurate weighted average
            clean_df = df.dropna(subset=[parameter, 'INTERVAL'])
            if clean_df['INTERVAL'].sum() == 0: return np.nan
            return (clean_df[parameter] * clean_df['INTERVAL']).sum() / clean_df['INTERVAL'].sum()
        
        quality_cols = [col for col in df_quality.columns if col in QUALITY_PARAMETERS and col != 'THICKNESS']
        
        # Strict filter for quality data as well
        df_quality_seam = df_quality[df_quality['LCODE'].isin(target_lcodes)].copy()
        
        if selected_sample_type != 'All Samples':
            df_quality_seam = df_quality_seam[df_quality_seam['SAMPLE_TYPE'] == selected_sample_type].copy()
            
        # Group and calculate weighted averages
        if not df_quality_seam.empty:
            wavg_results = df_quality_seam.groupby('BHID').apply(
                lambda x: pd.Series({col: calculate_wavg_for_seam(x, col) for col in quality_cols}),
                include_groups=False 
            ).reset_index()

            df_stats = pd.merge(df_stats, wavg_results, on='BHID', how='left')
        
    return df_stats



def calculate_quality_stats_data(df_quality, selected_param, selected_sample_type, bh_ids_to_analyze):
    """
    Calculates the **Weighted Average** of a quality parameter for each coal seam.
    Formula: Sum(Value * Interval) / Sum(Interval)
    """
    if df_quality is None or selected_param not in df_quality.columns:
        return pd.DataFrame()

    # 1. Filter quality data by selected boreholes and coal seams only
    df_filtered = df_quality[
        (df_quality['LCODE'].isin(COAL_SEAM_LCODES)) & 
        (df_quality['BHID'].isin(bh_ids_to_analyze))
    ].copy()

    # 2. Filter by sample type
    if selected_sample_type != 'All Samples':
        df_filtered = df_filtered[df_filtered['SAMPLE_TYPE'] == selected_sample_type].copy()
    
    # 3. Drop NaNs in the selected parameter AND Interval
    #    (Crucial for Weighted Average)
    df_filtered.dropna(subset=[selected_param, 'INTERVAL'], inplace=True)

    if df_filtered.empty:
        return pd.DataFrame()

    # 4. Calculate Weighted Values
    df_filtered['WEIGHTED_VAL'] = df_filtered[selected_param] * df_filtered['INTERVAL']

    # 5. Group by Seam (LCODE) and sum the weighted values and intervals
    df_grouped = df_filtered.groupby('LCODE')[['WEIGHTED_VAL', 'INTERVAL']].sum().reset_index()

    # 6. Calculate final Weighted Average
    #    Avoid division by zero
    df_grouped = df_grouped[df_grouped['INTERVAL'] > 0].copy()
    df_grouped[selected_param] = df_grouped['WEIGHTED_VAL'] / df_grouped['INTERVAL']

    # 7. Return only the LCODE and the Calculated Parameter
    return df_grouped[['LCODE', selected_param]]



# --- FINAL UPDATED FUNCTION:---
def plot_quality_plan_view(df_bh, df_boundary, df_quality, df_litho):
    
    # 1. Selection Controls
    col_seam, col_sample, col_param, col_secondary_param, col_colorscale = st.columns([1, 1, 1, 1, 1])
    
    param_list = list(QUALITY_PARAMETERS.keys())
    sample_type_list = ['All Samples']
    if df_quality is not None:
        sample_type_list.extend(df_quality['SAMPLE_TYPE'].unique().tolist())
    
    with col_seam:
        seam_list = COAL_SEAM_LCODES
        selected_seam = st.selectbox("1. Select Coal Seam:", seam_list, key='map_seam_select')
    
    available_params = ['THICKNESS']
    if df_quality is not None:
        available_params.extend([col for col in df_quality.columns if col in QUALITY_PARAMETERS and col != 'THICKNESS'])
    
    with col_sample:
        selected_sample_type = st.selectbox("2. Select Sample Type:", sample_type_list, key='map_sample_select')
    
    with col_param:
        if not available_params:
            st.warning("No quality data columns found.")
            return
        selected_param_key = st.selectbox("3. Color Parameter (Primary):", available_params, key='map_param_select')
        param_display_name = QUALITY_PARAMETERS.get(selected_param_key, selected_param_key)
    
    with col_secondary_param:
        available_secondary = ['None'] + available_params
        selected_secondary_key = st.selectbox("4. Label Parameter (Secondary):", available_secondary, index=0, key='map_secondary_param_select')
        secondary_display_name = QUALITY_PARAMETERS.get(selected_secondary_key, selected_secondary_key)

    with col_colorscale:
        sequential_colorscales = (['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo', 'Jet', 'Hot', 'Electric', 'Portland', 'Blackbody'])
        selected_colorscale = st.selectbox("5. Select Color Scale:", sequential_colorscales[:], index=9, key='map_colorscale_select')

    # --- CONTINUOUS MAP CONTROLS ---
    st.write("")
    c_toggle, c_power, c_radius = st.columns([1, 1, 1])
    with c_toggle:
        st.write("")
        show_continuous = st.toggle("Show Interpolated Map", value=False, key='toggle_continuous_map')
    with c_power:
        idw_power = st.slider("IDW Power (Smoothness):", 1.0, 5.0, 2.0, 0.5, help="Controls how fast influence drops with distance; 2.0 is standard.")
    with c_radius:
        radius_limit = st.number_input("Interpolation Radius (m):", min_value=100.0, max_value=5000.0, value=500.0, step=50.0)

    # Preprocess Data
    df_analyzed = preprocess_quality_data(df_litho, df_quality, df_bh, selected_seam, selected_sample_type)
    
    seam_depths_df = df_litho[df_litho['LCODE'] == selected_seam].groupby('BHID').agg(
        SEAM_FROM=('FROM', 'min'), SEAM_TO=('TO', 'max')
    ).reset_index()
    df_analyzed = pd.merge(df_analyzed, seam_depths_df, on='BHID', how='left')
    df_analyzed = pd.merge(df_analyzed, df_bh[['BHID', 'DEPTH']], on='BHID', how='left')

    # Filter Data (Non-zero)
    df_plot_data = df_analyzed[
        (df_analyzed[selected_param_key].notna()) & 
        (df_analyzed[selected_param_key] > 0.001)
    ].copy()
    
    fig = go.Figure()

    if not df_plot_data.empty:
        param_min_data = df_plot_data[selected_param_key].min()
        param_max_data = df_plot_data[selected_param_key].max()
        
        # --- IDW INTERPOLATION LOGIC ---
        if show_continuous:
            try:
                points_x = df_plot_data['X'].values
                points_y = df_plot_data['Y'].values
                values = df_plot_data[selected_param_key].values

                if len(points_x) >= 3:
                    # 1. Grid Generation
                    resolution = 200 
                    min_x, max_x = df_boundary['X'].min(), df_boundary['X'].max()
                    min_y, max_y = df_boundary['Y'].min(), df_boundary['Y'].max()
                    
                    grid_x_1d = np.linspace(min_x, max_x, resolution)
                    grid_y_1d = np.linspace(min_y, max_y, resolution)
                    grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)
                    
                    grid_points = np.column_stack((grid_x.flatten(), grid_y.flatten()))
                    data_points = np.column_stack((points_x, points_y))
                    
                    # 2. CALCULATE IDW
                    dists = cdist(grid_points, data_points, metric='euclidean')
                    dists[dists == 0] = 1e-9
                    weights = 1.0 / np.power(dists, idw_power)
                    weights_sum = weights.sum(axis=1)
                    weights_sum[weights_sum == 0] = 1.0
                    
                    grid_z_flat = np.sum(weights * values, axis=1) / weights_sum
                    grid_z = grid_z_flat.reshape(grid_x.shape)
                    
                    # 3. Apply Distance Masking
                    min_dists = dists.min(axis=1).reshape(grid_x.shape)
                    distance_mask = min_dists > radius_limit
                    grid_z[distance_mask] = np.nan

                    # 4. Boundary Clipping
                    poly_path = Path(list(zip(df_boundary['X'], df_boundary['Y'])))
                    mask_poly = poly_path.contains_points(grid_points).reshape(grid_x.shape)
                    grid_z[~mask_poly] = np.nan 

                    # 5. Add Contour Trace (MASTER LEGEND)
                    fig.add_trace(go.Contour(
                        z=grid_z, x=grid_x_1d, y=grid_y_1d,
                        colorscale=selected_colorscale,
                        # Match the min/max of the data exactly so it matches the boreholes
                        zmin=param_min_data, zmax=param_max_data,
                        contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white')),
                        # Use the primary parameter name for the legend title
                        colorbar=dict(title=dict(text=f'{param_display_name}', side='right'), x=1.15),
                        opacity=1, connectgaps=False, line_smoothing=0.5,
                        hoverinfo='z', name='IDW Model',
                        showscale=True # Always show scale if map is visible
                    ))
                else:
                    st.warning("Not enough data points (minimum 3) for interpolation.")
            except Exception as e:
                st.error(f"Error generating continuous map: {e}")

        # --- SCATTER PLOT (BOREHOLES) ---
        param_min_rounded = round(float(param_min_data), 2)
        param_max_rounded = round(float(param_max_data), 2)
        st.caption(f"Enter the Min/Max value for {param_display_name} (Data Range: {param_min_rounded:.2f} to {param_max_rounded:.2f})")
        col_min, col_max, col_highlight_mode = st.columns([1, 1, 2])
        with col_min:
            min_val = st.number_input("Min Value:", min_value=param_min_rounded, max_value=param_max_rounded, value=param_min_rounded, step=0.1, format="%.2f", key='quality_range_min')
        with col_max:
            max_val = st.number_input("Max Value:", min_value=float(min_val), max_value=param_max_rounded, value=param_max_rounded, step=0.1, format="%.2f", key='quality_range_max')
        with col_highlight_mode:
            st.write(""); st.write(""); highlight_mode = st.radio("Highlight Boreholes:", ('None', 'In Range', 'Outside Range'), index=0, key='highlight_mode', horizontal=True)

        EPSILON = 0.00001
        actual_min, actual_max = min(min_val, max_val), max(min_val, max_val)
        if highlight_mode == 'In Range':
            df_plot_data['Filtered'] = (df_plot_data[selected_param_key] >= actual_min - EPSILON) & (df_plot_data[selected_param_key] <= actual_max + EPSILON)
        elif highlight_mode == 'Outside Range':
            df_plot_data['Filtered'] = (df_plot_data[selected_param_key] < actual_min - EPSILON) | (df_plot_data[selected_param_key] > actual_max + EPSILON)
        else:
            df_plot_data['Filtered'] = False

        st.markdown("---")
        
        param_short_name = param_display_name.split('(')[0].strip()
        unit_str = ""
        if selected_param_key == 'THICKNESS': unit_str = " m"
        elif '(' in param_display_name and ')' in param_display_name:
            unit_str = " " + param_display_name[param_display_name.find('(') : param_display_name.find(')')+1]


        hover_template = (
            '<b>BHID:</b> %{customdata[0]}<br>' +
            '<b>RL:</b> %{customdata[1]:.2f} (m) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>TD:</b> %{customdata[2]:.2f} (m)<br>' +
            f'<b>{param_short_name}:</b><b> %{{marker.color:.2f}}{unit_str}</b> <b>({selected_seam})</b><br>' +
            f'<b>Depth:</b> %{{customdata[3]:.2f}} (m)' +
            f'<b> to</b> %{{customdata[4]:.2f}} (m)<br>' +
            '<b>X:</b> %{x:.2f}<br>' +
            '<b>Y:</b> %{y:.2f}<extra></extra>'
        )

        # 1. Borehole Scatter Trace
        # LOGIC CHANGE: Only show colorbar here if the continuous map (show_continuous) is FALSE.
        fig.add_trace(go.Scatter(
            x=df_plot_data['X'], y=df_plot_data['Y'], mode='markers',
            marker=dict(
                size=10, 
                color=df_plot_data[selected_param_key], 
                colorscale=selected_colorscale, 
                colorbar=dict(title=f'{param_display_name}', title_side='right'), 
                showscale=(not show_continuous), 
                cmin=param_min_data, cmax=param_max_data, 
                line=dict(width=1, color=NON_COAL_BORDER)
            ),
            name=f'Boreholes ({len(df_plot_data)})', hovertemplate=hover_template, customdata=df_plot_data[['BHID', 'RL', 'DEPTH', 'SEAM_FROM', 'SEAM_TO']], showlegend=True, legendgroup='data_points'
        ))

        # 2. BHID Labels
        fig.add_trace(go.Scatter(x=df_plot_data['X'], y=df_plot_data['Y'] - 60, mode='text', text=df_plot_data['BHID'], textposition="bottom center", textfont=dict(size=8, color=PLOT_TEXT_COLOR), showlegend=False, hoverinfo='skip', legendgroup='data_points'))
        
        # 3. Secondary Labels
        if selected_secondary_key != 'None':
            def format_secondary_label(row):
                val = row[selected_secondary_key]
                if pd.isna(val): return ''
                short = secondary_display_name.split('(')[0].strip()
                unit = secondary_display_name[secondary_display_name.find('(') : secondary_display_name.find(')')+1] if '(' in secondary_display_name else ''
                return f"{short}: {val:.2f} {unit}"
            df_plot_data['SECONDARY_LABEL'] = df_plot_data.apply(format_secondary_label, axis=1)
            fig.add_trace(go.Scatter(x=df_plot_data['X'] + 40, y=df_plot_data['Y'] + 60, mode='text', text=df_plot_data['SECONDARY_LABEL'], textfont=dict(size=10, color='darkgreen'), name='Secondary Labels', showlegend=False, hoverinfo='skip', legendgroup='data_points'))

        # 4. Highlight Filter
        if highlight_mode != 'None':
            df_highlight = df_plot_data[df_plot_data['Filtered']].copy()
            if not df_highlight.empty:
                fig.add_trace(go.Scatter(x=[None], y=[None], marker=dict(size=10, color='red', symbol='circle', line=dict(width=3, color='red')), name=f'Highlighted ({len(df_highlight)})', showlegend=True, legendgroup='highlight'))
                fig.add_trace(go.Scatter(x=df_highlight['X'], y=df_highlight['Y'], mode='markers', marker=dict(size=10, color=df_highlight[selected_param_key], colorscale=selected_colorscale, showscale=False, cmin=param_min_data, cmax=param_max_data, line=dict(width=3, color='red')), name='Highlighted Points', hovertemplate=hover_template, customdata=df_highlight[['BHID', 'RL', 'DEPTH', 'SEAM_FROM', 'SEAM_TO']], showlegend=False, legendgroup='highlight'))
                fig.add_trace(go.Scatter(x=df_highlight['X'], y=df_highlight['Y'] - 60, mode='text', text=df_highlight['BHID'], textposition="bottom center", textfont=dict(size=8, color='red'), showlegend=False, hoverinfo='skip', legendgroup='highlight'))
            else:
                st.info(f"No boreholes found **{highlight_mode.lower()}** range.")

        # 5. Boundary Trace
        fig.add_trace(go.Scatter(x=df_boundary['X'], y=df_boundary['Y'], mode='lines', line=dict(color='red', width=2, dash='dash'), name='Block Boundary', hoverinfo='skip', showlegend=True))
        
        fig.update_layout(
            xaxis_title="Easting (X)", yaxis_title="Northing (Y)", dragmode='pan', yaxis=dict(scaleanchor="x", scaleratio=1),
            title_text=f"Quality Map (IDW): {param_display_name} - {selected_seam}",
            hovermode="closest", height=700,
            font=dict(color=PLOT_TEXT_COLOR), margin=dict(l=50, r=200, t=80, b=50),
            legend=dict(font=dict(size=10), x=1.25, y=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistical Summary
        if not df_plot_data.empty:
            data_to_summarize = df_plot_data[selected_param_key].dropna()
            summary_data = {'Metric': ['Boreholes (n)', 'Min', 'Max', 'Mean', 'Median'], 'Value': [len(data_to_summarize), f"{data_to_summarize.min():.2f}", f"{data_to_summarize.max():.2f}", f"{data_to_summarize.mean():.2f}", f"{data_to_summarize.median():.2f}"]}
            st.subheader(f"Statistical Summary : ({selected_param_key})")
            st.dataframe(pd.DataFrame(summary_data).set_index('Metric').style.set_properties(**{'text-align': 'left'}), use_container_width=True)
            st.markdown("---")

        if highlight_mode != 'None' and not df_plot_data.empty:
            df_high = df_plot_data[df_plot_data['Filtered']].copy()
            if not df_high.empty:
                df_sum_h = df_high[['BHID', selected_param_key]].copy()
                df_sum_h.columns = ['BHID', f'Value {unit_str}']
                df_sum_h.insert(1, 'Seam', selected_seam)
                st.subheader(f"Highlighted Boreholes: **{highlight_mode}** ({len(df_high)} BHs)")
                st.dataframe(df_sum_h.style.format({f'Value {unit_str}': "{:.2f}"}), use_container_width=True)










@st.cache_data(show_spinner=False)
def calculate_fault_anomalies(df_master, seam_code, dip_threshold_deg):
    """
    Analyzes the structural floor of a seam to identify steep gradients (Potential Faults).
    """
    # 1. Filter Data for the specific seam (Floor elevation)
    seam_data = df_master[df_master['LCODE'] == seam_code].copy()
    
    # We need at least 3 points to form a surface
    if len(seam_data) < 3:
        return pd.DataFrame(), seam_data

    # Extract coordinates
    coords = seam_data[['X', 'Y']].values
    z_vals = seam_data['Z_TO'].values # Floor RL
    bhids = seam_data['BHID'].values
    
    # 2. Delaunay Triangulation to find neighbors
    tri = Delaunay(coords)
    
    # 3. Extract unique edges from triangles
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))
        edges.add(tuple(sorted((simplex[2], simplex[0]))))
    
    anomaly_list = []
    
    # 4. Calculate Dip and Throw for each edge
    for p1_idx, p2_idx in edges:
        # Horizontal Distance
        dx = coords[p1_idx][0] - coords[p2_idx][0]
        dy = coords[p1_idx][1] - coords[p2_idx][1]
        dist_horiz = np.sqrt(dx**2 + dy**2)
        
        if dist_horiz < 0.1: continue # Avoid division by zero
        
        # Vertical Throw
        z1 = z_vals[p1_idx]
        z2 = z_vals[p2_idx]
        throw = abs(z1 - z2)
        
        # Dip Angle (Degrees)
        dip_rad = np.arctan(throw / dist_horiz)
        dip_deg = np.degrees(dip_rad)
        
        # Filter based on threshold
        if dip_deg >= dip_threshold_deg:
            anomaly_list.append({
                'BHID_A': bhids[p1_idx],
                'BHID_B': bhids[p2_idx],
                'X1': coords[p1_idx][0], 'Y1': coords[p1_idx][1],
                'X2': coords[p2_idx][0], 'Y2': coords[p2_idx][1],
                'Dist_H': dist_horiz,
                'Throw_m': throw,
                'Dip_Deg': dip_deg
            })
            
    df_anomalies = pd.DataFrame(anomaly_list)
    return df_anomalies, seam_data







def plot_fault_map(df_anomalies, df_seam_points, df_boundary, seam_name, threshold):
    fig = go.Figure()

    # 1. Plot Block Boundary
    fig.add_trace(go.Scatter(
        x=df_boundary['X'], y=df_boundary['Y'], mode='lines', 
        line=dict(color='black', width=1, dash='dash'), 
        name='Boundary', hoverinfo='skip'
    ))

    # 2. Plot Boreholes (Colored by Floor RL) + LABELS ADDED
    if not df_seam_points.empty:
        fig.add_trace(go.Scatter(
            x=df_seam_points['X'], y=df_seam_points['Y'], 
            mode='markers+text', # <--- Changed to show text
            marker=dict(
                size=8, 
                color=df_seam_points['Z_TO'], 
                colorscale='Viridis', 
                # Adjusted Colorbar to avoid overlap
                colorbar=dict(
                    title='Floor RL (m)',
                    x=1, 
                    len=0.7, 
                    y=0.4
                ),
                line=dict(width=1, color='black')
            ),
            text=df_seam_points['BHID'], # <--- Labels
            textposition="top center",   # <--- Position
            textfont=dict(size=9, color='black'),
            customdata=df_seam_points['Z_TO'],
            hovertemplate='<b>%{text}</b><br>Floor RL: %{customdata:.2f}m<extra></extra>',
            name=f'{seam_name} Data Points'
        ))

    # 3. Plot Anomalies (Fault Lines)
    if not df_anomalies.empty:
        # Create line segments for the high-dip connections
        x_lines, y_lines = [], []
        
        for _, row in df_anomalies.iterrows():
            x_lines.extend([row['X1'], row['X2'], None])
            y_lines.extend([row['Y1'], row['Y2'], None])

        fig.add_trace(go.Scatter(
            x=x_lines, y=y_lines,
            mode='lines',
            line=dict(color='red', width=3),
            name=f'High Dip (> {threshold}°)',
            hoverinfo='skip'
        ))
        
        # Add labels at the midpoint of anomalies
        mid_x = (df_anomalies['X1'] + df_anomalies['X2']) / 2
        mid_y = (df_anomalies['Y1'] + df_anomalies['Y2']) / 2
        
        fig.add_trace(go.Scatter(
            x=mid_x, y=mid_y,
            mode='markers',
            marker=dict(size=8, color='blue', symbol='x'),
            text=df_anomalies.apply(lambda r: f"Dip: {r['Dip_Deg']:.0f}°", axis=1),
            hovertemplate='<b>Potential Fault</b><br>%{text}<br>Throw: %{customdata:.1f}m<extra></extra>',
            customdata=df_anomalies['Throw_m'],
            name='Anomaly Info'
        ))

    # 4. FIX LAYOUT OVERLAP
    fig.update_layout(
        title=f"Structural Anomaly Map: {seam_name} (Threshold: {threshold}°)",
        xaxis_title="Easting", yaxis_title="Northing",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=700, 
        plot_bgcolor='white',
        # Move Legend to top-left inside plot to stop fighting with colorbar
        legend=dict(
            title="Legend",
            x=1, 
            y=0.99, 
            # bgcolor="rgba(255,255,255,0.8)", 
            # # bordercolor="black", 
            # borderwidth=1
        ),
        margin=dict(r=150) # Add margin on right for colorbar
    )
    return fig










# --- 3D MODELLING FUNCTIONS (OPTIMIZED) ---

@st.cache_data(show_spinner=False)
def prepare_3d_data(df_bh, df_litho, selected_bhids, selected_seams_filter, include_waste):
    # This legacy function is kept if needed for cross-section, 
    # but the 3D model now uses df_master directly.
    df_bh_sel = df_bh[df_bh['BHID'].isin(selected_bhids)].copy()
    df_litho_sel = df_litho[df_litho['BHID'].isin(selected_bhids)].copy()
    
    if selected_seams_filter:
        cond_coal = df_litho_sel['LCODE'].isin(selected_seams_filter)
        if include_waste:
            cond_waste = ~df_litho_sel['LCODE'].isin(COAL_SEAM_LCODES)
            df_litho_sel = df_litho_sel[cond_coal | cond_waste]
        else:
            df_litho_sel = df_litho_sel[cond_coal]
    
    df_3d = pd.merge(df_litho_sel, df_bh_sel[['BHID', 'X', 'Y', 'RL']], on='BHID', how='inner')
    df_3d['Z_FROM'] = df_3d['RL'] - df_3d['FROM']
    df_3d['Z_TO'] = df_3d['RL'] - df_3d['TO']
    df_3d['COLOR'] = df_3d['LCODE'].apply(get_litho_color)
    return df_3d

def create_cylinder_mesh(df_segment, radius, z_exaggeration, color, name, lcode, draw_lines=True):
    # Kept for "High Res" mode
    x_coords, y_coords, z_coords = [], [], []
    i_indices, j_indices, k_indices = [], [], []
    hover_texts = []
    line_x, line_y, line_z = [], [], []
    
    angles = np.linspace(0, 2*np.pi, 7)[:-1] 
    cos_a = np.cos(angles) * radius
    sin_a = np.sin(angles) * radius
    cos_a_loop = np.append(cos_a, cos_a[0])
    sin_a_loop = np.append(sin_a, sin_a[0])
    
    current_vertex_offset = 0
    
    for _, row in df_segment.iterrows():
        cx, cy = row['X'], row['Y']
        z_top = row['Z_FROM'] * z_exaggeration
        z_bot = row['Z_TO'] * z_exaggeration
        
        # Vertices
        x_coords.extend(cx + cos_a); y_coords.extend(cy + sin_a); z_coords.extend([z_top] * 6)
        x_coords.extend(cx + cos_a); y_coords.extend(cy + sin_a); z_coords.extend([z_bot] * 6)
        
        # --- FIXED HOVER TEXT ---
        litho_desc = str(row.get('DETAILED LITHOLOGY', '')).strip()
        hover_info = (
            f"<b>BHID:</b> {row['BHID']}<br>"
            f"<b>LCODE:</b> {lcode}<br>"
            f"<b>Thickness:</b> {row.get('WIDTH',0):.2f}m<br>"
            f"<b>Detailed Litho:</b> {litho_desc}"
        )
        hover_texts.extend([hover_info] * 12)
        
        # Faces
        for s in range(6):
            t1, t2 = current_vertex_offset + s, current_vertex_offset + (s + 1) % 6
            b1, b2 = current_vertex_offset + 6 + s, current_vertex_offset + 6 + (s + 1) % 6
            i_indices.extend([t1, t2]); j_indices.extend([b1, b1]); k_indices.extend([t2, b2])
            
        current_vertex_offset += 12 
        
        if draw_lines:
            line_x.extend(cx + cos_a_loop); line_y.extend(cy + sin_a_loop); line_z.extend([z_bot] * 7)
            line_x.append(None); line_y.append(None); line_z.append(None)
        
    mesh_trace = go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, i=i_indices, j=j_indices, k=k_indices, color=color, name=name, hoverinfo='text', text=hover_texts, flatshading=True, showlegend=True, legendgroup=lcode)
    return mesh_trace, (line_x, line_y, line_z)



@st.cache_data(show_spinner=False)
def generate_seam_surface_optimized(df_master, seam_code, df_boundary, z_exaggeration, resolution, show_roof, show_floor, opacity, max_distance, method='linear'):
    # 1. Selection
    if seam_code == "TOPO":
        topo_data = df_master.drop_duplicates(subset=['BHID'])
        if len(topo_data) < 3: return []
        points_x, points_y = topo_data['X'].values, topo_data['Y'].values
        points_z_roof = topo_data['RL'].values
        points_z_floor = None 
        seam_color = '#DEB887'
    else:
        seam_data = df_master[df_master['LCODE'] == seam_code]
        if len(seam_data) < 3: return []
        points_x, points_y = seam_data['X'].values, seam_data['Y'].values
        points_z_roof = seam_data['Z_FROM'].values
        points_z_floor = seam_data['Z_TO'].values
        seam_color = get_litho_color(seam_code)

    # 2. Grid
    min_x, max_x = df_boundary['X'].min(), df_boundary['X'].max()
    min_y, max_y = df_boundary['Y'].min(), df_boundary['Y'].max()
    grid_x, grid_y = np.mgrid[min_x:max_x:complex(0, resolution), min_y:max_y:complex(0, resolution)]
    
    # 3. Masks
    grid_flat = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    poly_path = Path(list(zip(df_boundary['X'], df_boundary['Y'])))
    mask_poly = poly_path.contains_points(grid_flat)
    
    data_points = np.column_stack((points_x, points_y))
    dists = cdist(grid_flat, data_points, metric='euclidean')
    mask_dist = dists.min(axis=1) <= max_distance
    
    final_mask = (mask_poly & mask_dist).reshape(grid_x.shape)

    traces = []
    
    def get_surface_z(z_values):
        try:
            if method == 'linear':
                grid_z = griddata((points_x, points_y), z_values, (grid_x, grid_y), method='linear')
            else:
                rbf = Rbf(points_x, points_y, z_values, function='linear')
                grid_z = rbf(grid_x, grid_y)
            grid_z[~final_mask] = np.nan
            return grid_z * z_exaggeration
        except Exception: return None

    if show_roof:
        z = get_surface_z(points_z_roof)
        if z is not None:
            name = "Topography" if seam_code == "TOPO" else f'{seam_code} Roof'
            traces.append(go.Surface(z=z, x=grid_x, y=grid_y, colorscale=[[0, seam_color], [1, seam_color]], showscale=False, opacity=opacity, name=name, showlegend=True, legendgroup=seam_code))

    if show_floor and points_z_floor is not None:
        z = get_surface_z(points_z_floor)
        if z is not None:
            traces.append(go.Surface(z=z, x=grid_x, y=grid_y, colorscale=[[0, seam_color], [1, seam_color]], showscale=False, opacity=opacity, name=f'{seam_code} Floor', showlegend=True, legendgroup=seam_code))

    return traces






def generate_triangulated_surfaces(df_master, seam_code, show_roof, show_floor, show_wireframe, opacity, z_exaggeration):
    """
    Generates explicit Delaunay triangulation surfaces connecting boreholes directly.
    """
    traces = []
    
    # 1. Filter Data
    if seam_code == "TOPO":
        df_seam = df_master.drop_duplicates(subset=['BHID'])
        z_roof_col = 'RL'
        z_floor_col = None
        color = '#DEB887'
    else:
        df_seam = df_master[df_master['LCODE'] == seam_code]
        z_roof_col = 'Z_FROM'
        z_floor_col = 'Z_TO'
        color = get_litho_color(seam_code)

    if len(df_seam) < 3: return []

    # 2. Coordinates
    points_2d = df_seam[['X', 'Y']].values
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    
    # 3. Triangulation
    tri = Delaunay(points_2d)
    simplices = tri.simplices
    i = simplices[:, 0]
    j = simplices[:, 1]
    k = simplices[:, 2]

    # Helper to build traces
    def build_surface(z_values, name_suffix):
        # APPLY Z EXAGGERATION HERE (Safe because z_values is a Numpy array)
        z_scaled = z_values * z_exaggeration

        # A. Surface
        traces.append(go.Mesh3d(
            x=x, y=y, z=z_scaled,
            i=i, j=j, k=k,
            color=color, opacity=opacity,
            name=f"{seam_code} {name_suffix}",
            flatshading=True,
            showlegend=True, legendgroup=seam_code,
            hoverinfo='name+z'
        ))

        # B. Wireframe
        if show_wireframe:
            wf_x, wf_y, wf_z = [], [], []
            for tri_idx in simplices:
                pts_idx = [tri_idx[0], tri_idx[1], tri_idx[2], tri_idx[0]]
                wf_x.extend(x[pts_idx]); wf_x.append(None)
                wf_y.extend(y[pts_idx]); wf_y.append(None)
                wf_z.extend(z_scaled[pts_idx]); wf_z.append(None) # Use Scaled Z
            
            traces.append(go.Scatter3d(
                x=wf_x, y=wf_y, z=wf_z,
                mode='lines',
                line=dict(color='black', width=2),
                name=f"{seam_code} Wireframe",
                showlegend=False, legendgroup=seam_code,
                hoverinfo='skip'
            ))

    # 4. Generate
    if show_roof:
        z_roof = df_seam[z_roof_col].values
        build_surface(z_roof, "Roof")
        
    if show_floor and z_floor_col:
        z_floor = df_seam[z_floor_col].values
        build_surface(z_floor, "Floor")

    return traces




@st.cache_data(show_spinner=False)
def plot_3d_model_optimized(df_master, df_bh, z_exaggeration, radius, show_lines, df_boundary, selected_surface_seams, global_surf_config, resolution, render_mode, interp_method, show_wireframe):
    fig = go.Figure()
    
    if not df_master.empty:
        # 1. ORGANIZE ORDER: Coal (Specific Sequence) -> Surfaces -> Waste
        present_lcodes = df_master['LCODE'].unique()
        coal_lcodes_sorted = [lc for lc in COAL_SEAM_LCODES if lc in present_lcodes]
        waste_lcodes = [lc for lc in present_lcodes if lc not in COAL_SEAM_LCODES]
        
        # Helper function to add Borehole Traces
        def add_borehole_trace(lcode_list):
            for lcode in lcode_list:
                data = df_master[df_master['LCODE'] == lcode]
                color = get_litho_color(lcode)
                
                if render_mode == 'Fast (Lines)':
                    x_lines, y_lines, z_lines = [], [], []
                    text_lines = [] 
                    for _, row in data.iterrows():
                        litho_desc = str(row.get('DETAILED LITHOLOGY', '')).strip()
                        hover_str = (
                            f"<b>BHID:</b> {row['BHID']}<br>"
                            f"<b>LCODE:</b> {lcode}<br>"
                            f"<b>Thick:</b> {row.get('WIDTH',0):.2f}m<br>"
                            f"<b>Litho:</b> {litho_desc}"
                        )
                        # Top
                        x_lines.append(row['X']); y_lines.append(row['Y']); z_lines.append(row['Z_FROM'] * z_exaggeration); text_lines.append(hover_str)
                        # Bottom
                        x_lines.append(row['X']); y_lines.append(row['Y']); z_lines.append(row['Z_TO'] * z_exaggeration); text_lines.append(hover_str)
                        # Break
                        x_lines.append(None); y_lines.append(None); z_lines.append(None); text_lines.append("")
                    
                    width = radius * 3 if lcode in COAL_SEAM_LCODES else radius
                    fig.add_trace(go.Scatter3d(
                        x=x_lines, y=y_lines, z=z_lines, mode='lines', 
                        line=dict(color=color, width=width), name=lcode, 
                        showlegend=True, legendgroup=lcode, text=text_lines, hoverinfo='text'
                    ))
                else:
                    # High Res Mode (Mesh)
                    r = radius if lcode in COAL_SEAM_LCODES else radius * 0.8
                    mesh_trace, lines = create_cylinder_mesh(data, r, z_exaggeration, color, lcode, lcode, draw_lines=show_lines)
                    fig.add_trace(mesh_trace)
                    if show_lines and lines[0]:
                        fig.add_trace(go.Scatter3d(x=lines[0], y=lines[1], z=lines[2], mode='lines', line=dict(color='black', width=1), showlegend=False))

        # --- STEP 1: ADD COAL SEAMS ---
        add_borehole_trace(coal_lcodes_sorted)

        # --- STEP 2: ADD SURFACES ---
        if selected_surface_seams and global_surf_config:
            for seam_code in selected_surface_seams:
                config = global_surf_config.get(seam_code, {})
                
                if interp_method == 'Triangulation':
                    # <<< FIX IS HERE: PASS z_exaggeration DIRECTLY >>>
                    traces = generate_triangulated_surfaces(
                        df_master, seam_code, 
                        config.get('show_roof'), config.get('show_floor'), 
                        show_wireframe, config.get('opacity'),
                        z_exaggeration 
                    )
                else:
                    # RBF/Linear Logic
                    traces = generate_seam_surface_optimized(
                        df_master, seam_code, df_boundary, z_exaggeration, resolution, 
                        config.get('show_roof'), config.get('show_floor'), 
                        config.get('opacity'), config.get('max_dist'), 
                        method=interp_method
                    )
                
                for t in traces: fig.add_trace(t)

        # --- STEP 3: ADD WASTE ---
        add_borehole_trace(waste_lcodes)

    # Add Collars
    fig.add_trace(go.Scatter3d(x=df_bh['X'], y=df_bh['Y'], z=df_bh['RL'] * z_exaggeration, mode='markers+text', marker=dict(size=3, color='black'), text=df_bh['BHID'], textposition="top center", name='Collars'))

    fig.update_layout(
        title=f"3D Model ({render_mode} | {interp_method})", 
        scene=dict(xaxis_title='E', yaxis_title='N', zaxis_title='RL', aspectmode='data'), 
        height=800, margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(traceorder='normal')
    )
    return fig


def generate_boundary_nails(df_boundary, spacing=200):
    """
    Generates equidistant points ('Nails') along the boundary polygon perimeter.
    """
    if df_boundary is None or len(df_boundary) < 3:
        return pd.DataFrame()
    
    # Create a Shapely LinearRing from the boundary
    coords = list(zip(df_boundary['X'], df_boundary['Y']))
    boundary_line = LineString(coords)
    
    # Calculate total length and number of points
    total_length = boundary_line.length
    num_points = int(total_length // spacing)
    
    nails = []
    for i in range(num_points):
        # Interpolate point at specific distance
        distance = i * spacing
        point = boundary_line.interpolate(distance)
        nails.append({
            'NAIL_ID': f'BP-{i+1}',
            'X': point.x,
            'Y': point.y
        })
    
    return pd.DataFrame(nails)


def get_slice_profile_data(df_3d, df_boundary, start_pt, end_pt, seam_list, resolution=100, method='RBF'):
    """
    Calculates the vertical profile (Slice) using specific interpolation methods.
    method: 'RBF' (Smooth) or 'Linear' (Jagged/Triangulated)
    """
    # 1. Generate Line Coordinates
    dist_total = np.sqrt((end_pt[0] - start_pt[0])**2 + (end_pt[1] - start_pt[1])**2)
    
    # Create evaluation points along the line
    x_line = np.linspace(start_pt[0], end_pt[0], resolution)
    y_line = np.linspace(start_pt[1], end_pt[1], resolution)
    d_line = np.linspace(0, dist_total, resolution)
    
    slice_data = {}
    
    # 2. Interpolate
    for seam in seam_list:
        seam_data = df_3d[df_3d['LCODE'] == seam]
        if len(seam_data) < 3: continue 
        
        points_x = seam_data['X'].values
        points_y = seam_data['Y'].values
        z_roof = seam_data['Z_FROM'].values
        z_floor = seam_data['Z_TO'].values
        
        try:
            if method == 'Linear' or method == 'Triangulation':
                # LinearNDInterpolator uses Delaunay triangulation under the hood
                # This matches the "Triangulation" or "Fast" look in 3D
                interp_roof = LinearNDInterpolator(list(zip(points_x, points_y)), z_roof)
                interp_floor = LinearNDInterpolator(list(zip(points_x, points_y)), z_floor)
                
                pred_roof = interp_roof(x_line, y_line)
                pred_floor = interp_floor(x_line, y_line)
            else:
                # Default to RBF (Smooth)
                rbf_roof = Rbf(points_x, points_y, z_roof, function='linear')
                rbf_floor = Rbf(points_x, points_y, z_floor, function='linear')
                
                pred_roof = rbf_roof(x_line, y_line)
                pred_floor = rbf_floor(x_line, y_line)
            
            slice_data[seam] = pd.DataFrame({
                'Dist': d_line,
                'Roof': pred_roof,
                'Floor': pred_floor
            })
        except Exception as e:
            continue
            
    return slice_data, dist_total



def plot_cross_section_selector_static(df_bh, df_boundary, df_nails, start_idx, end_idx, influence_dist):
    """
    Plots the map with labels for Boreholes and Boundary Points.
    Visualizes the Influence Buffer Zone.
    """
    fig = go.Figure()
    
    # --- 1. Influence Buffer Zone ---
    if start_idx is not None and end_idx is not None and start_idx != end_idx:
        pt1 = df_nails.iloc[start_idx]
        pt2 = df_nails.iloc[end_idx]
        p1 = np.array([pt1['X'], pt1['Y']])
        p2 = np.array([pt2['X'], pt2['Y']])
        vec_line = p2 - p1
        length = np.linalg.norm(vec_line)
        
        if length > 0:
            unit_vec = vec_line / length
            perp_vec = np.array([-unit_vec[1], unit_vec[0]]) * influence_dist
            c1, c2, c3, c4 = p1 + perp_vec, p1 - perp_vec, p2 - perp_vec, p2 + perp_vec
            
            fig.add_trace(go.Scatter(
                x=[c1[0], c2[0], c3[0], c4[0], c1[0]],
                y=[c1[1], c2[1], c3[1], c4[1], c1[1]],
                fill='toself', fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(color='red', width=1, dash='dot'),
                name=f'Influence (+/- {influence_dist}m)',
                hoverinfo='skip'
            ))

    # --- 2. Block Boundary ---
    fig.add_trace(go.Scatter(
        x=df_boundary['X'], y=df_boundary['Y'], mode='lines', 
        line=dict(color='red', width=2, dash='dash'),
        hoverinfo='skip', name='Block Boundary'
    ))
    
    # --- 3. Boreholes (Markers + Labels) ---
    hover_template = (
        '<b>BHID:</b> %{customdata[0]}<br>' +
        '<b>RL:</b> %{customdata[1]:.2f}<br>' +
        '<b>TD:</b> %{customdata[2]:.2f}<br>' +
        '<b>X:</b> %{x:.2f}<br>' +
        '<b>Y:</b> %{y:.2f}<extra></extra>'
    )
    
    # Markers
    fig.add_trace(go.Scatter(
        x=df_bh['X'], y=df_bh['Y'], mode='markers',
        marker=dict(size=6, color=NON_COAL_COLOR, line=dict(width=1, color=NON_COAL_BORDER)),
        hovertemplate=hover_template,
        customdata=df_bh[['BHID', 'RL', 'DEPTH']],
        name='Boreholes'
    ))

    # BHID Labels (Added Explicitly)
    bh_label_y_offset = -(df_bh['Y'].max() - df_bh['Y'].min()) * 0.02
    fig.add_trace(go.Scatter(
        x=df_bh['X'], y=df_bh['Y'] + bh_label_y_offset, mode='text', 
        text=df_bh['BHID'], textposition="bottom center",
        textfont=dict(size=9, color='black'), 
        showlegend=False, hoverinfo='skip'
    ))

    # --- 4. Boundary Nails (Markers + Labels) ---
    fig.add_trace(go.Scatter(
        x=df_nails['X'], y=df_nails['Y'], mode='markers+text',
        marker=dict(size=6, color='orange', symbol='square', line=dict(width=1, color='black')),
        text=df_nails['NAIL_ID'], textposition="top center",
        textfont=dict(size=9, color='darkorange'),
        name='Boundary Points', hoverinfo='text'
    ))

    # --- 5. Cut Line ---
    if start_idx is not None and end_idx is not None and start_idx != end_idx:
        pt1 = df_nails.iloc[start_idx]
        pt2 = df_nails.iloc[end_idx]
        
        fig.add_trace(go.Scatter(
            x=[pt1['X'], pt2['X']], y=[pt1['Y'], pt2['Y']],
            mode='lines+markers+text', 
            line=dict(color='blue', width=4),
            marker=dict(size=12, color='blue', symbol='star'),
            text=['A', "A'"], textposition=["top left", "top right"],
            textfont=dict(size=16, color='blue', family="Arial Black"),
            name='Section Line', hoverinfo='skip'
        ))

    fig.update_layout(
        title="Section Plan View & Influence Zone",
        xaxis_title="Easting (X)", yaxis_title="Northing (Y)",
        xaxis=dict(showgrid=True, zeroline=False), 
        yaxis=dict(showgrid=True, zeroline=False, scaleanchor="x", scaleratio=1),
        dragmode='pan',
        height=500,  
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='white',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    return fig

def get_projected_boreholes(df_bh, df_litho, start_pt, end_pt, influence_dist):
    """
    Finds boreholes within 'influence_dist' of the line segment.
    Returns a dataframe of lithology intervals with 'PROJECTED_DIST' (x-axis for section).
    Includes all fields needed for detailed hover info.
    """
    p1 = np.array([start_pt['X'], start_pt['Y']])
    p2 = np.array([end_pt['X'], end_pt['Y']])
    vec_line = p2 - p1
    length = np.linalg.norm(vec_line)
    unit_vec = vec_line / length if length > 0 else np.array([0,0])
    
    projected_data = []
    
    unique_bhs = df_bh['BHID'].unique()
    
    for bhid in unique_bhs:
        row = df_bh[df_bh['BHID'] == bhid].iloc[0]
        p_bh = np.array([row['X'], row['Y']])
        vec_bh = p_bh - p1
        
        proj_dist = np.dot(vec_bh, unit_vec)
        
        if -influence_dist <= proj_dist <= length + influence_dist:
            perp_vec = vec_bh - (proj_dist * unit_vec)
            perp_dist = np.linalg.norm(perp_vec)
            
            if perp_dist <= influence_dist:
                # Merge lithology with collar info
                litho_bh = df_litho[df_litho['BHID'] == bhid].copy()
                if not litho_bh.empty:
                    litho_bh['Z_FROM'] = row['RL'] - litho_bh['FROM']
                    litho_bh['Z_TO'] = row['RL'] - litho_bh['TO']
                    litho_bh['PROJECTED_DIST'] = proj_dist
                    # Pass through explicit columns for Tooltip
                    litho_bh['RL_COLLAR'] = row['RL'] 
                    litho_bh['TD'] = row['DEPTH']
                    projected_data.append(litho_bh)
                    
    if projected_data:
        return pd.concat(projected_data)
    return pd.DataFrame()



def data_upload_tab():
    st.header("Data Management and Upload")
    st.markdown("Upload the required CSV files. **Borehole Location and Boundary are mandatory.**")
    
    col_bh, col_boundary, col_litho, col_quality = st.columns(4)
    with col_bh: uploaded_bh_file = st.file_uploader("1. Borehole Collar", type="csv", key="bh_uploader")
    with col_boundary: uploaded_boundary_file = st.file_uploader("2. Boundary", type="csv", key="bound_uploader")
    with col_litho: uploaded_litho_file = st.file_uploader("3. Lithology", type="csv", key="lith_uploader")
    with col_quality: uploaded_quality_file = st.file_uploader("4. Quality", type="csv", key="qual_uploader")

    st.write("")
    if st.button("🚀 Process and Finalize All Data", type="primary", use_container_width=True):
        if not uploaded_bh_file or not uploaded_boundary_file:
            st.error("Missing mandatory data (Collar or Boundary).")
        else:
            with st.spinner("Processing datasets..."):
                st.session_state['df_bh'] = process_bh_data(uploaded_bh_file)
                st.session_state['df_boundary'] = process_boundary_data(uploaded_boundary_file)
                if uploaded_litho_file: st.session_state['df_litho'] = process_litho_data(uploaded_litho_file)
                if uploaded_quality_file: st.session_state['df_quality'] = process_quality_data(uploaded_quality_file)
                
                # --- NEW OPTIMIZATION STEP ---
                if st.session_state['df_bh'] is not None and st.session_state['df_litho'] is not None:
                     st.session_state['df_master'] = create_master_composite(st.session_state['df_bh'], st.session_state['df_litho'])
                
                st.success("All datasets processed and optimized!")
                st.rerun()

    st.markdown("---")
    col_status = st.columns(5) # Increased to 5 to show Optimized Status
    status_map = [
        ('Location', st.session_state['df_bh']), 
        ('Boundary', st.session_state['df_boundary']),
        ('Lithology', st.session_state['df_litho']), 
        ('Quality', st.session_state['df_quality']),
        (' ', st.session_state['df_master'])
    ]
    for i, (name, val) in enumerate(status_map):
        col_status[i].metric(name, "✅ Ready" if val is not None else "❌ Missing")


# --- MAIN EXECUTION ---

# 1. Define the 5 MAIN Tabs
tab_data, tab_block_overview, tab_litho_log, tab_quality, tab_3d_suite = st.tabs([
    "1. Data Management", "2. Block Overview", "3. Borehole Correlation", 
    "4. Quality Analysis", "5. 3D Modeling Suite"
])

# --- TAB 1: DATA ---
with tab_data: data_upload_tab()

if st.session_state['df_bh'] is None or st.session_state['df_boundary'] is None:
    st.stop()

# --- TAB 2: OVERVIEW ---
with tab_block_overview:
    if st.session_state['df_bh'] is not None and st.session_state['df_boundary'] is not None:
        st.plotly_chart(plot_plan_view(st.session_state['df_bh'], st.session_state['df_boundary']), use_container_width=True)
        st.write("---")
        st.header("Data Previews")
        t1, t2 = st.tabs(["Borehole Collar Data", "Block Boundary Data"])
        with t1:
            st.dataframe(st.session_state['df_bh'].drop(columns=['Hover_Label'], errors='ignore').style.set_properties(**{'text-align': 'left'}), use_container_width=True)
        with t2:
            st.dataframe(st.session_state['df_boundary'].style.set_properties(**{'text-align': 'left'}), use_container_width=True)
            


# --- TAB 3: CORRELATION ---
with tab_litho_log:
    if st.session_state['df_litho'] is not None:
        st.subheader("Borehole Correlation")
        bhid_list = st.session_state['df_bh']['BHID'].unique().tolist()
        seam_list_with_none = ['None'] + COAL_SEAM_LCODES
        
        st.plotly_chart(plot_plan_view(st.session_state['df_bh'], st.session_state['df_boundary'], st.session_state['corr_bhid_select']), use_container_width=True, key="corr_map")
        st.markdown("---")
        
        c1, c2, c3, c4 = st.columns([2.5, 1.5, 1.5, 1])
        with c1: 
            selected_bhids = st.multiselect("1. Select Boreholes:", bhid_list, default=bhid_list[:1], key='corr_bhid_select')
        with c2: 
            ref_seam = st.selectbox("2. Select Seam for Correlation:", seam_list_with_none, key='corr_reference_seam')
        with c3: 
            sel_lines = st.multiselect("3. Plot Correlation Lines for:", COAL_SEAM_LCODES, key='corr_lines_select')
        with c4: 
            filt = st.radio("4. Lithology Filter:", ('All Lithology', 'Coal Seams Only'), key='corr_litho_filter')
        
        if selected_bhids:
            fig_corr, excl = plot_litho_correlation(st.session_state['df_bh'], st.session_state['df_litho'], selected_bhids, sel_lines, filt, ref_seam)
            if excl: st.warning(f"**Note:** Borehole(s) `{', '.join(excl)}` were excluded from the plot as they do not contain the reference seam.")
            st.plotly_chart(fig_corr, use_container_width=True)
            


            # --- EXPORT SECTION STARTS HERE ---
                        
            st.markdown("---")
            with st.expander("📄Export Graphic Logs (PDF)", expanded=False):
                st.markdown("Select boreholes to generate ** Graphic Log**.")
                
                # Allow selecting multiple boreholes
                export_bhids = st.multiselect(
                    "Select Boreholes to Export (Max 5 recommended for width):", 
                    selected_bhids, 
                    default=selected_bhids[:2], 
                    key='pdf_export_bhids'
                )
                
                if st.button("Generate PDF", key='btn_gen_pdf'):
                    if export_bhids:
                        with st.spinner("Generating Graphic Logs..."):
                            pdf_data = generate_graphic_log_pdf(st.session_state['df_master'], export_bhids)
                            
                            st.download_button(
                                label="⬇️ Download Graphic Log.pdf",
                                data=pdf_data,
                                file_name="Borehole_Graphic_Log.pdf",
                                mime="application/pdf"
                            )
                    else:
                        st.error("Please select at least one borehole.")


            # --- EXPORT SECTION ENDS HERE ---


            # --- LITHOLOGY TABLE INTEGRATION ---
            st.markdown("---")
            st.subheader("Borehole Lithology Data:")
            col_select, col_button = st.columns([1, 1])
            with col_select:
                selected_bhid_table = st.selectbox("Select Borehole:", selected_bhids, key='litho_table_bhid_select') if selected_bhids else None
            with col_button:
                st.write("")
                if st.button("Coal Only", key='toggle_coal_only_button',type="primary", use_container_width=True):
                    st.session_state['show_coal_only'] = not st.session_state['show_coal_only']

            if selected_bhid_table:
                df_filtered = st.session_state['df_litho'][st.session_state['df_litho']['BHID'] == selected_bhid_table].copy()
                if st.session_state['show_coal_only']:
                    df_filtered = df_filtered[df_filtered['LCODE'].isin(COAL_SEAM_LCODES)]
                    st.info(f"Displaying only Coal Seams for **{selected_bhid_table}**.")
                else:
                    st.info(f"Displaying All Lithologies for **{selected_bhid_table}**.")
                
                def color_rows(s):
                    is_coal = s['LCODE'] in COAL_SEAM_LCODES
                    color = SEAM_COLOR_MAP.get(s['LCODE'], DEFAULT_SEAM_COLOR) if is_coal else 'white'
                    return [f'background-color: {color if is_coal else "white"}'] * len(s)

                if not df_filtered.empty:
                    df_display = df_filtered[['BHID', 'FROM', 'TO', 'WIDTH', 'LCODE', 'DETAILED LITHOLOGY']].rename(columns={'WIDTH': 'THICKNESS (m)', 'DETAILED LITHOLOGY': 'LITHOLOGY DESCRIPTION'}).sort_values(by='FROM').reset_index(drop=True)
                    st.dataframe(df_display.style.apply(color_rows, axis=1).format({'FROM': "{:.2f}", 'TO': "{:.2f}", 'THICKNESS (m)': "{:.2f}"}), use_container_width=True)
    else:
        st.warning("Please upload and process Borehole Lithology data.")








# --- TAB 4: QUALITY ---
with tab_quality:
    if st.session_state['df_quality'] is not None and st.session_state['df_litho'] is not None:
        t_map, t_stats, t_qstats, t_analytics = st.tabs(["Quality Map", "Thickness Stats", "Quality Stats", "Quality Analytics"])
        
        with t_map:
            st.subheader("Quality Distribution")
            plot_quality_plan_view(st.session_state['df_bh'], st.session_state['df_boundary'], st.session_state['df_quality'], st.session_state['df_litho'])
            
        with t_stats:
            st.subheader("Thickness stats")
            c_sel, c_btn = st.columns([3, 1])
            is_avg = st.session_state.get('show_avg_all', False)
            bhid_list = st.session_state['df_bh']['BHID'].unique().tolist()
            with c_sel:
                stat_bhids = st.multiselect("Select Boreholes:", bhid_list, default=bhid_list[:1] if not is_avg else None, key='stats_bhids_new', disabled=is_avg)
            with c_btn:
                st.write(""); st.write("")
                if st.button("Toggle Block-Wide Average", key='toggle_avg_all_new'): st.session_state['show_avg_all'] = not is_avg
            
            bh_analyze = bhid_list if is_avg else stat_bhids
            if bh_analyze:
                df_src = st.session_state['df_litho'][st.session_state['df_litho']['BHID'].isin(bh_analyze)].copy() if not is_avg else st.session_state['df_litho']
                df_tot = df_src[df_src['LCODE'].isin(COAL_SEAM_LCODES)].groupby(['BHID', 'LCODE'], as_index=False)['WIDTH'].sum()
                df_summ = df_tot.groupby('LCODE').agg(AVERAGE_THICKNESS_M=('WIDTH', 'mean')).reset_index()
                
                if not df_summ.empty:
                    f, t = plot_seam_stats(df_summ, f"Avg Thickness (n={len(bh_analyze)})", "Avg Thickness (m)", 'THICKNESS', 'Bar Chart', COAL_SEAM_LCODES)
                    st.plotly_chart(f, use_container_width=True)
                    st.dataframe(t, use_container_width=True)

        with t_qstats:
            st.subheader("Seam-Wise Quality Comparison")
            c1, c2, c3, c4 = st.columns([1.5, 1.5, 3, 1])
            with c1: qp = st.selectbox("Parameter:", list(QUALITY_PARAMETERS.keys()), key='q_stats_p')
            with c2: qs = st.selectbox("Sample Type:", ['All Samples'] + list(st.session_state['df_quality']['SAMPLE_TYPE'].unique()), key='q_stats_s')
            with c3: qb = st.multiselect("Boreholes:", bhid_list, default=bhid_list[:3] if not is_avg else [], key='q_stats_b', disabled=is_avg)
            with c4: 
                st.write(""); st.write("")
                if st.button("Block-Wide Avg", key='toggle_avg_q'): st.session_state['show_avg_all'] = not is_avg
            
            q_bh_list = bhid_list if is_avg else qb
            if q_bh_list:
                df_q_calc = calculate_quality_stats_data(st.session_state['df_quality'], qp, qs, q_bh_list)
                if not df_q_calc.empty:
                    fq, tq = plot_seam_stats(df_q_calc, f"Avg {qp}", f"Avg {qp}", qp, 'Bar Chart', COAL_SEAM_LCODES)
                    st.plotly_chart(fq, use_container_width=True)
                    st.dataframe(tq, use_container_width=True)
        
        with t_analytics:
            t_cross, t_dist = st.tabs(["Cross-Plot", "Distribution"])
            with t_cross:
                st.subheader("Bivariate Analysis")
                c1, c2, c3, c4 = st.columns(4)
                valid_params = [k for k in QUALITY_PARAMETERS.keys() if k in st.session_state['df_quality'].columns]
                
                with c1: xp_a = st.selectbox("X-Axis:", valid_params, index=0, key='xp_a')
                with c2: yp_a = st.selectbox("Y-Axis:", valid_params, index=1 if len(valid_params) > 1 else 0, key='yp_a')
                with c3: sp_a = st.selectbox("Seam:", COAL_SEAM_LCODES, key='sp_a')
                with c4: stp_a = st.selectbox("Sample Type:", ['All Samples'] + list(st.session_state['df_quality']['SAMPLE_TYPE'].unique()), key='stp_a')
                
                fig_cross = plot_quality_crossplot(st.session_state['df_quality'], sp_a, stp_a, xp_a, yp_a)
                st.plotly_chart(fig_cross, use_container_width=True)
            
            with t_dist:
                st.info("Additional distribution analysis (Histograms).")
    else:
        st.warning("Please upload Quality and Lithology Data.")


# --- TAB 5: 3D MODELING SUITE (PARENT TAB) ---
with tab_3d_suite:
    # Check Data
    if 'df_master' not in st.session_state or st.session_state['df_master'] is None:
        st.warning("Please upload Lithology data and Process it in the Data Tab.")
    else:
        # Define 3 Sub-Tabs inside the Parent Tab
        tab_3d, tab_xsec, tab_faults = st.tabs(["3D Modeling", "Cross-Section", "Fault Detection"])

        # --- SUB-TAB 1: 3D MODELING ---
        with tab_3d:
            st.subheader("3D Geological Model")
            with st.expander("Model Configuration", expanded=True):
                c1, c2 = st.columns(2)
                bhid_list = st.session_state['df_bh']['BHID'].unique().tolist()
                
                with c1:
                    st.markdown("##### 1. Structure & Performance")
                    perf_c1, perf_c2 = st.columns(2)
                    with perf_c1: 
                        render_mode = st.radio("Render Mode:", ["Fast (Lines)", "High Res (Cylinders)"], index=0)
                    with perf_c2: 
                        interp_method = st.radio("Interp Method:", ["Triangulation", "Linear (Fast)", "RBF (Smooth)"], index=0)

                    sub_c1, sub_c2 = st.columns(2)
                    with sub_c1:
                        select_all_bh = st.checkbox("All Boreholes", value=True, key='3d_all_bh')
                        selected_bhids_3d = bhid_list if select_all_bh else st.multiselect("Boreholes:", bhid_list, default=bhid_list[:5], key='3d_bh_sel')
                    with sub_c2:
                        select_all_seams = st.checkbox("All Seams", value=True, key='3d_all_seams')
                        selected_seams_3d = COAL_SEAM_LCODES if select_all_seams else st.multiselect("Seams:", COAL_SEAM_LCODES, default=COAL_SEAM_LCODES[:1], key='3d_seam_sel')
                    
                    z_exaggeration = st.number_input("Vertical Exaggeration:", 1.0, 100.0, 1.0, 0.5)
                    radius_val = st.number_input("BH Radius (m):", 0.1, 50.0, 4.0, 0.5)
                    
                with c2:
                    st.markdown("##### 2. Surfaces")
                    
                    show_wireframe = st.checkbox("Show Wireframe (Triangulation only)", value=True)
                    
                    is_grid_method = interp_method != 'Triangulation'
                    
                    interp_limit = st.slider("Interpolation Limit (m)", 100, 2000, 500, 50, disabled=not is_grid_method)
                    resolution_val = st.slider("Grid Resolution", 50, 300, 200, 10, disabled=not is_grid_method)
                    global_opacity = st.slider("Opacity", 0.1, 1.0, 1.0)
                    
                    show_topo = st.checkbox("Show Topography", value=False)
                    
                    c_surf_r, c_surf_f = st.columns(2)
                    with c_surf_r: selected_roofs = st.multiselect("Roofs:", COAL_SEAM_LCODES, key='3d_roofs')
                    with c_surf_f: selected_floors = st.multiselect("Floors:", COAL_SEAM_LCODES, key='3d_floors')

                st.markdown("---")
                generate_btn = st.button("🚀 Generate 3D Model", type="primary", use_container_width=True)

            if generate_btn and selected_bhids_3d:
                with st.spinner("Rendering..."):
                    df_master_filt = st.session_state['df_master'][st.session_state['df_master']['BHID'].isin(selected_bhids_3d)]
                    df_3d_input = df_master_filt.copy() 
                    
                    target_seams = []
                    surf_conf = {}
                    if show_topo:
                        target_seams.append("TOPO")
                        surf_conf["TOPO"] = {'show_roof': True, 'show_floor': False, 'opacity': global_opacity, 'max_dist': interp_limit}
                    
                    for s in set(selected_roofs + selected_floors):
                        target_seams.append(s)
                        surf_conf[s] = {'show_roof': s in selected_roofs, 'show_floor': s in selected_floors, 'opacity': global_opacity, 'max_dist': interp_limit}

                    fig_3d = plot_3d_model_optimized(
                        df_3d_input, 
                        st.session_state['df_bh'][st.session_state['df_bh']['BHID'].isin(selected_bhids_3d)], 
                        z_exaggeration, radius_val, False, 
                        st.session_state['df_boundary'], 
                        target_seams, surf_conf, resolution_val, 
                        render_mode, interp_method, show_wireframe
                    )
                    st.session_state['fig_3d_generated'] = fig_3d
            
            if 'fig_3d_generated' in st.session_state:
                st.plotly_chart(st.session_state['fig_3d_generated'], use_container_width=True)

        # --- SUB-TAB 2: CROSS-SECTION ---
        with tab_xsec:
            # Initialize Nails if not present
            if 'df_nails' not in st.session_state:
                st.session_state['df_nails'] = generate_boundary_nails(st.session_state['df_boundary'], spacing=200)
            
            nails_df = st.session_state['df_nails']
            nails_list = nails_df['NAIL_ID'].tolist()
            
            st.subheader("Section Plan View")
            
            # Default indices
            def_start = 0
            def_end = min(5, len(nails_list)-1)
            
            curr_start = st.session_state.get('xs_start', nails_list[def_start])
            curr_end = st.session_state.get('xs_end', nails_list[def_end])
            curr_inf = st.session_state.get('xs_inf', 150.0)
            
            try:
                s_idx = nails_list.index(curr_start)
                e_idx = nails_list.index(curr_end)
            except ValueError:
                s_idx, e_idx = def_start, def_end

            fig_map = plot_cross_section_selector_static(
                st.session_state['df_bh'], 
                st.session_state['df_boundary'], 
                nails_df, s_idx, e_idx, curr_inf
            )
            st.plotly_chart(fig_map, use_container_width=True)

            st.subheader("Settings")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: sel_start = st.selectbox("Start Point (A)", nails_list, index=s_idx, key='xs_start')
            with c2: sel_end = st.selectbox("End Point (A')", nails_list, index=e_idx, key='xs_end')
            with c3: influence = st.number_input("Influence Width (m)", min_value=10.0, max_value=500.0, value=curr_inf, step=10.0, key='xs_inf')
            with c4: ve_slider = st.slider("Vertical Exaggeration", 1.0, 20.0, 1.0, 0.5, key='xs_ve')
            with c5: xsec_method = st.selectbox("Method", ["RBF (Smooth)", "Linear", "Triangulation"], index=0, key='xs_method')

            generate_btn_xsec = st.button("Generate Vertical Section", type="primary", use_container_width=True)

            if generate_btn_xsec:
                st.markdown("### 2. Vertical Profile (A - A')")
                
                start_idx = nails_list.index(sel_start)
                end_idx = nails_list.index(sel_end)
                
                if start_idx == end_idx:
                    st.error("Start and End points must be different.")
                else:
                    pt_start = nails_df.iloc[start_idx]
                    pt_end = nails_df.iloc[end_idx]
                    
                    all_bhids = st.session_state['df_bh']['BHID'].unique()
                    df_3d_all = prepare_3d_data(st.session_state['df_bh'], st.session_state['df_litho'], all_bhids, COAL_SEAM_LCODES, include_waste=False)
                    
                    with st.spinner("Processing Model..."):
                        method_clean = xsec_method.split(' ')[0] 
                        slice_results, total_dist = get_slice_profile_data(
                            df_3d_all, st.session_state['df_boundary'],
                            (pt_start['X'], pt_start['Y']), (pt_end['X'], pt_end['Y']), 
                            COAL_SEAM_LCODES,
                            method=method_clean 
                        )
                        
                        df_proj_bh = get_projected_boreholes(
                            st.session_state['df_bh'], st.session_state['df_litho'],
                            pt_start, pt_end, influence
                        )
                    
                    fig_profile = go.Figure()
                    
                    # Topography
                    try:
                        topo_x = st.session_state['df_bh']['X'].values
                        topo_y = st.session_state['df_bh']['Y'].values
                        topo_z = st.session_state['df_bh']['RL'].values
                        
                        num_points = 100 
                        x_line_topo = np.linspace(pt_start['X'], pt_end['X'], num_points)
                        y_line_topo = np.linspace(pt_start['Y'], pt_end['Y'], num_points)
                        d_line_topo = np.linspace(0, total_dist, num_points)
                        
                        if len(topo_x) >= 3:
                            if method_clean == 'Linear' or method_clean == 'Triangulation':
                                 interp_topo = LinearNDInterpolator(list(zip(topo_x, topo_y)), topo_z)
                                 topo_profile_z = interp_topo(x_line_topo, y_line_topo)
                            else:
                                 rbf_topo = Rbf(topo_x, topo_y, topo_z, function='linear')
                                 topo_profile_z = rbf_topo(x_line_topo, y_line_topo)
                            
                            fig_profile.add_trace(go.Scatter(
                                x=d_line_topo, y=topo_profile_z, mode='lines',
                                line=dict(color='#8B4513', width=2, dash='solid'),
                                name='Topography Surface', hoverinfo='text',
                                hovertext=[f'Topo RL: {z:.2f}m' for z in topo_profile_z]
                            ))
                    except Exception as e: pass

                    # Plot Seams
                    for seam in reversed(COAL_SEAM_LCODES):
                        if seam in slice_results:
                            df_s = slice_results[seam]
                            color = get_litho_color(seam)
                            l_rank = COAL_SEAM_LCODES.index(seam)
                            
                            fig_profile.add_trace(go.Scatter(
                                x=df_s['Dist'], y=df_s['Floor'], mode='lines',
                                line=dict(width=0), showlegend=False, hoverinfo='skip', legendrank=l_rank 
                            ))
                            fig_profile.add_trace(go.Scatter(
                                x=df_s['Dist'], y=df_s['Roof'], mode='lines',
                                line=dict(width=0.5, color='black'),
                                fill='tonexty', fillcolor=color, opacity=0.8,
                                name=seam, legendgroup=seam, showlegend=True, legendrank=l_rank
                            ))

                    # Plot Boreholes
                    if not df_proj_bh.empty:
                        df_proj_bh['COLOR'] = df_proj_bh['LCODE'].apply(get_litho_color)
                        df_proj_bh['HOVER_TEXT'] = (
                            '<b>BHID:</b> ' + df_proj_bh['BHID'] + '<br>' + 
                            '<b>RL:</b> ' + df_proj_bh['Z_FROM'].round(2).astype(str) + ' to ' + df_proj_bh['Z_TO'].round(2).astype(str) + ' m<br>'+ 
                            '<b>Depth:</b> ' + df_proj_bh['FROM'].round(2).astype(str) + ' to ' + df_proj_bh['TO'].round(2).astype(str) + ' m<br>' + 
                            '<b>Thickness:</b> ' + df_proj_bh['WIDTH'].round(2).astype(str) + ' m<br>' + 
                            '<b>LCODE:</b> ' + df_proj_bh['LCODE'] + '<br>' + 
                            '<b>Lithology:</b> ' + df_proj_bh['DETAILED LITHOLOGY'].fillna('')
                        )

                        fig_profile.add_trace(go.Bar(
                            x=df_proj_bh['PROJECTED_DIST'], y=df_proj_bh['WIDTH'], base=df_proj_bh['Z_TO'], 
                            marker=dict(color=df_proj_bh['COLOR'], line=dict(width=0.5, color='black')),
                            width=total_dist * 0.012, name='Projected Borehole',
                            text=df_proj_bh['LCODE'], textposition='inside', textfont=dict(color=PLOT_TEXT_COLOR, size=9), 
                            orientation='v', hovertext=df_proj_bh['HOVER_TEXT'], showlegend=True, legendgroup='Boreholes'
                        ))
                        
                        bh_labels = df_proj_bh.groupby('BHID').agg({'PROJECTED_DIST': 'first', 'Z_FROM': 'max', 'RL_COLLAR': 'first', 'TD': 'first'}).reset_index()
                        label_hover = (
                            '<b>' + bh_labels['BHID'] + '</b><br>' +
                            'RL: ' + bh_labels['RL_COLLAR'].round(2).astype(str) + '<br>' +
                            'TD: ' + bh_labels['TD'].round(2).astype(str)
                        )
                        fig_profile.add_trace(go.Scatter(
                            x=bh_labels['PROJECTED_DIST'], y=bh_labels['Z_FROM'] + 5,
                            mode='text', text=bh_labels['BHID'],
                            textposition='top center', textfont=dict(size=10, color='black', weight='bold'),
                            hovertext=label_hover, hoverinfo='text', showlegend=False, legendgroup='Boreholes'
                        ))

                    fig_profile.update_layout(
                        title=f"Section {sel_start} - {sel_end} (Length: {total_dist:.0f}m) | Method: {method_clean}",
                        xaxis_title="Distance along Section (m)", yaxis_title="Elevation (RL)",
                        height=700, hovermode="closest",
                        yaxis=dict(scaleanchor="x", scaleratio=ve_slider), 
                        xaxis=dict(range=[0, total_dist], showgrid=True, zeroline=True, constrain='domain'),
                        plot_bgcolor='white', barmode='overlay',
                        legend=dict(font=dict(size=10), tracegroupgap=0, yanchor="top", y=1, xanchor="left", x=1.02)
                    )
                    st.plotly_chart(fig_profile, use_container_width=True)

        # --- SUB-TAB 3: FAULT DETECTION ---
        with tab_faults:
            st.subheader("Structural Anomaly & Fault Detection")
            st.markdown("This tool analyzes the **Floor Elevation** of a selected seam to identify high dip angles (>20-45°) indicating potential **Faults**.")
            
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1: fault_seam = st.selectbox("Select Reference Seam:", COAL_SEAM_LCODES, index=0, key='fault_seam_sel')
            with col_c2: dip_thresh = st.slider("Dip Threshold (Degrees):", 5.0, 85.0, 20.0, 5.0)
            with col_c3: 
                st.write(""); run_fault_btn = st.button("Analyze Structure", type="primary", use_container_width=True)
            
            st.markdown("---")
            if run_fault_btn:
                with st.spinner(f"Analyzing structure of {fault_seam}..."):
                    df_anomalies, df_seam_pts = calculate_fault_anomalies(st.session_state['df_master'], fault_seam, dip_thresh)
                    fig_fault = plot_fault_map(df_anomalies, df_seam_pts, st.session_state['df_boundary'], fault_seam, dip_thresh)
                    st.plotly_chart(fig_fault, use_container_width=True)
                    
                    if not df_anomalies.empty:
                        st.warning(f"Found {len(df_anomalies)} connections with dip > {dip_thresh}°.")
                        df_display = df_anomalies[['BHID_A', 'BHID_B', 'Dist_H', 'Throw_m', 'Dip_Deg']].copy().sort_values('Dip_Deg', ascending=False)
                        st.dataframe(df_display.style.format({'Dist_H': "{:.1f}", 'Throw_m': "{:.2f}", 'Dip_Deg': "{:.1f}"}).background_gradient(subset=['Dip_Deg'], cmap='Reds'), use_container_width=True)
                    else:
                        st.success(f"No structural anomalies found greater than {dip_thresh}°.")
