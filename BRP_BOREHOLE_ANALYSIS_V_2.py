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


# --- CONFIGURATION ---

# List of LCODEs that represent Coal Seams (Dark Color) - MANDATORY SEQUENCE ORDER
COAL_SEAM_LCODES = [
    'PAR', 'LAJ4', 'L4B', 'LAJ3', 'L2T3', 'L2T2', 'L2T1', 'L2T1T', 'L2T1B',
    'L2B', 'LAJ1', 'LL1', 'R5', 'R5T', 'R5B', 'R4', 'R3T', 'R3B', 'R12',
    'IBT', 'IBB'
]

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

# --- FIXED COLOR MAP (Matched to your Legend Image) ---
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

# --- FIX: Explicitly Define Default Color ---
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
        plot_bgcolor='white', # Optimized for Light Theme
        paper_bgcolor='white', # Optimized for Light Theme
        hovermode="closest", height=700,
        legend=dict(font=dict(size=10)),
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
        title, yaxis_title = "Geological Correlation (True Elevation)", "Elevation (RL) above MSL (m)"
        yaxis_config = dict(title=yaxis_title, showgrid=True, zeroline=True, zerolinecolor=PLOT_TEXT_COLOR, range=[min_y_range, max_y_range])
    
    # We remove the old custom lithology legend logic at the end since we use the dummy traces now
    
    fig.update_layout(
        title_text=title, title_font=dict(size=16),
        xaxis=dict(title="Cumulative Distance along Section (m)", tickvals=df_selected_bh['CUM_DISTANCE'], ticktext=[f'{d:.0f} m' for d in df_selected_bh['CUM_DISTANCE']], showgrid=False, zeroline=False),
        yaxis=yaxis_config, height=700, barmode='stack', 
        plot_bgcolor='white', 
        paper_bgcolor='white', 
        font=dict(color=PLOT_TEXT_COLOR),
        legend=dict(font=dict(size=10), x=1.02, y=1, bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1),
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


# --- 3D MODELLING FUNCTIONS ---


@st.cache_data(show_spinner=False)
def prepare_3d_data(df_bh, df_litho, selected_bhids, selected_seams_filter, include_waste):
    """
    Merges collar RL with lithology depths to calculate absolute Z coordinates.
    Filters for selected boreholes.
    Applies logic for filtering by specific coal seams + optional waste.
    """
    # 1. Filter by Borehole
    df_bh_sel = df_bh[df_bh['BHID'].isin(selected_bhids)].copy()
    df_litho_sel = df_litho[df_litho['BHID'].isin(selected_bhids)].copy()
    
    # 2. Filter by Seam Selection
    if selected_seams_filter:
        # Condition A: It is one of the selected coal seams
        cond_coal = df_litho_sel['LCODE'].isin(selected_seams_filter)
        
        if include_waste:
            # Condition B: It is NOT a coal seam (keep all waste)
            cond_waste = ~df_litho_sel['LCODE'].isin(COAL_SEAM_LCODES)
            # Keep if (Selected Coal) OR (Waste)
            df_litho_sel = df_litho_sel[cond_coal | cond_waste]
        else:
            # Keep ONLY the selected coal seams (floating coal mode)
            df_litho_sel = df_litho_sel[cond_coal]
    
    # Merge to get Collar RL into the Lithology table
    df_3d = pd.merge(df_litho_sel, df_bh_sel[['BHID', 'X', 'Y', 'RL']], on='BHID', how='inner')
    
    # Calculate Absolute Z Coordinates (Elevation)
    # The top of the hole is RL. Depth 0 corresponds to RL.
    df_3d['Z_FROM'] = df_3d['RL'] - df_3d['FROM']
    df_3d['Z_TO'] = df_3d['RL'] - df_3d['TO']
    
    # Assign Colors immediately to simplify plotting logic
    df_3d['COLOR'] = df_3d['LCODE'].apply(get_litho_color)
    
    return df_3d

def create_cylinder_mesh(df_segment, radius, z_exaggeration, color, name, lcode, draw_lines=True):
    """
    Creates a 3D Mesh (Hexagonal Prism/Cylinder) for a group of segments.
    Also returns the wireframe coordinates for the bottom face (hairline).
    """
    x_coords, y_coords, z_coords = [], [], []
    i_indices, j_indices, k_indices = [], [], []
    hover_texts = []
    
    # Wireframe line lists
    line_x, line_y, line_z = [], [], []
    
    # 6-sided polygon (hexagon) for efficiency
    angles = np.linspace(0, 2*np.pi, 7)[:-1] 
    cos_a = np.cos(angles) * radius
    sin_a = np.sin(angles) * radius
    
    # Append the first point to close the loop for lines
    cos_a_loop = np.append(cos_a, cos_a[0])
    sin_a_loop = np.append(sin_a, sin_a[0])
    
    current_vertex_offset = 0
    
    for _, row in df_segment.iterrows():
        cx, cy = row['X'], row['Y']
        z_top = row['Z_FROM'] * z_exaggeration
        z_bot = row['Z_TO'] * z_exaggeration
        
        # --- MESH GENERATION ---
        
        # Top Circle Vertices
        x_coords.extend(cx + cos_a)
        y_coords.extend(cy + sin_a)
        z_coords.extend([z_top] * 6)
        
        # Bottom Circle Vertices
        x_coords.extend(cx + cos_a)
        y_coords.extend(cy + sin_a)
        z_coords.extend([z_bot] * 6)
        
        # Construct Hover Text for this specific interval
        hover_info = (
            f"<b>BHID:</b> {row['BHID']}<br>"
            f"<b>RL:</b> {row['Z_FROM']:.2f} to {row['Z_TO']:.2f} m<br>"
            f"<b>Depth:</b> {row['FROM']:.2f} m "
            f"to {row['TO']:.2f} m<br>"
            f"<b>Thickness:</b> {row['WIDTH']:.2f} m<br>"
            f"<b>LCODE:</b> {lcode}<br>"
            f"<b>Lithology:</b> {row.get('DETAILED LITHOLOGY', '')}"
        )
        # Duplicate hover text for all 12 vertices of this cylinder
        hover_texts.extend([hover_info] * 12)
        
        # Faces (Side walls)
        for s in range(6):
            t1 = current_vertex_offset + s
            t2 = current_vertex_offset + (s + 1) % 6
            b1 = current_vertex_offset + 6 + s
            b2 = current_vertex_offset + 6 + (s + 1) % 6
            
            # Triangle 1 (t1, b1, t2)
            i_indices.append(t1); j_indices.append(b1); k_indices.append(t2)
            # Triangle 2 (t2, b1, b2)
            i_indices.append(t2); j_indices.append(b1); k_indices.append(b2)
            
        current_vertex_offset += 12 
        
        # --- WIREFRAME GENERATION (Bottom Face) ---
        if draw_lines:
            # Create a loop for the bottom face
            line_x.extend(cx + cos_a_loop)
            line_y.extend(cy + sin_a_loop)
            line_z.extend([z_bot] * 7) # 7 points to close the hexagon
            
            # Add None to break the line segment
            line_x.append(None)
            line_y.append(None)
            line_z.append(None)
        
    mesh_trace = go.Mesh3d(
        x=x_coords, y=y_coords, z=z_coords,
        i=i_indices, j=j_indices, k=k_indices,
        color=color,
        name=name,
        hoverinfo='text',
        text=hover_texts, # Per-vertex hover text
        flatshading=True,
        showlegend=True,
        legendgroup='Coal' if lcode in COAL_SEAM_LCODES else 'Litho'
    )
    
    return mesh_trace, (line_x, line_y, line_z)



@st.cache_data(show_spinner=False)
def generate_seam_surface_rbf(df_3d, seam_code, df_boundary, z_exaggeration, resolution, show_roof, show_floor, opacity, max_distance=1000):
    """
    Generates 3D surface traces using RBF interpolation with Distance Masking.
    """
    
    # 1. DATA SELECTION LOGIC
    if seam_code == "TOPO":
        # TOPOGRAPHY MODE
        topo_data = df_3d.drop_duplicates(subset=['BHID'])
        if len(topo_data) < 3: return []
        points_x = topo_data['X'].values
        points_y = topo_data['Y'].values
        points_z_roof = topo_data['RL'].values
        # TOPO has no floor, we ignore it
        points_z_floor = None 
        seam_color = '#DEB887' # BurlyWood / Tan
    else:
        # COAL SEAM MODE
        seam_data = df_3d[df_3d['LCODE'] == seam_code]
        if len(seam_data) < 3: return []
        points_x = seam_data['X'].values
        points_y = seam_data['Y'].values
        points_z_roof = seam_data['Z_FROM'].values
        points_z_floor = seam_data['Z_TO'].values
        seam_color = get_litho_color(seam_code)

    # 2. GRID GENERATION
    min_x, max_x = df_boundary['X'].min(), df_boundary['X'].max()
    min_y, max_y = df_boundary['Y'].min(), df_boundary['Y'].max()
    
    # Create 1D arrays for the grid
    grid_x_1d = np.linspace(min_x, max_x, resolution)
    grid_y_1d = np.linspace(min_y, max_y, resolution)
    grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

    # Flatten for masking calculations
    grid_flat = np.column_stack((grid_x.flatten(), grid_y.flatten()))

    # 3. MASKS (Boundary + Distance)
    
    # A. Polygon Boundary Mask
    poly_path = Path(list(zip(df_boundary['X'], df_boundary['Y'])))
    mask_poly = poly_path.contains_points(grid_flat)
    
    # B. Distance Mask (Confidence Limit)
    # Calculate distance from every grid point to the nearest actual data point
    data_points = np.column_stack((points_x, points_y))
    dists = cdist(grid_flat, data_points, metric='euclidean')
    min_dists = dists.min(axis=1)
    mask_dist = min_dists <= max_distance
    
    # Combine Masks
    final_mask = mask_poly & mask_dist
    final_mask = final_mask.reshape(grid_x.shape)

    traces = []

    # 4. RBF INTERPOLATION HELPER
    def get_rbf_surface(z_values):
        try:
            rbf = Rbf(points_x, points_y, z_values, function='linear')
            grid_z = rbf(grid_x, grid_y)
            # Apply the combined mask
            grid_z[~final_mask] = np.nan
            return grid_z * z_exaggeration
        except Exception:
            return None

    # 5. TRACE CONSTRUCTION
    
    # ROOF (Or Topo)
    if show_roof:
        grid_z_roof = get_rbf_surface(points_z_roof)
        if grid_z_roof is not None:
            name_str = "Topography" if seam_code == "TOPO" else f'{seam_code} Roof'
            traces.append(go.Surface(
                z=grid_z_roof, x=grid_x, y=grid_y, 
                colorscale=[[0, seam_color], [1, seam_color]],
                showscale=False, opacity=opacity, 
                name=name_str, hoverinfo='all', 
                showlegend=True, legendgroup=seam_code
            ))

    # FLOOR (Skip for TOPO)
    if show_floor and points_z_floor is not None:
        grid_z_floor = get_rbf_surface(points_z_floor)
        if grid_z_floor is not None:
            traces.append(go.Surface(
                z=grid_z_floor, x=grid_x, y=grid_y, 
                colorscale=[[0, seam_color], [1, seam_color]], 
                showscale=False, opacity=opacity, 
                name=f'{seam_code} Floor', hoverinfo='all', 
                showlegend=True, legendgroup=seam_code
            ))

    return traces




@st.cache_data(show_spinner=False)
def plot_3d_model_combined(df_3d, df_bh_filtered, z_exaggeration, radius, show_lines, df_boundary, 
                           selected_surface_seams, global_surf_config, resolution): # <--- ADDED ARGUMENT
    fig = go.Figure()
    all_lines_x, all_lines_y, all_lines_z = [], [], []
    
    # --- OPTIMIZED BOREHOLE STICKS ---
    unique_lcodes = df_3d['LCODE'].unique()
    
    for lcode in unique_lcodes:
        lcode_data = df_3d[df_3d['LCODE'] == lcode]
        color = get_litho_color(lcode)
        r = radius if lcode in COAL_SEAM_LCODES else radius * 0.8
        mesh_trace, lines = create_cylinder_mesh(lcode_data, r, z_exaggeration, color, lcode, lcode, draw_lines=show_lines)
        mesh_trace.update(showlegend=True, name=lcode)
        fig.add_trace(mesh_trace)
        if show_lines:
            all_lines_x.extend(lines[0]); all_lines_y.extend(lines[1]); all_lines_z.extend(lines[2])

    # --- BOUNDARY WIREFRAME ---
    if show_lines and all_lines_x:
        fig.add_trace(go.Scatter3d(
            x=all_lines_x, y=all_lines_y, z=all_lines_z, 
            mode='lines', line=dict(color='black', width=1), 
            name='Interval Boundaries', showlegend=False, hoverinfo='skip'
        ))

    # --- COLLAR POINTS ---
    fig.add_trace(go.Scatter3d(
        x=df_bh_filtered['X'], y=df_bh_filtered['Y'], z=df_bh_filtered['RL'] * z_exaggeration, 
        mode='markers+text', marker=dict(size=4, color='black'), 
        text=df_bh_filtered['BHID'], textposition="top center", 
        name='Collars'
    ))

    # 2. Surfaces (Dynamic Config applied per seam)
    if selected_surface_seams and global_surf_config: 
        for seam_code in selected_surface_seams:
            config = global_surf_config.get(seam_code, {})
            
            traces = generate_seam_surface_rbf(
                df_3d, 
                seam_code, 
                df_boundary, 
                z_exaggeration, 
                resolution, # <--- USED HERE (Instead of 50)
                config.get('show_roof', True),
                config.get('show_floor', False),
                config.get('opacity', 0.5),
                config.get('max_dist', 1000)
            )
            for trace in traces:
                fig.add_trace(trace)

    fig.update_layout(
        title="3D Geological Model (Optimized)",
        scene=dict(
            xaxis_title='Easting', yaxis_title='Northing', 
            zaxis_title=f'Elevation (x{z_exaggeration})',
            aspectmode='data'
        ),
        height=850, margin=dict(l=0, r=0, b=0, t=50)
    )
    return fig






# --- TAB DEFINITION ---
def data_upload_tab():
    st.header("Data Management and Upload")
    st.markdown("Upload the required CSV files. **Borehole Location and Boundary are mandatory to proceed.**")
    
    # 1. File Uploaders Layout
    col_bh, col_boundary, col_litho, col_quality = st.columns(4)
    with col_bh:
        uploaded_bh_file = st.file_uploader("1. Borehole Collar Data", type="csv", key="bh_uploader_tab")
    with col_boundary:
        uploaded_boundary_file = st.file_uploader("2. Block Boundary Data", type="csv", key="boundary_uploader_tab")
    with col_litho:
        uploaded_litho_file = st.file_uploader("3. Lithology Data", type="csv", key="litho_uploader_tab_litho")
    with col_quality:
        uploaded_quality_file = st.file_uploader("4. Quality Data", type="csv", key="quality_uploader_tab_quality")

    st.write("") # Spacer
    
    # 2. Single Unified Process Button
    if st.button("🚀 Process and Finalize All Data", type="primary", use_container_width=True):
        # Verification check for mandatory files
        if not uploaded_bh_file or not uploaded_boundary_file:
            st.error("Missing mandatory data! Please upload both Borehole Location and Block Boundary files.")
        else:
            with st.spinner("Processing datasets..."):
                # Process each file if it has been uploaded
                st.session_state['df_bh'] = process_bh_data(uploaded_bh_file)
                st.session_state['df_boundary'] = process_boundary_data(uploaded_boundary_file)
                
                # Optional files
                if uploaded_litho_file:
                    st.session_state['df_litho'] = process_litho_data(uploaded_litho_file)
                if uploaded_quality_file:
                    st.session_state['df_quality'] = process_quality_data(uploaded_quality_file)
                
                st.success("All uploaded datasets processed successfully!")
                st.rerun() # Refresh to update metrics and unlock other tabs

    # 3. Current Status Section
    st.markdown("---")
    st.subheader("Current Data Status")
    col_status = st.columns(4)
    data_status = {
        'Location': st.session_state['df_bh'] is not None, 
        'Boundary': st.session_state['df_boundary'] is not None, 
        'Lithology': st.session_state['df_litho'] is not None, 
        'Quality': st.session_state['df_quality'] is not None
    }
    
    status_items = list(data_status.items())
    for i in range(4):
        name, loaded = status_items[i]
        col_status[i].metric(name, "✅ Loaded" if loaded else "❌ Missing")



# --- MAIN EXECUTION ---

tab_data, tab_block_overview, tab_litho_log, tab_quality, tab_3d = st.tabs([
    "1. Data Management", "2. Block Overview", "3. Borehole Correlation", "4. Quality Analysis", "5. 3D Modeling"
])

with tab_data: data_upload_tab()

if st.session_state['df_bh'] is None or st.session_state['df_boundary'] is None:
    st.stop()



with tab_block_overview:
    if st.session_state['df_bh'] is not None and st.session_state['df_boundary'] is not None:
        st.plotly_chart(plot_plan_view(st.session_state['df_bh'], st.session_state['df_boundary']), use_container_width=True)
        st.write("---")
        st.header("Data Previews")
        t1, t2 = st.tabs(["Borehole Location Data", "Block Boundary Data"])
        with t1:
            st.dataframe(st.session_state['df_bh'].drop(columns=['Hover_Label'], errors='ignore').style.set_properties(**{'text-align': 'left'}), use_container_width=True)
        with t2:
            st.dataframe(st.session_state['df_boundary'].style.set_properties(**{'text-align': 'left'}), use_container_width=True)



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
            with c3: qb = st.multiselect("Boreholes:", bhid_list, default=bhid_list[:1] if not is_avg else [], key='q_stats_b', disabled=is_avg)
            with c4: 
                st.write(""); st.write("")
                if st.button("Toggle Block-Wide Avg", key='toggle_avg_q'): st.session_state['show_avg_all'] = not is_avg
            
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
                
                # Filter params to only those that exist in df_quality
                valid_params = [k for k in QUALITY_PARAMETERS.keys() if k in st.session_state['df_quality'].columns]
                
                with c1: xp_a = st.selectbox("X-Axis:", valid_params, index=0, key='xp_a')
                with c2: yp_a = st.selectbox("Y-Axis:", valid_params, index=1 if len(valid_params) > 1 else 0, key='yp_a')
                with c3: sp_a = st.selectbox("Seam:", COAL_SEAM_LCODES, key='sp_a')
                with c4: stp_a = st.selectbox("Sample Type:", ['All Samples'] + list(st.session_state['df_quality']['SAMPLE_TYPE'].unique()), key='stp_a')
                
                # Logic: Call the plotting function with the specific keys used above
                fig_cross = plot_quality_crossplot(st.session_state['df_quality'], sp_a, stp_a, xp_a, yp_a)
                st.plotly_chart(fig_cross, use_container_width=True)
            
            with t_dist:
                st.info("Additional distribution analysis (Histograms) can be added here.")
    else:
        st.warning("Please upload Quality and Lithology Data.")




with tab_3d:
    if st.session_state['df_litho'] is None:
        st.warning("Please upload Lithology data.")
        st.stop()
        
    st.subheader("3D Geological Model")

    with st.expander("Model Configuration", expanded=True):
        
        c1, c2 = st.columns(2)
        bhid_list = st.session_state['df_bh']['BHID'].unique().tolist()
        
        # --- MAIN COLUMN 1: BOREHOLE & STRUCTURE ---
        with c1:
            st.markdown("##### 1. Borehole & Structure Selection")
            
            sub_c1, sub_c2 = st.columns(2)
            
            # -- Sub-Col 1: Boreholes --
            with sub_c1:
                select_all_bh = st.checkbox("Select All Boreholes", value=True, key='3d_select_all_bh')
                if select_all_bh:
                    selected_bhids_3d = bhid_list
                    st.info(f"All {len(bhid_list)} boreholes selected.")
                else:
                    selected_bhids_3d = st.multiselect("Choose Boreholes:", bhid_list, default=bhid_list[:1], key='3d_bhid_select')

            # -- Sub-Col 2: Coal Seam Cylinders --
            with sub_c2:
                select_all_seams = st.checkbox("Select All Coal Seams", value=True, key='3d_all_seams')
                if select_all_seams:
                    selected_seams_3d = COAL_SEAM_LCODES
                    st.info(f"All {len(COAL_SEAM_LCODES)} seams selected.")
                else:
                    selected_seams_3d = st.multiselect("Choose Seams:", COAL_SEAM_LCODES, default=COAL_SEAM_LCODES[:1], key='3d_seam_sel')
            
            st.markdown("---")
            st.markdown("**Visualization Settings**")
            
            z_exaggeration = st.number_input("Vertical Exaggeration:", 1.0, 100.0, 1.0, 0.5)
            radius_val = st.number_input("Borehole Radius (m):", 0.1, 50.0, 4.0, 0.5)
            
            c1a, c1b = st.columns(2)
            with c1a: include_waste_3d = st.checkbox("Show Parting", value=True)
            with c1b: show_hairlines_3d = st.checkbox("Show Hairlines", value=True)
            
        # --- MAIN COLUMN 2: SURFACE GENERATION ---
        with c2:
            st.markdown("##### 2. Surface Generation")
            
            c_glob1, c_glob2 = st.columns(2)
            with c_glob1:
                interp_limit = st.slider("Interpolation Limit (m)", 100, 2000, 500, 50)
            with c_glob2:
                global_opacity = st.slider("Global Opacity", 0.1, 1.0, 1.0)

            # --- NEW SLIDER FOR SMOOTHNESS ---
            resolution_val = st.slider("Grid Resolution (Smoothness)", 50, 300, 150, 25, help="Higher values = Smoother edges but slower generation.")

            st.write("") 
            show_topo = st.checkbox("Show Topography (TOPO)", value=False)
            
            c_surf_r, c_surf_f = st.columns(2)
            with c_surf_r:
                selected_roofs = st.multiselect("Show Roof For:", COAL_SEAM_LCODES, key='sel_roofs_3d')
            with c_surf_f:
                selected_floors = st.multiselect("Show Floor For:", COAL_SEAM_LCODES, key='sel_floors_3d')

        st.write("")
        st.markdown("---")
        
        generate_btn = st.button("🚀 Generate 3D Model", type="primary", use_container_width=True)

    # --- PROCESSING LOGIC ---
    surface_configs = {}
    target_surface_seams = []

    # 1. Handle Topo
    if show_topo:
        target_surface_seams.append("TOPO")
        surface_configs["TOPO"] = {'show_roof': True, 'show_floor': False, 'opacity': global_opacity, 'max_dist': interp_limit}

    # 2. Handle Coal Seams
    all_selected_seams = sorted(list(set(selected_roofs + selected_floors)), key=lambda x: COAL_SEAM_LCODES.index(x) if x in COAL_SEAM_LCODES else 999)
    for seam in all_selected_seams:
        target_surface_seams.append(seam)
        surface_configs[seam] = {'show_roof': seam in selected_roofs, 'show_floor': seam in selected_floors, 'opacity': global_opacity, 'max_dist': interp_limit}

    # TRIGGER
    if generate_btn:
        if selected_bhids_3d:
            with st.spinner("Generating 3D Scene..."):
                df_3d_data = prepare_3d_data(st.session_state['df_bh'], st.session_state['df_litho'], selected_bhids_3d, selected_seams_3d, include_waste_3d)
                df_bh_filtered = st.session_state['df_bh'][st.session_state['df_bh']['BHID'].isin(selected_bhids_3d)]
                
                fig_3d = plot_3d_model_combined(
                    df_3d_data, 
                    df_bh_filtered, 
                    z_exaggeration, 
                    radius_val, 
                    show_hairlines_3d,
                    st.session_state['df_boundary'], 
                    target_surface_seams, 
                    surface_configs,
                    resolution_val # <--- PASSING THE NEW SLIDER VALUE HERE
                )
                
                st.session_state['fig_3d_generated'] = fig_3d
        else: 
            st.warning("Select at least one borehole.")

    if 'fig_3d_generated' in st.session_state:
        st.plotly_chart(st.session_state['fig_3d_generated'], use_container_width=True)
        if target_surface_seams:
            st.caption(f"Visualizing surfaces: {', '.join(target_surface_seams)}")
