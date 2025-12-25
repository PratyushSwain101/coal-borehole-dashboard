import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.colors as pcolors
from scipy.interpolate import Rbf  # Radial Basis Function for Extrapolation
from matplotlib.path import Path   # For Polygon Clipping

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
    'THICKNESS': 'Total Coal Seam Thickness (m)',
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

st.set_page_config(layout="wide", page_title="BRP Coal Project")
st.title("BRP Coal Project_V3.2")

# Initialize Session State
if 'df_bh' not in st.session_state: st.session_state['df_bh'] = None
if 'df_boundary' not in st.session_state: st.session_state['df_boundary'] = None
if 'df_litho' not in st.session_state: st.session_state['df_litho'] = None
if 'df_quality' not in st.session_state: st.session_state['df_quality'] = None
    
if 'show_avg_all' not in st.session_state: st.session_state['show_avg_all'] = False 
if 'show_coal_only' not in st.session_state: st.session_state['show_coal_only'] = False
if 'corr_bhid_select' not in st.session_state: st.session_state['corr_bhid_select'] = []

# --- FILE PROCESSING FUNCTIONS ---

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
        fig.update_layout(xaxis={'categoryorder':'array'}, yaxis=dict(range=[0, y_max]), plot_bgcolor='white', paper_bgcolor='white', font=dict(color=PLOT_TEXT_COLOR), height=500, legend=dict(font=dict(size=10)))
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

    bh_label_y_offset = -50 
    fig.add_trace(go.Scatter(
        x=df_bh['X'], y=df_bh['Y'] + bh_label_y_offset, mode='text', text=df_bh['BHID'], textposition="bottom center",
        textfont=dict(size=8, color=PLOT_TEXT_COLOR), showlegend=False, legendgroup='boreholes', hoverinfo='skip'
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
                hovertext=df_litho_bh['LCODE'], 
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
    if df_quality is None or x_param not in df_quality.columns or y_param not in df_quality.columns: return go.Figure().add_annotation(text=f"Quality data not loaded or missing columns: {x_param} and/or {y_param}.", showarrow=False)
    df_plot = df_quality[(df_quality['LCODE'] == selected_seam)].copy()
    if selected_sample_type != 'All Samples': df_plot = df_plot[df_plot['SAMPLE_TYPE'] == selected_sample_type].copy()
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
    # --- LOGIC REMAINS FIXED: ONLY USES THE EXACT SELECTED SEAM FOR THICKNESS/QUALITY (FOR MAP VIEW) ---
    
    # Thickness calculation now only uses the exact selected seam LCODE.
    target_lcodes = [selected_seam] 
    
    df_thickness = df_litho[df_litho['LCODE'].isin(target_lcodes)].groupby('BHID')['WIDTH'].sum().reset_index()
    df_thickness.columns = ['BHID', 'THICKNESS']
    
    df_stats = pd.merge(df_bh[['BHID', 'X', 'Y', 'RL']].copy(), df_thickness, on='BHID', how='left')
    df_stats['THICKNESS'] = df_stats['THICKNESS'].fillna(0)
    
    if df_quality is not None:
        def calculate_wavg_for_seam(df, parameter):
            if df['INTERVAL'].sum() == 0 or (df[parameter] * df['INTERVAL']).isnull().all(): return np.nan
            return (df[parameter] * df['INTERVAL']).sum() / df['INTERVAL'].sum()
        
        quality_cols = [col for col in df_quality.columns if col in QUALITY_PARAMETERS and col != 'THICKNESS']
        
        # Quality data is also filtered only by the exact LCODE for weighted average.
        df_quality_seam = df_quality[df_quality['LCODE'] == selected_seam].copy()
        
        if selected_sample_type and selected_sample_type != 'All Samples':
            df_quality_seam = df_quality_seam[df_quality_seam['SAMPLE_TYPE'] == selected_sample_type].copy()
            
        # --- FIX: Ensure 'BHID' is the column name after reset_index() ---
        # 1. Group and apply to get weighted averages (BHID is the index)
        wavg_results = df_quality_seam.groupby('BHID').apply(lambda x: pd.Series({col: calculate_wavg_for_seam(x, col) for col in quality_cols}))
        
        # 2. Explicitly name the index before resetting to ensure the column is named 'BHID'
        wavg_results.index.name = 'BHID'
        wavg_results = wavg_results.reset_index()
        # --- END FIX ---

        df_stats = pd.merge(df_stats, wavg_results, on='BHID', how='left')
        
    return df_stats


def calculate_quality_stats_data(df_quality, selected_param, selected_sample_type, bh_ids_to_analyze):
    """
    Calculates the **Arithmetic Average (Mean)** of a quality parameter for each coal seam, 
    scoped by selected boreholes or the entire block.
    """
    if df_quality is None or selected_param not in df_quality.columns:
        return pd.DataFrame()

    # 1. Filter quality data by selected boreholes and coal seams only
    df_filtered = df_quality[
        (df_quality['LCODE'].isin(COAL_SEAM_LCODES)) & 
        (df_quality['BHID'].isin(bh_ids_to_analyze))
    ].copy()

    # 2. Filter by sample type and drop NaNs in the selected parameter
    if selected_sample_type != 'All Samples':
        df_filtered = df_filtered[df_filtered['SAMPLE_TYPE'] == selected_sample_type].copy()
        
    df_filtered.dropna(subset=[selected_param], inplace=True)

    if df_filtered.empty:
        return pd.DataFrame()

    # 3. Final aggregation: Calculate the simple ARITHMETIC MEAN (Average) for each seam (LCODE)
    
    df_summary_calc = df_filtered.groupby('LCODE').agg(
        **{selected_param: (selected_param, 'mean')} # Simple Mean (Average)
    ).reset_index()
    
    return df_summary_calc

# --- FUNCTION FOR QUALITY PLAN VIEW (WITH ALL FIXES AND NEW TABLE) ---
def plot_quality_plan_view(df_bh, df_boundary, df_quality, df_litho):
    
    # 1. Selection Controls for Seam, Sample Type, Parameter, and Color Scale
    # Added col_secondary_param for the new feature
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
    
    # NEW: Secondary Parameter Control
    with col_secondary_param:
        available_secondary = ['None'] + available_params
        selected_secondary_key = st.selectbox("4. Label Parameter (Secondary):", available_secondary, index=0, key='map_secondary_param_select')
        secondary_display_name = QUALITY_PARAMETERS.get(selected_secondary_key, selected_secondary_key)

    with col_colorscale:
        sequential_colorscales = (['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo', 'Jet', 'Hot', 'Electric', 'Portland', 'Blackbody'])
        selected_colorscale = st.selectbox("5. Select Color Scale:", sequential_colorscales[:], index=9, key='map_colorscale_select')

    # Preprocess/Aggregate data 
    df_analyzed = preprocess_quality_data(df_litho, df_quality, df_bh, selected_seam, selected_sample_type)
    
    # Create a temporary dataframe with seam depths and merge
    seam_depths_df = df_litho[df_litho['LCODE'] == selected_seam].groupby('BHID').agg(
        SEAM_FROM=('FROM', 'min'),
        SEAM_TO=('TO', 'max')
    ).reset_index()
    df_analyzed = pd.merge(df_analyzed, seam_depths_df, on='BHID', how='left')
    df_analyzed = pd.merge(df_analyzed, df_bh[['BHID', 'DEPTH']], on='BHID', how='left')


    # --- NON-ZERO FILTERING ---
    df_plot_data = df_analyzed[
        (df_analyzed[selected_param_key].notna()) & 
        (df_analyzed[selected_param_key] > 0.001)
    ].copy()
    
    fig = go.Figure()

    if not df_plot_data.empty:
        # --- Range Input and Highlighting Controls ---
        param_min_data = df_plot_data[selected_param_key].min()
        param_max_data = df_plot_data[selected_param_key].max()
        
        # Round the actual min/max data points to 2 decimals for the input fields
        param_min_rounded = round(float(param_min_data), 2)
        param_max_rounded = round(float(param_max_data), 2)
        
        # Use the rounded display values in the caption
        st.caption(f"Enter the Min/Max value for {param_display_name} (Data Range: {param_min_rounded:.2f} to {param_max_rounded:.2f})")
        
        col_min, col_max, col_highlight_mode = st.columns([1, 1, 2])
        
        with col_min:
            min_val = st.number_input(
                "Min Value:", 
                min_value=param_min_rounded, 
                max_value=param_max_rounded, 
                value=param_min_rounded, 
                step=0.1, 
                format="%.2f",
                key='quality_range_min'
            )
        with col_max:
            max_val = st.number_input(
                "Max Value:", 
                min_value=float(min_val), # Prevents Max < Min
                max_value=param_max_rounded, 
                value=param_max_rounded, 
                step=0.1,
                format="%.2f",
                key='quality_range_max'
            )
        
        with col_highlight_mode:
            st.write(""); st.write(""); highlight_mode = st.radio("Highlight Boreholes:", ('None', 'In Range', 'Outside Range'), index=0, key='highlight_mode', horizontal=True)
        
        # The subsequent validation check can remain, but is less likely to trigger
        if min_val > max_val:
            st.error("Minimum value cannot be greater than Maximum value. Re-adjusting...")
            pass 
        
        # --- FIX FOR PRECISION ISSUE (INCLUSIVE RANGE LOGIC) ---
        
        # Define a small tolerance (epsilon) for float comparisons
        EPSILON = 0.00001
        
        actual_min = min(min_val, max_val)
        actual_max = max(min_val, max_val)
        
        if highlight_mode == 'In Range':
            # Use >= (Min - EPSILON) and <= (Max + EPSILON) to be truly inclusive of float boundaries
            df_plot_data['Filtered'] = (
                (df_plot_data[selected_param_key] >= actual_min - EPSILON) & 
                (df_plot_data[selected_param_key] <= actual_max + EPSILON)
            )
        elif highlight_mode == 'Outside Range':
            # Use strict < (Min - EPSILON) or > (Max + EPSILON)
            df_plot_data['Filtered'] = (
                (df_plot_data[selected_param_key] < actual_min - EPSILON) | 
                (df_plot_data[selected_param_key] > actual_max + EPSILON)
            )
        else:
            df_plot_data['Filtered'] = False
            
        # --- END FIX FOR PRECISION ISSUE ---
        
        st.markdown("---")
        
        # <<< HOVER TEMPLATE CODE >>>
        param_short_name = param_display_name.split('(')[0].strip() # e.g. "Total Coal Seam Thickness"
        
        # Logic to extract the unit
        unit_str = ""
        if selected_param_key == 'THICKNESS':
            unit_str = "m"
        elif '(' in param_display_name and ')' in param_display_name:
            # Extracts the unit from parenthesis, e.g., "(%)" or "(Kcal/kg)"
            unit_str = param_display_name[param_display_name.find('(') : param_display_name.find(')')+1]
        
        # Add a space if the unit exists
        if unit_str:
            unit_str = " " + unit_str

        hover_template = (
            '<b>BHID:</b> %{customdata[0]}<br>' +
            '<b>RL:</b> %{customdata[1]:.2f} (m) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>TD:</b> %{customdata[2]:.2f} (m)<br>' +
            f'<b>{param_short_name}:</b> %{{marker.color:.2f}}{unit_str} <b>({selected_seam})</b><br>' +
            f'<b>From Depth:</b> %{{customdata[3]:.2f}} (m)<br>' +
            f'<b>To Depth:</b> %{{customdata[4]:.2f}} (m)<br>' +
            '<b>X:</b> %{x:.2f}<br>' +
            '<b>Y:</b> %{y:.2f}<extra></extra>'
        )

        # 1. Add Boreholes with Quality Data (Colored)
        fig.add_trace(
            go.Scatter(
                x=df_plot_data['X'], y=df_plot_data['Y'], mode='markers',
                marker=dict(
                    size=10,
                    color=df_plot_data[selected_param_key],
                    colorscale=selected_colorscale, 
                    colorbar=dict(title=f'{param_display_name} in {selected_seam} ({selected_sample_type})', title_side='right'), 
                    showscale=True, cmin=param_min_data, cmax=param_max_data, line=dict(width=1, color=NON_COAL_BORDER) 
                ),
                name=f'{selected_param_key} Data ({len(df_plot_data)})',
                hovertemplate=hover_template,
                # Pass all necessary columns to customdata
                customdata=df_plot_data[['BHID', 'RL', 'DEPTH', 'SEAM_FROM', 'SEAM_TO']],
                showlegend=True, 
                legendgroup='data_points'
            )
        )

        # 2. Add BHID Labels
        bh_label_y_offset = -60 
        fig.add_trace(
            go.Scatter(x=df_plot_data['X'], y=df_plot_data['Y'] + bh_label_y_offset, mode='text', text=df_plot_data['BHID'], textposition="bottom center",
                textfont=dict(size=8, color=PLOT_TEXT_COLOR), showlegend=False, hoverinfo='skip', legendgroup='data_points'
            )
        )
        
        # 2b. NEW: Add Secondary Quality Parameter Labels
        if selected_secondary_key != 'None':
            
            # Helper function to format the text label
            def format_secondary_label(row):
                value = row[selected_secondary_key]
                if pd.isna(value):
                    return ''
                # Get the short name and unit for display
                short_name = secondary_display_name.split('(')[0].strip()
                unit_match = secondary_display_name[secondary_display_name.find('(') : secondary_display_name.find(')')+1]
                unit = unit_match if '(' in secondary_display_name else ''
                
                # Format: ShortName: Value (Unit)
                return f"{short_name}: {value:.2f} {unit}"
            
            df_plot_data['SECONDARY_LABEL'] = df_plot_data.apply(format_secondary_label, axis=1)

            # Define offset for the text to sit next to the marker
            label_x_offset = 40 
            label_y_offset = 60
            
            fig.add_trace(
                go.Scatter(
                    x=df_plot_data['X'] + label_x_offset, 
                    y=df_plot_data['Y'] + label_y_offset, 
                    mode='text', 
                    text=df_plot_data['SECONDARY_LABEL'], 
                    # textposition="middle left",
                    textfont=dict(size=9, color='darkgreen'), 
                    name=f'{selected_secondary_key} Labels',
                    showlegend=False, 
                    hoverinfo='skip', 
                    legendgroup='data_points'
                )
            )
        
        # 3. Highlight Filtered Boreholes (Colored with Highlighting Info)
        if highlight_mode != 'None':
            # Filter the data for highlighted boreholes
            df_highlight = df_plot_data[df_plot_data['Filtered']].copy()

            if not df_highlight.empty:
                
                # Add an annotation to the legend to signify the filter mode
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],  # Invisible trace just for the legend entry
                        marker=dict(
                            size=10,
                            color='red',
                            symbol='circle',
                            line=dict(width=3, color='red')
                        ),
                        name=f'Highlighted ({len(df_highlight)}) - {highlight_mode}',
                        showlegend=True,
                        legendgroup='highlight'
                    )
                )

                # Scatter trace for the highlighted points, inheriting color/hover logic
                fig.add_trace(
                    go.Scatter(
                        x=df_highlight['X'],
                        y=df_highlight['Y'],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=df_highlight[selected_param_key],
                            colorscale=selected_colorscale,
                            colorbar=dict(
                                title=f'{param_display_name} in {selected_seam} ({selected_sample_type})',
                                title_side='right'
                            ),
                            showscale=False, cmin=param_min_data, cmax=param_max_data,
                            line=dict(width=3, color='red')  # Red border for highlight
                        ),
                        name=f'Highlighted Points - {highlight_mode}',
                        hovertemplate=hover_template,
                        customdata=df_highlight[['BHID', 'RL', 'DEPTH', 'SEAM_FROM', 'SEAM_TO']],
                        showlegend=False,
                        legendgroup='highlight'
                    )
                )

                # Add BHID labels for highlighted points (in red)
                fig.add_trace(
                    go.Scatter(
                        x=df_highlight['X'],
                        y=df_highlight['Y'] + bh_label_y_offset,
                        mode='text',
                        text=df_highlight['BHID'],
                        textposition="bottom center",
                        textfont=dict(size=8, color='red'),
                        showlegend=False,
                        hoverinfo='skip',
                        legendgroup='highlight'
                    )
                )

            else:
                st.info(f"No boreholes found **{highlight_mode.lower()}** the range of {min_val:.2f} to {max_val:.2f}.")

        
        # 4. Add Block Boundary
        fig.add_trace(go.Scatter(
            x=df_boundary['X'], y=df_boundary['Y'], mode='lines', line=dict(color='red', width=1, dash='dash'),
            name='Block Boundary', hovertemplate='Boundary Point<extra></extra>', showlegend=True
        ))
        
        # 5. Final Layout 
        fig.update_layout(
            xaxis_title="Easting (X) - UTM", yaxis_title="Northing (Y) - UTM", dragmode='pan', yaxis=dict(scaleanchor="x", scaleratio=1),
            title_text=f"Plan View: {param_display_name} Distribution in Seam {selected_seam} ({selected_sample_type})",
            plot_bgcolor='white', 
            paper_bgcolor='white', 
            hovermode="closest", height=700,
            font=dict(color=PLOT_TEXT_COLOR),
            margin=dict(l=50, r=250, t=80, b=50), 
            legend=dict(font=dict(size=10), x=1.1, y=1, yanchor='top', xanchor='left', bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1)
        )

        st.plotly_chart(fig, use_container_width=True)


        # Statistical Summary Table (Existing Block)
        if not df_plot_data.empty:
            data_to_summarize = df_plot_data[selected_param_key].dropna()
            if not data_to_summarize.empty:
                summary_data = {
                    'Metric': ['Boreholes Plotted (n)', 'Minimum Value', 'Maximum Value', 'Average (Mean)', 'Median'],
                    'Value': [len(data_to_summarize), f"{data_to_summarize.min():.2f}", f"{data_to_summarize.max():.2f}", f"{data_to_summarize.mean():.2f}", f"{data_to_summarize.median():.2f}"]
                }
                df_summary = pd.DataFrame(summary_data).set_index('Metric')
                with st.container():
                    st.subheader(f"Statistical Summary : ({selected_param_key})")
                    st.dataframe(df_summary.style.set_properties(**{'text-align': 'left'}), use_container_width=True)
                    st.markdown("---")


        # --- START NEW FEATURE: HIGHLIGHTED BOREHOLES SUMMARY TABLE ---
        if highlight_mode != 'None' and not df_plot_data.empty:
            df_highlight = df_plot_data[df_plot_data['Filtered']].copy()
            
            if not df_highlight.empty:
                
                # 1. Create the summary dataframe
                df_summary_highlight = df_highlight[['BHID', selected_param_key]].copy()
                df_summary_highlight.columns = ['BHID', 'Value']
                
                # 2. Add static contextual information
                df_summary_highlight.insert(1, 'Seam', selected_seam)
                df_summary_highlight.insert(2, 'Sample Type', selected_sample_type)
                df_summary_highlight.insert(3, 'Parameter', QUALITY_PARAMETERS.get(selected_param_key, selected_param_key).split('(')[0].strip()) # Use short name
                
                # 3. Rename the Value column for clarity in the table
                param_unit_match = QUALITY_PARAMETERS.get(selected_param_key, selected_param_key)
                param_unit = param_unit_match[param_unit_match.find('(') : param_unit_match.find(')')+1] if '(' in param_unit_match else ''
                df_summary_highlight = df_summary_highlight.rename(columns={'Value': f'Value {param_unit}'})
                
                st.markdown("---")
                st.subheader(f"Highlighted Boreholes Summary: **{highlight_mode}** ({len(df_highlight)} BHs)")
                
                st.dataframe(
                    df_summary_highlight.style.format({f'Value {param_unit}': "{:.2f}"}).set_properties(**{'text-align': 'left'}), 
                    use_container_width=True
                )
            else:
                st.info(f"No boreholes found **{highlight_mode.lower()}** the range of {min_val:.2f} to {max_val:.2f}.")

        # --- END NEW FEATURE ---

# --- 3D MODELLING FUNCTIONS ---

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
            f"BHID: {row['BHID']}<br>"
            f"RL: {row['Z_FROM']:.2f} to {row['Z_TO']:.2f} m<br>"
            f"From Depth: {row['FROM']:.2f} m<br>"
            f"To Depth: {row['TO']:.2f} m<br>"
            f"Width: {row['WIDTH']:.2f} m<br>"
            f"LCODE: {lcode}<br>"
            f"Detailed Lithology: {row.get('DETAILED LITHOLOGY', '')}"
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

# --- SURFACE GENERATION WITH RBF ---

def generate_seam_surface_rbf(df_3d, seam_code, df_boundary, z_exaggeration, resolution, show_roof, show_floor, opacity):
    seam_data = df_3d[df_3d['LCODE'] == seam_code]
    if len(seam_data) < 3: return []

    points_x, points_y = seam_data['X'].values, seam_data['Y'].values
    points_z_roof, points_z_floor = seam_data['Z_FROM'].values, seam_data['Z_TO'].values
    
    # Get Fixed Color
    seam_color = get_litho_color(seam_code)

    min_x, max_x = df_boundary['X'].min(), df_boundary['X'].max()
    min_y, max_y = df_boundary['Y'].min(), df_boundary['Y'].max()
    grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, resolution), np.linspace(min_y, max_y, resolution))

    poly_path = Path(list(zip(df_boundary['X'], df_boundary['Y'])))
    mask = poly_path.contains_points(np.column_stack((grid_x.flatten(), grid_y.flatten()))).reshape(grid_x.shape)

    traces = []

    def get_rbf_surface(z_values):
        try:
            rbf = Rbf(points_x, points_y, z_values, function='linear')
            grid_z = rbf(grid_x, grid_y)
            grid_z[~mask] = np.nan
            return grid_z * z_exaggeration
        except: return None

    if show_roof:
        grid_z_roof = get_rbf_surface(points_z_roof)
        if grid_z_roof is not None:
            traces.append(go.Surface(
                z=grid_z_roof, x=grid_x, y=grid_y, 
                colorscale=[[0, seam_color], [1, seam_color]], # Use Fixed Color
                showscale=False, opacity=opacity, 
                name=f'{seam_code} Roof', hoverinfo='all', 
                showlegend=True, legendgroup=seam_code
            ))

    if show_floor:
        grid_z_floor = get_rbf_surface(points_z_floor)
        if grid_z_floor is not None:
            # Darken floor slightly for visual contrast, or keep same
            traces.append(go.Surface(
                z=grid_z_floor, x=grid_x, y=grid_y, 
                colorscale=[[0, seam_color], [1, seam_color]], 
                showscale=False, opacity=opacity, 
                name=f'{seam_code} Floor', hoverinfo='all', 
                showlegend=True, legendgroup=seam_code
            ))

    return traces

def plot_3d_model_combined(df_3d, df_bh_filtered, z_exaggeration, radius, show_lines, df_boundary, 
                          selected_surface_seams, global_surf_config):
    fig = go.Figure()
    all_lines_x, all_lines_y, all_lines_z = [], [], []
    
    # 1. Borehole Sticks
    unique_seams = df_3d[df_3d['LCODE'].isin(COAL_SEAM_LCODES)]['LCODE'].unique()
    sorted_seams = [s for s in COAL_SEAM_LCODES if s in unique_seams]
    
    for lcode in sorted_seams:
        seam_data = df_3d[df_3d['LCODE'] == lcode]
        color = get_litho_color(lcode)
        mesh_trace, lines = create_cylinder_mesh(seam_data, radius, z_exaggeration, color, lcode, lcode, draw_lines=show_lines)
        fig.add_trace(mesh_trace)
        if show_lines:
            all_lines_x.extend(lines[0]); all_lines_y.extend(lines[1]); all_lines_z.extend(lines[2])

    non_coal_data = df_3d[~df_3d['LCODE'].isin(COAL_SEAM_LCODES)]
    if not non_coal_data.empty:
        mesh_trace, lines = create_cylinder_mesh(non_coal_data, radius * 0.9, z_exaggeration, NON_COAL_COLOR, 'Non-Coal', 'Non-Coal', draw_lines=show_lines)
        fig.add_trace(mesh_trace)
        if show_lines:
            all_lines_x.extend(lines[0]); all_lines_y.extend(lines[1]); all_lines_z.extend(lines[2])

    if show_lines and all_lines_x:
        fig.add_trace(go.Scatter3d(x=all_lines_x, y=all_lines_y, z=all_lines_z, mode='lines', line=dict(color='black', width=2), name='Boundaries', showlegend=False, hoverinfo='skip'))

    fig.add_trace(go.Scatter3d(x=df_bh_filtered['X'], y=df_bh_filtered['Y'], z=df_bh_filtered['RL'] * z_exaggeration, mode='markers+text', marker=dict(size=3, color='black'), text=df_bh_filtered['BHID'], textposition="top center", textfont=dict(size=10, color='black'), name='Collars'))

    # 2. Surfaces (Global Config applied to all)
    if selected_surface_seams:
        for seam_code in selected_surface_seams:
            traces = generate_seam_surface_rbf(
                df_3d, seam_code, df_boundary, z_exaggeration, 50,
                global_surf_config['show_roof'], global_surf_config['show_floor'],
                global_surf_config['opacity']
            )
            for trace in traces:
                fig.add_trace(trace)

    fig.update_layout(title="3D Geological Model", scene=dict(xaxis_title='Easting', yaxis_title='Northing', zaxis_title=f'Elevation x {z_exaggeration}', aspectmode='data'), height=800, margin=dict(l=0, r=0, b=0, t=50))
    return fig

# --- TAB DEFINITION ---
def data_upload_tab():
    st.header("Data Management and Upload")
    st.markdown("Upload the required CSV files. **Borehole Location and Boundary are mandatory to proceed.**")
    col_bh, col_boundary, col_litho, col_quality = st.columns(4)
    with col_bh:
        uploaded_bh_file = st.file_uploader("1. Borehole Location Data", type="csv", key="bh_uploader_tab")
        if st.button("Process Borehole Data", key="process_bh_tab"):
            st.session_state['df_bh'] = process_bh_data(uploaded_bh_file); st.rerun()
    with col_boundary:
        uploaded_boundary_file = st.file_uploader("2. Block Boundary Data", type="csv", key="boundary_uploader_tab")
        if st.button("Process Boundary Data", key="process_boundary_tab"):
            st.session_state['df_boundary'] = process_boundary_data(uploaded_boundary_file); st.rerun()
    with col_litho:
        uploaded_litho_file = st.file_uploader("3. Lithology Data", type="csv", key="litho_uploader_tab_litho")
        if st.button("Process Lithology Data", key="process_litho_tab"):
            st.session_state['df_litho'] = process_litho_data(uploaded_litho_file); st.rerun()
    with col_quality:
        uploaded_quality_file = st.file_uploader("4. Quality Data", type="csv", key="quality_uploader_tab_quality")
        if st.button("Process Quality Data", key="process_quality_tab"):
            st.session_state['df_quality'] = process_quality_data(uploaded_quality_file); st.rerun()
    st.markdown("---")
    st.subheader("Current Data Status")
    col_status = st.columns(4)
    data_status = {'Location': st.session_state['df_bh'] is not None, 'Boundary': st.session_state['df_boundary'] is not None, 'Lithology': st.session_state['df_litho'] is not None, 'Quality': st.session_state['df_quality'] is not None}
    for i, (name, loaded) in enumerate(data_status.items()):
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
            st.subheader("Borehole Lithology Data Table")
            col_select, col_button = st.columns([1, 1])
            with col_select:
                selected_bhid_table = st.selectbox("Select Borehole:", selected_bhids, key='litho_table_bhid_select') if selected_bhids else None
            with col_button:
                st.write("")
                if st.button("Toggle Coal Only", key='toggle_coal_only_button'):
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
                if st.button("Toggle Avg", key='toggle_avg_q'): st.session_state['show_avg_all'] = not is_avg
            
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
                c1, c2 = st.columns(2)
                with c1: xp = st.selectbox("X:", list(QUALITY_PARAMETERS.keys()), index=0, key='xp')
                with c2: yp = st.selectbox("Y:", list(QUALITY_PARAMETERS.keys()), index=1, key='yp')
                sp = st.selectbox("Seam:", COAL_SEAM_LCODES, key='sp')
                stp = st.selectbox("Type:", ['All Samples'] + list(st.session_state['df_quality']['SAMPLE_TYPE'].unique()), key='stp')
                st.plotly_chart(plot_quality_crossplot(st.session_state['df_quality'], sp, stp, xp, yp), use_container_width=True)
            with t_dist:
                # Add distribution logic if needed, keeping simple for now based on previous requests
                pass
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
        
        with c1:
            st.markdown("##### 1. Borehole & Structure")
            select_all_bh = st.checkbox("Select All Boreholes", value=True, key='3d_select_all')
            selected_bhids_3d = bhid_list if select_all_bh else st.multiselect("Choose Boreholes:", bhid_list, default=bhid_list[:1], key='3d_bhid_select')
            
            z_exaggeration = st.number_input("Vertical Exaggeration:", 1.0, 100.0, 1.0, 0.5)
            radius_val = st.number_input("Borehole Radius (m):", 0.1, 50.0, 4.0, 0.5)
            
            c1a, c1b = st.columns(2)
            with c1a: include_waste_3d = st.checkbox("Show Parting Layers", value=True)
            with c1b: show_hairlines_3d = st.checkbox("Show Hairlines", value=True)
            
        with c2:
            st.markdown("##### 2. Seam & Surface Filter")
            select_all_seams = st.checkbox("Select All Coal Seams", value=True, key='3d_all_seams')
            selected_seams_3d = COAL_SEAM_LCODES if select_all_seams else st.multiselect("Choose Seams:", COAL_SEAM_LCODES, default=COAL_SEAM_LCODES[:1], key='3d_seam_sel')
            
            st.markdown("---")
            target_surface_seams = st.multiselect("Generate Surfaces For:", COAL_SEAM_LCODES, help="Select multiple seams to stack.")
            
            # GLOBAL SURFACE CONTROLS (De-cluttered UI)
            if target_surface_seams:
                st.caption("Surface Styling (Applied to All Selected)")
                c2_s1, c2_s2, c2_s3 = st.columns(3)
                with c2_s1: show_r_global = st.checkbox("Roof", value=True, key="gr")
                with c2_s2: show_f_global = st.checkbox("Floor", value=True, key="gf")
                with c2_s3: op_global = st.slider("Opacity", 0.1, 1.0, 0.5, key="gop")
                
                # Create config object to pass to function
                global_surf_config = {
                    'show_roof': show_r_global,
                    'show_floor': show_f_global,
                    'opacity': op_global
                }
            else:
                global_surf_config = None
            
        st.write("")
        generate_btn = st.button("Generate 3D Model", type="primary", use_container_width=True)

    if generate_btn:
        if selected_bhids_3d:
            with st.spinner("Generating 3D Scene (Calculating RBF Surfaces... this may take time)..."):
                df_3d_data = prepare_3d_data(st.session_state['df_bh'], st.session_state['df_litho'], selected_bhids_3d, selected_seams_3d, include_waste_3d)
                df_bh_filtered = st.session_state['df_bh'][st.session_state['df_bh']['BHID'].isin(selected_bhids_3d)]
                
                fig_3d = plot_3d_model_combined(
                    df_3d_data, df_bh_filtered, 
                    z_exaggeration, radius_val, show_hairlines_3d,
                    st.session_state['df_boundary'], target_surface_seams, global_surf_config
                )
                st.session_state['fig_3d_generated'] = fig_3d
        else: st.warning("Select at least one borehole.")

    if 'fig_3d_generated' in st.session_state:
        st.plotly_chart(st.session_state['fig_3d_generated'], use_container_width=True)
        if target_surface_seams:
            st.success(f"Generated Surfaces for: {', '.join(target_surface_seams)}")


