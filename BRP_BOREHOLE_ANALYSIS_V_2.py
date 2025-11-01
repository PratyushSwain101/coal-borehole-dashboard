import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.colors as pcolors

# --- CONFIGURATION ---

# List of LCODEs that represent Coal Seams (Dark Color) - MANDATORY SEQUENCE ORDER
COAL_SEAM_LCODES = [
    'PAR', 'LAJ4', 'L4B', 'LAJ3', 'L2T3', 'L2T2', 'L2T1', 'L2T1T', 'L2T1B',
    'L2B', 'LAJ1', 'LL1', 'R5', 'R5T', 'R5B', 'R4', 'R3T', 'R3B', 'R12',
    'IBT', 'IBB'
]

# Standard Quality Parameters (Mapping uploaded column headers to display names)
QUALITY_PARAMETERS = {
    'THICKNESS': 'Total Coal Seam Thickness (m)', # Special case: calculated from lithology
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
# Base colors for the visualization
PLOT_TEXT_COLOR = 'black' 
NON_COAL_COLOR = '#ADD8E6' # Lightblue color
NON_COAL_BORDER = 'black'
CORRELATION_COLORS = pcolors.qualitative.Bold

# 1. DEFINE UNIQUE COLORS FOR EACH COAL SEAM
unique_palette = pcolors.qualitative.Alphabet + pcolors.qualitative.T10
SEAM_COLOR_MAP = {
    seam: unique_palette[i % len(unique_palette)]
    for i, seam in enumerate(COAL_SEAM_LCODES)
}
# Fallback for plotting in the loop in case the map is very large
DEFAULT_SEAM_COLOR = 'gray'

# Simple Color Mapping for Lithology
def get_litho_color(lcode):
    """Returns a color based on the LCODE for visualization (unique colors for seams)."""
    if lcode in SEAM_COLOR_MAP:
        return SEAM_COLOR_MAP[lcode]
    else:
        return NON_COAL_COLOR

# --- STREAMLIT APP SETUP ---

st.set_page_config(layout="wide")
st.title("Burapahar Coal Project")

# Initialize Session State for file data 
if 'df_bh' not in st.session_state:
    st.session_state['df_bh'] = None
if 'df_boundary' not in st.session_state:
    st.session_state['df_boundary'] = None
if 'df_litho' not in st.session_state:
    st.session_state['df_litho'] = None
if 'df_quality' not in st.session_state:
    st.session_state['df_quality'] = None
    
if 'show_avg_all' not in st.session_state:
    st.session_state['show_avg_all'] = False 
if 'raw_data_filter' not in st.session_state:
    st.session_state['raw_data_filter'] = 'All Lithologies'
if 'corr_log_filter' not in st.session_state:
    st.session_state['corr_log_filter'] = 'All Lithologies'
if 'highlight_in_range' not in st.session_state:
    st.session_state['highlight_in_range'] = False
    
if 'selected_sample_type' not in st.session_state:
    st.session_state['selected_sample_type'] = None
if 'dist_scope' not in st.session_state:
    st.session_state['dist_scope'] = 'Single Borehole'
    
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
            
            # The 'Hover_Label' column is no longer needed
            
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

# --- CORE PLOTTING FUNCTIONS ---
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
        title_font=dict(color=PLOT_TEXT_COLOR), font=dict(color=PLOT_TEXT_COLOR), plot_bgcolor='white',
        paper_bgcolor='white', hovermode="closest", height=700,
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
        for bhid in selected_bhids:
            ref_seam_data = df_combined[(df_combined['BHID'] == bhid) & (df_combined['LCODE'] == reference_seam)]
            if not ref_seam_data.empty: bh_offsets[bhid] = -ref_seam_data['TO RL'].min()
            else: excluded_bhids.append(bhid)
        plottable_bhids = [bhid for bhid in selected_bhids if bhid not in excluded_bhids]
    else:
        bh_offsets = {bhid: 0 for bhid in selected_bhids}
        plottable_bhids = selected_bhids

    if not plottable_bhids:
        fig = go.Figure().add_annotation(text=f"None of the selected boreholes contain the reference seam '{reference_seam}'." if is_flattened_mode else "No boreholes selected.", showarrow=False)
        return fig, excluded_bhids

    df_selected_bh = df_bh[df_bh['BHID'].isin(plottable_bhids)].set_index('BHID').loc[plottable_bhids].reset_index()
    bh_x_positions = [0.0]
    for i in range(1, len(df_selected_bh)):
        x1, y1 = df_selected_bh.loc[i-1, ['X', 'Y']]; x2, y2 = df_selected_bh.loc[i, ['X', 'Y']]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        bh_x_positions.append(bh_x_positions[-1] + distance * scale_multiplier)
    df_selected_bh['CUM_DISTANCE'] = bh_x_positions

    df_plot_data = df_combined[df_combined['LCODE'].isin(COAL_SEAM_LCODES)].copy() if filter_mode == 'Coal Seams Only' else df_combined.copy()

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
    for i, row in df_selected_bh.iterrows():
        bhid, rl, final_depth, x_pos, y_offset = row['BHID'], row['RL'], row['DEPTH'], row['CUM_DISTANCE'], bh_offsets.get(row['BHID'], 0)
        df_litho_bh = df_plot_data[df_plot_data['BHID'] == bhid]
        fig.add_trace(go.Bar(x=[x_pos], y=[final_depth], base=[rl - final_depth + y_offset], marker=dict(color='rgba(0,0,0,0)', line=dict(color='black', width=1.0)), orientation='v', width=BAR_WIDTH_VISUAL, hoverinfo='skip', showlegend=False))
        if not df_litho_bh.empty:
            df_litho_bh['COLOR'] = df_litho_bh['LCODE'].apply(get_litho_color)
            hover_text_series = ('BHID: ' + bhid + '<br>' + 'RL: ' + df_litho_bh['FROM RL'].round(2).astype(str) + ' to ' + df_litho_bh['TO RL'].round(2).astype(str) + ' m<br>' + 'From Depth: ' + df_litho_bh['FROM'].round(2).astype(str) + ' m<br>' + 'To Depth: ' + df_litho_bh['TO'].round(2).astype(str) + ' m<br>' + 'Width: ' + df_litho_bh['WIDTH'].round(2).astype(str) + ' m<br>' + 'LCODE: ' + df_litho_bh['LCODE'] + '<br>' + 'Detailed Lithology: ' + df_litho_bh['DETAILED LITHOLOGY'])
            fig.add_trace(go.Bar(x=[x_pos] * len(df_litho_bh), y=df_litho_bh['RL_WIDTH'], base=df_litho_bh['TO RL'] + y_offset, marker=dict(color=df_litho_bh['COLOR'], line=dict(color='black', width=1.0)), text=df_litho_bh['LCODE'], textposition='inside', textfont=dict(color=PLOT_TEXT_COLOR, size=9), orientation='v', width=BAR_WIDTH_VISUAL, hoverinfo='text', hovertext=hover_text_series, showlegend=False))
        fig.add_annotation(x=x_pos, y=max_header_y + HEADER_HEIGHT_OFFSET, text=f"<b>{bhid}</b><br>RL: {rl:.1f}<br>TD: {final_depth:.1f} m", showarrow=False, font=dict(color=PLOT_TEXT_COLOR, size=10), xanchor='center', yanchor='bottom')
        seam_boundaries_rl = df_combined[df_combined['LCODE'].isin(COAL_SEAM_LCODES) & (df_combined['BHID'] == bhid)][['FROM RL', 'TO RL']].stack().unique().tolist()
        seam_boundaries_rl.append(rl - final_depth)
        unique_rls_to_label = sorted(list(set([r for r in seam_boundaries_rl if r <= rl + 0.1])), reverse=True)
        for rl_tick in unique_rls_to_label:
            fig.add_annotation(x=x_pos + BAR_WIDTH_VISUAL / 2 + 5, y=rl_tick + y_offset, text=f"{(rl - rl_tick):.2f} m", showarrow=False, font=dict(color=PLOT_TEXT_COLOR, size=8), xanchor='left', yanchor='middle')

    if selected_seams and selected_seams != ['None']:
        for idx, s_seam in enumerate(selected_seams):
            if s_seam == 'None': continue
            line_color = CORRELATION_COLORS[idx % len(CORRELATION_COLORS)]
            x_coords, y_top, y_bottom = [], [], []
            for _, bh_row in df_selected_bh.iterrows():
                bhid, dist, offset = bh_row['BHID'], bh_row['CUM_DISTANCE'], bh_offsets.get(bh_row['BHID'], 0)
                seam_data = df_combined[(df_combined['BHID'] == bhid) & (df_combined['LCODE'] == s_seam)]
                if not seam_data.empty: x_coords.append(dist); y_top.append(seam_data['FROM RL'].max() + offset); y_bottom.append(seam_data['TO RL'].min() + offset)
            if len(x_coords) > 1:
                fig.add_trace(go.Scatter(x=x_coords, y=y_top, mode='lines+markers', line=dict(color=line_color, width=1), name=f'{s_seam} Top', showlegend=True))
                fig.add_trace(go.Scatter(x=x_coords, y=y_bottom, mode='lines+markers', line=dict(color=line_color, width=1, dash='dot'), name=f'{s_seam} Bottom', showlegend=True))
    if not is_flattened_mode and len(collar_y_coords) > 1:
        x_c, y_c = zip(*collar_y_coords); fig.add_trace(go.Scatter(x=list(x_c), y=list(y_c), mode='lines', line=dict(color='blue', width=2, dash='dash'), name='Surface Profile', showlegend=True))

    if is_flattened_mode:
        title, yaxis_title = f"Seam Correlation (Datum: Floor of '{reference_seam}')", "Relative Elevation from Datum (m)"
        yaxis_config = dict(title=yaxis_title, showticklabels=True, showgrid=True, zeroline=True, zerolinecolor='red', zerolinewidth=2, range=[min_y_range, max_y_range])
    else:
        title, yaxis_title = "Geological Correlation (True Elevation)", "Elevation (RL) above MSL (m)"
        yaxis_config = dict(title=yaxis_title, showgrid=True, zeroline=True, zerolinecolor=PLOT_TEXT_COLOR, range=[min_y_range, max_y_range])

    # Filter legend items based on what is actually plotted
    df_legend_data = df_plot_data[df_plot_data['BHID'].isin(plottable_bhids)].copy()
    if not df_legend_data.empty:
        lcode_shallowest = df_legend_data.groupby('LCODE')['RL'].max().sort_values(ascending=False).index.tolist()
        for lcode in lcode_shallowest:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=get_litho_color(lcode), line=dict(width=1, color='black')), name=lcode, showlegend=True))
    
    fig.update_layout(
        title_text=title, title_font=dict(size=16),
        xaxis=dict(title="Cumulative Distance along Section (m)", tickvals=df_selected_bh['CUM_DISTANCE'], ticktext=[f'{d:.0f} m' for d in df_selected_bh['CUM_DISTANCE']], showgrid=False, zeroline=False),
        yaxis=yaxis_config, height=700, barmode='stack', plot_bgcolor='white', paper_bgcolor='white', font=dict(color=PLOT_TEXT_COLOR),
        legend=dict(font=dict(size=10), x=1.02, y=1, bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1),
        margin=dict(l=50, r=100, t=100, b=50)
    )
    return fig, excluded_bhids

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
    df_thickness = df_litho[df_litho['LCODE'] == selected_seam].groupby('BHID')['WIDTH'].sum().reset_index()
    df_thickness.columns = ['BHID', 'THICKNESS']
    df_stats = pd.merge(df_bh[['BHID', 'X', 'Y', 'RL']].copy(), df_thickness, on='BHID', how='left')
    df_stats['THICKNESS'] = df_stats['THICKNESS'].fillna(0)
    if df_quality is not None:
        def calculate_wavg_for_seam(df, parameter):
            if df['INTERVAL'].sum() == 0 or (df[parameter] * df['INTERVAL']).isnull().all(): return np.nan
            return (df[parameter] * df['INTERVAL']).sum() / df['INTERVAL'].sum()
        quality_cols = [col for col in df_quality.columns if col in QUALITY_PARAMETERS and col != 'THICKNESS']
        df_quality_seam = df_quality[df_quality['LCODE'] == selected_seam].copy()
        if selected_sample_type and selected_sample_type != 'All Samples':
            df_quality_seam = df_quality_seam[df_quality_seam['SAMPLE_TYPE'] == selected_sample_type].copy()
        wavg_results = df_quality_seam.groupby('BHID').apply(lambda x: pd.Series({col: calculate_wavg_for_seam(x, col) for col in quality_cols})).reset_index()
        df_stats = pd.merge(df_stats, wavg_results, on='BHID', how='left')
    return df_stats


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
        sequential_colorscales = sorted(['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo', 'Jet', 'Hot', 'Electric', 'Portland', 'Blackbody'])
        selected_colorscale = st.selectbox("5. Select Color Scale:", sequential_colorscales, index=0, key='map_colorscale_select')

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
            label_x_offset = 100 
            
            fig.add_trace(
                go.Scatter(
                    x=df_plot_data['X'] + label_x_offset, 
                    y=df_plot_data['Y'], 
                    mode='text', 
                    text=df_plot_data['SECONDARY_LABEL'], 
                    textposition="middle left",
                    textfont=dict(size=9, color='darkgreen'), 
                    name=f'{selected_secondary_key} Labels',
                    showlegend=False, 
                    hoverinfo='skip', 
                    legendgroup='data_points'
                )
            )
        
        # 3. Highlight Filtered Boreholes (Red Ring)
        if highlight_mode != 'None':
            df_highlight = df_plot_data[df_plot_data['Filtered']]
            if not df_highlight.empty:
                fig.add_trace(go.Scatter(x=df_highlight['X'], y=df_highlight['Y'], mode='markers', marker=dict(size=10, color='rgba(255, 0, 0, 0)', symbol='circle', line=dict(width=3, color='red')), name=f'Highlighted ({len(df_highlight)})', hoverinfo='skip', showlegend=True, legendgroup='highlight'))
                fig.add_trace(go.Scatter(x=df_highlight['X'], y=df_highlight['Y'] - 60, mode='text', text=df_highlight['BHID'], textposition="bottom center", textfont=dict(size=8, color='red'), showlegend=False, hoverinfo='skip', legendgroup='highlight'))
    else: 
        st.info(f"No non-zero data for {param_display_name} in seam {selected_seam} for sample type {selected_sample_type}.")
    
    # 4. Add Block Boundary
    fig.add_trace(go.Scatter(
        x=df_boundary['X'], y=df_boundary['Y'], mode='lines', line=dict(color='red', width=1, dash='dash'),
        name='Block Boundary', hovertemplate='Boundary Point<extra></extra>', showlegend=True
    ))
    
    # 5. Final Layout 
    fig.update_layout(
        xaxis_title="Easting (X) - UTM", yaxis_title="Northing (Y) - UTM", dragmode='pan', yaxis=dict(scaleanchor="x", scaleratio=1),
        title_text=f"Plan View: {param_display_name} Distribution in Seam {selected_seam} ({selected_sample_type})",
        plot_bgcolor='white', paper_bgcolor='white', hovermode="closest", height=700,
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
            # df_summary_highlight.insert(5, 'Min Range', f"{min_val:.2f}")
            # df_summary_highlight.insert(6, 'Max Range', f"{max_val:.2f}")
            
            # 3. Rename the Value column for clarity in the table
            param_unit_match = QUALITY_PARAMETERS.get(selected_param_key, selected_param_key)
            param_unit = param_unit_match[param_unit_match.find('(') : param_unit_match.find(')')+1] if '(' in param_unit_match else ''
            df_summary_highlight = df_summary_highlight.rename(columns={'Value': f'Value {param_unit}'})
            
            st.markdown("---")
            st.subheader(f"Highlighted Boreholes Summary: **{highlight_mode}** ({len(df_highlight)} BHs)")
            
            # Display the table
            st.dataframe(
                df_summary_highlight.style.format({f'Value {param_unit}': "{:.2f}"}).set_properties(**{'text-align': 'left'}), 
                use_container_width=True
            )
        else:
            st.info(f"No boreholes found **{highlight_mode.lower()}** the range of {min_val:.2f} to {max_val:.2f}.")

    # --- END NEW FEATURE ---
    

# --- TAB 0: Data Management (Definition) ---
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

# --- FINAL CODE EXECUTION FLOW ---

litho_loaded = st.session_state['df_litho'] is not None
quality_loaded = st.session_state['df_quality'] is not None

tab_data, tab_block_overview, tab_litho_log, tab_quality = st.tabs([
    "1. Data Management", "2. Block Overview", "3. Borehole Correlation", "4. Quality Analysis"
])

if st.session_state['df_bh'] is None or st.session_state['df_boundary'] is None:
    with tab_data:
        st.title("Data Loading Required")
        data_upload_tab()
    st.stop()
    
with tab_data: data_upload_tab()

with tab_block_overview:
    st.header("Overview: Borehole Locations and ML Boundary")
    fig_plan_view = plot_plan_view(st.session_state['df_bh'], st.session_state['df_boundary'])
    st.plotly_chart(fig_plan_view, use_container_width=True, key="block_overview_map")
    st.write("---")
    st.header("Data Previews")
    tab1_data, tab2_data = st.tabs(["Borehole Location Data", "Block Boundary Data"])
    with tab1_data:
        df_display = st.session_state['df_bh'].drop(columns=['Hover_Label'], errors='ignore')
        st.dataframe(df_display.style.set_properties(**{'text-align': 'left'}), use_container_width=True)
    with tab2_data:
        st.dataframe(st.session_state['df_boundary'].style.set_properties(**{'text-align': 'left'}), use_container_width=True)

with tab_litho_log:
    if not litho_loaded: st.warning("Please upload and process Borehole Lithology data to use the correlation tool."); st.stop()
    df_bh, df_litho = st.session_state['df_bh'], st.session_state['df_litho']
    bhid_list = df_bh['BHID'].unique().tolist()
    seam_list = COAL_SEAM_LCODES
    seam_list_with_none = ['None'] + seam_list
    st.header("Borehole Correlation")
    selected_bhids = st.session_state.get('corr_bhid_select', [])
    fig_map = plot_plan_view(st.session_state['df_bh'], st.session_state['df_boundary'], selected_bhids)
    st.plotly_chart(fig_map, use_container_width=True, key="correlation_map")
    st.markdown("---")
    col1, col2, col3, col4 = st.columns([2.5, 1.5, 1.5, 1])
    with col1: selected_bhids = st.multiselect("1. Select Boreholes:", bhid_list, default=bhid_list[:1] if len(bhid_list) > 1 else bhid_list, key='corr_bhid_select')
    with col2: reference_seam = st.selectbox("2. Select Seam for Correlation:", seam_list_with_none, key='corr_reference_seam', help="Select 'None' for true elevation view. Select a seam to flatten the plot on that seam's floor.")
    with col3: selected_seams_lines = st.multiselect("3. Plot Correlation Lines for:", seam_list, key='corr_lines_select')
    with col4: st.write(""); st.write(""); filter_mode = st.radio("4. Lithology Filter:", ('All Lithology', 'Coal Seams Only'), key='corr_litho_filter')
    if not selected_bhids: st.info("Please select at least one borehole to generate a correlation plot.")
    else:
        st.markdown("---")
        fig_corr, excluded = plot_litho_correlation(df_bh, df_litho, selected_bhids, selected_seams_lines, filter_mode, reference_seam)
        if excluded: st.warning(f"**Note:** Borehole(s) `{', '.join(excluded)}` were excluded from the plot as they do not contain the reference seam '{reference_seam}'.")
        st.plotly_chart(fig_corr, use_container_width=True, key="main_correlation_plot")

with tab_quality:
    if not litho_loaded: st.warning("Please upload Lithology data for thickness calculations and quality analysis."); st.stop()
    df_bh, df_litho, df_quality = st.session_state['df_bh'], st.session_state['df_litho'], st.session_state['df_quality']
    bhid_list = df_bh['BHID'].unique().tolist()
    quality_param_keys = [k for k in QUALITY_PARAMETERS.keys() if k != 'THICKNESS']
    tab_map, tab_stats, tab_analytics = st.tabs(["Quality Map", "Thickness Stats", "Quality Analytics"])

    with tab_map:
        st.subheader("Quality Data Distribution")
        plot_quality_plan_view(df_bh, st.session_state['df_boundary'], df_quality, df_litho)

    with tab_stats:
        col_selector, col_button = st.columns([3, 1])
        def toggle_average_view_new(): st.session_state['show_avg_all'] = not st.session_state.get('show_avg_all', False)
        with col_selector:
            selected_stats_bhids = st.multiselect("Select Borehole(s) for Group Analysis (Optional):", bhid_list, default=bhid_list[:1] if not st.session_state.get('show_avg_all', False) and bhid_list else None, key='stats_bhids_new', disabled=st.session_state.get('show_avg_all', False))
        with col_button:
            st.write(""); st.write("")
            button_label = "Show Block-Wide Average" if not st.session_state.get('show_avg_all', False) else "Show Selected Borehole(s)"
            st.button(button_label, key='toggle_avg_all_new', on_click=toggle_average_view_new)
        st.write("---")
        
        if st.session_state.get('show_avg_all', False) or selected_stats_bhids:
            df_source_litho = df_litho
            bh_ids_to_analyze = bhid_list if st.session_state.get('show_avg_all', False) else selected_stats_bhids
            if not st.session_state.get('show_avg_all', False):
                df_source_litho = df_source_litho[df_source_litho['BHID'].isin(selected_stats_bhids)].copy()

            n_bh = len(bh_ids_to_analyze)
            df_total_per_bh_seam = df_source_litho[df_source_litho['LCODE'].isin(COAL_SEAM_LCODES)].groupby(['BHID', 'LCODE'], as_index=False)['WIDTH'].sum()
            df_summary_calc = df_total_per_bh_seam.groupby('LCODE').agg(AVERAGE_THICKNESS_M=('WIDTH', 'mean')).reset_index()
            y_title = "Average Thickness (m)"
            
            if not df_summary_calc.empty:
                fig_stats, df_table = plot_seam_stats(df_summary_calc, f"Average Seam Thickness (n={n_bh})", y_title, 'THICKNESS', 'Bar Chart', COAL_SEAM_LCODES)
                st.plotly_chart(fig_stats, use_container_width=True)
                
                summary_col, totals_col = st.columns([2, 1])
                with summary_col:
                    st.subheader(f"Summary: {y_title}")
                    st.dataframe(df_table.style.format({y_title: "{:.2f}"}).set_properties(**{'text-align': 'left'}), use_container_width=True)
                with totals_col:
                    st.write(" ")
                    df_coal_only = df_source_litho[df_source_litho['LCODE'].isin(COAL_SEAM_LCODES)]
                    total_coal_thickness = df_coal_only['WIDTH'].sum()
                    df_bh_filtered = df_bh[df_bh['BHID'].isin(bh_ids_to_analyze)]
                    total_td = df_bh_filtered['DEPTH'].sum() if not df_bh_filtered.empty else 0
                    total_non_coal_thickness = total_td - total_coal_thickness
                    st.metric(label="Total Coal Thickness (Cumulative)", value=f"{total_coal_thickness:,.2f} m")
                    st.metric(label="Total Non-Coal Thickness (Cumulative)", value=f"{total_non_coal_thickness:,.2f} m")
        else: st.info("Please select borehole(s) or switch to 'Block-Wide Average'.")
    
    with tab_analytics:
        if not quality_loaded: st.warning("Please upload Quality Data to enable analytics."); st.stop()
        tab_cross, tab_dist = st.tabs(["Cross-Plot", "Distribution"])
        with tab_cross:
            st.subheader("Bivariate Correlation Analysis (Cross-Plot)")
            sample_type_list = ['All Samples'] + df_quality['SAMPLE_TYPE'].unique().tolist()
            col_seam_a, col_sample_a = st.columns(2)
            with col_seam_a: selected_seam_a = st.selectbox("1. Select Coal Seam:", COAL_SEAM_LCODES, key='cross_plot_seam')
            with col_sample_a: selected_sample_type_a = st.selectbox("2. Select Sample Type:", sample_type_list, key='cross_plot_sample')
            st.markdown("---")
            col_x, col_y = st.columns(2)
            available_q_params = [k for k in quality_param_keys if k in df_quality.columns]
            if available_q_params:
                with col_x: selected_x = st.selectbox("3. Select X-Axis:", available_q_params, index=available_q_params.index('ASH_PERC') if 'ASH_PERC' in available_q_params else 0, key='cross_x')
                with col_y: selected_y = st.selectbox("4. Select Y-Axis:", available_q_params, index=available_q_params.index('GCV_KCAL') if 'GCV_KCAL' in available_q_params else 1, key='cross_y')
                st.write("---")
                fig_cross = plot_quality_crossplot(df_quality, selected_seam_a, selected_sample_type_a, selected_x, selected_y)
                st.plotly_chart(fig_cross, use_container_width=True)

        with tab_dist:
            st.subheader("Quality Distribution Seam-Wise")
            col_scope, col_bhid_d, col_param_d, col_sample_d = st.columns(4)
            with col_scope: selected_scope_d = st.radio("1. Scope:", ('Single Borehole', 'All Boreholes'), key='dist_scope')
            with col_bhid_d: selected_bhid_d = st.selectbox("2. Borehole:", bhid_list, key='dist_bhid', disabled=(selected_scope_d != 'Single Borehole'))
            with col_param_d: selected_param_d = st.selectbox("3. Parameter:", [k for k in quality_param_keys if k in df_quality.columns], key='dist_param')
            with col_sample_d: selected_sample_type_d = st.selectbox("4. Sample Type:", ['All Samples'] + df_quality['SAMPLE_TYPE'].unique().tolist(), key='dist_sample')
            st.write("---")
            df_plot_base = df_quality.copy()
            scope_label = "Block-Wide"
            if selected_scope_d == 'Single Borehole':
                if selected_bhid_d:
                    df_plot_base = df_plot_base[df_plot_base['BHID'] == selected_bhid_d]
                    scope_label = f"BH: {selected_bhid_d}"
                else: st.info("Please select a borehole for the 'Single Borehole' scope."); st.stop()
            if selected_sample_type_d != 'All Samples': df_plot_base = df_plot_base[df_plot_base['SAMPLE_TYPE'] == selected_sample_type_d]
            param_name = QUALITY_PARAMETERS.get(selected_param_d, selected_param_d)
            fig_dist, df_summary_table = plot_seam_stats(df_plot_base, f"{param_name} Distribution ({scope_label})", param_name, selected_param_d, 'Box Plot', COAL_SEAM_LCODES)
            st.plotly_chart(fig_dist, use_container_width=True)
            if not df_summary_table.empty:
                st.subheader(f"Statistical Summary for {selected_param_d}")
                st.dataframe(df_summary_table.style.set_properties(**{'text-align': 'left'}), use_container_width=True)

