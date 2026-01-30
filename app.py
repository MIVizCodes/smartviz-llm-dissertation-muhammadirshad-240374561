import streamlit as st
import pandas as pd
from prettytable import PrettyTable
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from huggingface_hub import InferenceClient
import matplotlib.pyplot as plt
import seaborn as sns
import time
import socket

# === OLLAMA LOCAL SUPPORT ===
try:
    import ollama
    ollama_available = True
except ImportError:
    ollama_available = False
    print("Ollama Python client not installed - run: pip install ollama")

# === PAGE CONFIG ===
st.set_page_config(
    page_title="PES Building Analytics Assistant",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Bismillah! PES Building Analytics Assistant 🌟")
st.markdown("**Full RAG-Powered IoT Analytics Chat** | Alhamdulillah")

# Show success message if recently filtered
if 'filter_message' in st.session_state:
    st.success(st.session_state.filter_message)
    del st.session_state.filter_message

def is_ollama_running(host="127.0.0.1", port=11434, timeout=1):
    """Quick non-blocking check if Ollama server is alive"""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False

def generate_llm_response(messages, temperature=0.65, max_tokens=350, stream_to_ui=False):
    """
    Unified LLM response generator with timing + local/cloud label
    """
    model_local = "llama3.2:3b"
    model_cloud = "google/gemma-2-9b-it"
    
    start_time = time.time()  # Start timing
    
    if use_local and ollama_available:
        try:
            stream = ollama.chat(
                model=model_local,
                messages=messages,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens
                },
                stream=True
            )
            full_response = ""
            placeholder = st.empty() if stream_to_ui else None
            for chunk in stream:
                delta = chunk['message']['content']
                full_response += delta
                if stream_to_ui:
                    placeholder.markdown(full_response + "▌")
            if stream_to_ui:
                placeholder.markdown(full_response)
            
            end_time = time.time()
            response_time = end_time - start_time
            st.caption(f"Response time: {response_time:.3f} seconds ({'Local' if use_local else 'Cloud'})")
            return full_response.strip()
        
        except Exception as ollama_err:
            st.warning(f"Local Ollama error: {str(ollama_err)} — falling back to Hugging Face cloud")
    
    # Fallback to Hugging Face
    try:
        resp = ""
        placeholder = st.empty() if stream_to_ui else None
        for chunk in client.chat_completion(
            model=model_cloud,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        ):
            delta = chunk.choices[0].delta.content or ""
            resp += delta
            if stream_to_ui:
                placeholder.markdown(resp + "▌")
        if stream_to_ui:
            placeholder.markdown(resp)
        
        end_time = time.time()
        response_time = end_time - start_time
        st.caption(f"Response time: {response_time:.1f} seconds (Cloud)")
        return resp.strip()
    
    except Exception as hf_err:
        st.error(f"Both local and cloud LLM failed: {str(hf_err)}")
        return "Error generating response. Please check Ollama status or Hugging Face token."

# === HUGGING FACE TOKEN ===
HF_TOKEN = st.text_input("Enter your Hugging Face API Token (hf_...)", type="password")
if not HF_TOKEN:
    st.warning("Get free token from https://huggingface.co/settings/tokens")
    st.stop()

client = InferenceClient(token=HF_TOKEN)

# === IMPROVED HELPER FUNCTIONS ===

def get_latest_room_ranking(metric, top_n=1, ascending=False):
    """Get top rooms at the very latest timestamp"""
    try:
        current = st.session_state.current_df
    except (NameError, AttributeError, KeyError):
        current = df
    
    data = current[current['metric_name'] == metric]
    if data.empty:
        return pd.Series(dtype=float), "No data"
    
    latest_time = data['timestamp'].max()
    latest_data = data[data['timestamp'] == latest_time]
    
    if metric == 'Occupancy':
        agg_data = latest_data[latest_data['aggregation'] == 'max']
    else:
        agg_data = latest_data[latest_data['aggregation'] == 'mean']
    
    if agg_data.empty:
        return pd.Series(dtype=float), "No suitable aggregation data"
    
    ranking = agg_data.groupby('display_name')['value'].mean().sort_values(ascending=ascending)
    return ranking.head(top_n), f"Latest: {latest_time.strftime('%Y-%m-%d %H:%M UTC')}"

def rooms_above_average(metric, threshold, top_n=12, ascending=False):
    try:
        current = st.session_state.current_df
    except (NameError, AttributeError, KeyError):
        current = df
    
    data = current[current['metric_name'] == metric]
    if data.empty:
        return pd.Series(dtype=float)
    
    avg = data.groupby('display_name')['value'].mean()
    
    if ascending:  # lowest first
        filtered = avg[avg < threshold].sort_values(ascending=True)
    else:
        filtered = avg[avg >= threshold].sort_values(ascending=False)
    
    return filtered.head(top_n)

def find_long_high_periods(metric='co2', threshold=1000, min_hours=3, top_n=12):
    try:
        current = st.session_state.current_df
    except (NameError, AttributeError, KeyError):
        current = df
    
    data = current[(current['metric_name'] == metric) & (current['value'] >= threshold)].copy()
    
    if data.empty:
        return pd.DataFrame()
    
    data = data.sort_values(['display_name', 'timestamp'])
    data['time_diff_h'] = data.groupby('display_name')['timestamp'].diff().dt.total_seconds() / 3600
    data['new_group'] = (data['time_diff_h'] > 1.5) | data['time_diff_h'].isna()
    data['period_id'] = data.groupby('display_name')['new_group'].cumsum()
    
    periods = data.groupby(['display_name', 'period_id']).agg(
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        count=('value', 'count')
    ).reset_index()
    
    periods['duration_hours'] = (periods['end_time'] - periods['start_time']).dt.total_seconds() / 3600
    
    long_periods = periods[periods['duration_hours'] >= min_hours].copy()
    long_periods = long_periods.sort_values('duration_hours', ascending=False).head(top_n)
    long_periods = long_periods.rename(columns={'display_name': 'room'})
    
    return long_periods[['room', 'start_time', 'end_time', 'duration_hours']]

def find_long_high_temp_periods(threshold=24, min_hours=3, top_n=12):
    try:
        current = st.session_state.current_df
    except (NameError, AttributeError, KeyError):
        current = df
    
    data = current[
        (current['metric_name'] == 'temp') &
        (current['aggregation'] == 'mean') &
        (current['value'] >= threshold)
    ].copy()
    
    if data.empty:
        return pd.DataFrame()
    
    data = data.sort_values(['display_name', 'timestamp'])
    data['time_diff_h'] = data.groupby('display_name')['timestamp'].diff().dt.total_seconds() / 3600
    data['new_group'] = (data['time_diff_h'] > 1.5) | data['time_diff_h'].isna()
    data['period_id'] = data.groupby('display_name')['new_group'].cumsum()
    
    periods = data.groupby(['display_name', 'period_id']).agg(
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        count=('value', 'count'),
        max_temp=('value', 'max'),
        avg_temp=('value', 'mean')
    ).reset_index()
    
    periods['duration_hours'] = (periods['end_time'] - periods['start_time']).dt.total_seconds() / 3600
    
    long_periods = periods[periods['duration_hours'] >= min_hours].copy()
    long_periods = long_periods.sort_values('duration_hours', ascending=False).head(top_n)
    long_periods = long_periods.rename(columns={'display_name': 'room'})
    
    return long_periods[['room', 'start_time', 'end_time', 'duration_hours', 'max_temp', 'avg_temp']]

# === LOAD DATA ===
@st.cache_data
def load_and_prepare_data():
    hr_df = pd.read_csv("C:/Users/Hp/Desktop/data work/PES_Building_Analytics_App/data/metrics_app_hierarchy_202506111454_backup.csv")
    timeag_df = pd.read_csv("C:/Users/Hp/Desktop/data work/PES_Building_Analytics_App/data/metrics_app_timeaggregated_202506111450_backup.csv")
    timeag_df3 = pd.read_csv("C:/Users/Hp/Desktop/data work/PES_Building_Analytics_App/data/metrics_app_timeaggregated_202507091430_backup-1752067848951_cleaned.csv")
    timeag_df3['project_name'] = 'sbs'
    timeag_combined = pd.concat([timeag_df, timeag_df3], ignore_index=True)
    timeag_df = timeag_combined.copy()
    timeag_df['start_time'] = pd.to_datetime(timeag_df['start_time'], utc=True)
    timeag_df['timestamp'] = timeag_df['start_time']
    
    merged_df = timeag_df.merge(hr_df[['geometry_id', 'display_name']], on='geometry_id', how='left')
    merged_df['display_name'] = merged_df['display_name'].fillna('Unknown Room (ID ' + merged_df['geometry_id'].astype(str) + ')')
    
    # === MEMORY OPTIMIZATION – do BEFORE filtering (biggest savings) ===
    for col in ['metric_name', 'aggregation', 'display_name']:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].astype('category')
    
    # Efficient filtering: use loc + select only needed columns (no extra copy)
    needed_cols = ['timestamp', 'metric_name', 'aggregation', 'value', 'geometry_id', 'display_name']
    df = merged_df.loc[merged_df['is_valid'], needed_cols].copy(deep=False)  # shallow copy is safe here
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df, hr_df

# Load once
df, hr_df = load_and_prepare_data()

# === SESSION STATE ===
if 'filtered' not in st.session_state:
    st.session_state.filtered = False
if 'current_df' not in st.session_state:
    st.session_state.current_df = df.copy()

# === EXISTING HELPER FUNCTIONS (kept + minor cleanup) ===
def filter_df_by_date(start_str, end_str):
    try:
        start = pd.to_datetime(start_str).tz_localize('UTC')
        end = pd.to_datetime(end_str).tz_localize('UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
        return filtered if not filtered.empty else None
    except Exception as e:
        st.error(f"Date format error: {e}\nUse YYYY-MM-DD")
        return None


def get_top_rooms_current(metric_name, top_n=5):
    current = st.session_state.current_df
    agg = 'max' if metric_name == 'Occupancy' else 'mean'
    data = current[(current['metric_name'] == metric_name) & (current['aggregation'] == agg)]
    
    if data.empty:
        return f"No {metric_name} data in current period.", pd.DataFrame()
    
    latest_time = data['timestamp'].max()
    latest_data = data[data['timestamp'] == latest_time].sort_values('value', ascending=False).head(top_n)
    
    unit = {'Occupancy': 'people', 'temp': '°C', 'co2': 'ppm'}.get(metric_name, '')
    title = {'Occupancy': 'Busiest', 'temp': 'Hottest', 'co2': 'Highest CO₂'}.get(metric_name, metric_name)
    
    table = PrettyTable(["Rank", "Room", "Value"])
    table.align["Room"] = "l"
    
    for i, row in enumerate(latest_data.itertuples(), 1):
        val = f"{row.value:.1f} {unit}" if metric_name in ['temp', 'co2'] else f"{int(row.value)} {unit}"
        table.add_row([i, row.display_name, val])
    
    period = "(selected period)" if st.session_state.filtered else f"(latest: {latest_time.strftime('%Y-%m-%d %H:%M')})"
    result = f"**Top {top_n} {title} Rooms {period}**\n\n```\n{table}\n```"
    return result, latest_data


def plot_top_rooms_chart(metric_name, agg='max', top_n=5, color='skyblue', ylabel='Value'):
    current = st.session_state.current_df
    data = current[(current['metric_name'] == metric_name) & (current['aggregation'] == agg)]
    
    if data.empty:
        return None
    
    if agg == 'max':
        grouped = data.groupby('display_name')['value'].max().sort_values(ascending=False).head(top_n)
    else:
        grouped = data.groupby('display_name')['value'].mean().sort_values(ascending=False).head(top_n)
    
    if grouped.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(6, 4.8))
    grouped.plot.bar(ax=ax, color=color)
    
    period_title = ""
    if st.session_state.filtered:
        s_date = current['timestamp'].min().date()
        e_date = current['timestamp'].max().date()
        period_title = f" ({s_date} to {e_date})" if s_date != e_date else f" ({s_date})"
    
    ax.set_title(f"Top {top_n} {metric_name.capitalize()} Rooms{period_title}")
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Room')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(pad=1.2)
    
    return fig

def plot_heatmap(metric_name='Occupancy', agg='mean', cmap='YlOrRd'):
    try:
        current = st.session_state.current_df
    except:
        current = df
    
    if metric_name == 'Occupancy':
        agg = 'max'  # usually more meaningful for occupancy
    else:
        agg = 'mean'

    data = current[
        (current['metric_name'] == metric_name) &
        (current['aggregation'] == agg)
    ].copy()
    
    if data.empty:
        return None
    
    data['hour'] = data['timestamp'].dt.hour
    data['dayofweek'] = data['timestamp'].dt.day_name()
    
    # Order days properly
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    data['dayofweek'] = pd.Categorical(data['dayofweek'], categories=day_order, ordered=True)
    
    # Pivot: hour × dayofweek
    heatmap_data = data.groupby(['hour', 'dayofweek'])['value'].mean().unstack()
    
    # If missing days/hours, fill with NaN (heatmap will show)
    heatmap_data = heatmap_data.reindex(columns=day_order)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        annot=True,          # show numbers
        fmt=".0f",           # no decimal for occupancy/co2, but .1f for temp if you prefer
        linewidths=0.4,
        cbar_kws={'label': f'Avg {metric_name}'},
        ax=ax
    )
    
    ax.set_title(f"Average {metric_name} by Hour and Day of Week\n({len(data):,} measurements)")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day (0–23)")
    
    plt.tight_layout()
    return fig

def get_anomaly_rooms(metric_name):
    current = st.session_state.current_df
    data = current[current['metric_name'] == metric_name].copy()
    if data.empty:
        return pd.Series()
    data['z'] = (data['value'] - data['value'].mean()) / data['value'].std()
    high = data[data['z'] > 3]
    return high['display_name'].value_counts().sort_values(ascending=False)


def make_anomaly_table(metric_name, series, top_n=5):
    if series.empty:
        return f"No {metric_name} anomalies in period."
    table = PrettyTable(["Rank", "Room", f"{metric_name} Anomalies"])
    table.align["Room"] = "l"
    for i, (room, cnt) in enumerate(series.head(top_n).items(), 1):
        table.add_row([i, room, cnt])
    title = {'Occupancy': 'High Occupancy Spikes', 'temp': 'High Temp Spikes', 'co2': 'High CO₂ Spikes'}.get(metric_name, metric_name)
    return f"**Top {top_n} {title}**\n\n```\n{table}\n```"


# === RETRIEVER ===
def retriever_agent(query):
    ql = query.lower()
    import re
    m = re.search(r'(?:between|from)\s+(\d{4}-\d{2}-\d{2})(?:\s+(?:and|to)\s+(\d{4}-\d{2}-\d{2}))?', ql)
    if m:
        s = m.group(1)
        e = m.group(2) or s
        start = pd.to_datetime(s).tz_localize('UTC')
        end = pd.to_datetime(e).tz_localize('UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
        if not filtered.empty:
            st.session_state.current_df = filtered.copy()
            return filtered, f"**Data filtered to {s} → {e}** ({len(filtered):,} rows)"
        return df.copy(), "**No data in date range – using full data**"
    return st.session_state.current_df.copy(), "**Showing full available data**"

# === ANALYZER AGENT ===
def analyzer_agent(query):
    ql = query.lower().strip()

    date_pattern = r'(?:between|from)\s+(\d{4}-\d{2}-\d{2})(?:\s+(?:and|to)\s+(\d{4}-\d{2}-\d{2}))?'
    if re.search(date_pattern, ql) and len(ql.split()) <= 8:  # rough check it's mostly date
        return (
            "**Date filter applied successfully!**\n\n"
            "Data is now limited to the selected period.\n"
            "You can now ask questions like:\n"
            "- Hottest room right now?\n"
            "- Rooms with continuous high CO₂\n"
            "- Plot average temperature\n\n",
            "text"
        )

    # Current / Right now / Highest single-room questions
    if any(x in ql for x in ['right now', 'current', 'latest', 'just now', 'now', 'highest', 'top', 'worst']):
        if 'hottest' in ql or 'highest temp' in ql:
            ranking, msg = get_latest_room_ranking('temp', top_n=1)
            if ranking.empty:
                return "No temperature data right now.", "text"
            room = ranking.index[0]
            value = ranking.iloc[0]
            return f"**Hottest room right now**: {room} ({value:.1f} °C) {msg}", "text"

        if any(w in ql for w in ['highest co2', 'high co2', 'highest co₂', 'co2']):
            ranking, msg = get_latest_room_ranking('co2', top_n=1)
            if ranking.empty:
                return "No CO₂ data right now.", "text"
            room = ranking.index[0]
            value = ranking.iloc[0]
            return f"**Room with highest CO₂ right now**: {room} ({int(value)} ppm) {msg}", "text"

        if any(w in ql for w in ['busiest', 'highest occupancy', 'most occupied']):
            ranking, msg = get_latest_room_ranking('Occupancy', top_n=1)
            if ranking.empty:
                return "No occupancy data right now.", "text"
            room = ranking.index[0]
            value = ranking.iloc[0]
            return f"**Busiest room right now**: {room} ({int(value)} people) {msg}", "text"

# Flexible continuous high periods
    high_period_keywords = [
        'continuous', 'consecutive', 'longest', 'prolonged', 'long', 'hours',
        'period', 'periods', 'bad', 'poor', 'ventilat', 'too long'
    ]

    if any(kw in ql for kw in high_period_keywords):
        co2_indicators = ['co2', 'co₂', 'carbon', 'ppm', 'air quality', 'ventilation', 'co 2']
        if any(ind in ql for ind in co2_indicators):
            periods = find_long_high_periods('co2', threshold=1000, min_hours=3, top_n=15)
            if periods.empty:
                return "No long high CO₂ periods found.", "text"
            return (periods, "continuous_co2_table")
        
        temp_indicators = [
            'temp', 'temperature', 'hot', 'heat', 'warm', 'too hot',
            'overheat', 'high temperature', 'high temp', 'cooling'
        ]
        if any(ind in ql for ind in temp_indicators):
            periods = find_long_high_temp_periods(threshold=24, min_hours=3, top_n=15)
            if periods.empty:
                return "No long high temperature periods found.", "text"
            return (periods, "continuous_temp_table")
        
        return (
            "Please specify what is high:\n"
            "- high CO₂ / ppm / ventilation\n"
            "- high temperature / hot / temp",
            "text"
        )

    # Rooms usually above 21°C on average
    if any(w in ql for w in ['hot', 'above', 'higher', 'exceed', 'over', 'temperature', 'temp']) and \
       any(w in ql for w in ['average', 'avg', 'mean', 'usually', 'on average']) and \
       any(w in ql for w in ['room', 'rooms', '21', '22', '23']):
        
        bad = rooms_above_average('temp', threshold=21, top_n=12)
        if bad.empty:
            return "No rooms with average temperature >21°C in this period.", "text"
        
        lines = [f"{i}. {room}: {value:.1f}°C" for i, (room, value) in enumerate(bad.items(), 1)]
        return "\n".join(lines), "temp_avg_list"

    # Under-utilized rooms
    if any(w in ql for w in ['under', 'low', 'little', 'less', 'underutil', 'under utilised', 'under utilized', 'empty', 'unused', 'utili', 'utiliz', 'utilised', 'utilized']) and \
       any(w in ql for w in ['average', 'avg', 'on average', 'occupancy', 'people', 'usage', 'used', 'room', 'rooms', 'space']):
        
        low = rooms_above_average('Occupancy', threshold=5, top_n=15, ascending=True)
        if low.empty:
            return "No under-utilized rooms.", "text"
        
        lines = [f"{i}. {room}: {value:.1f} people" for i, (room, value) in enumerate(low.items(), 1)]
        return "\n".join(lines), "occ_list"

    # Heatmap requests
    if any(word in ql for word in ['heatmap', 'heat map', 'hour day', 'time pattern', 'daily pattern', 'weekly pattern', 'hourly pattern']):
        metric = 'Occupancy'
        cmap = 'YlOrRd'
        
        # Improved temp check – remove 'heat' to avoid false positives
        if any(w in ql for w in ['temp', 'temperature', 'hot', 'overheat']):
            metric = 'temp'
            cmap = 'Oranges'
        elif any(w in ql for w in ['co2', 'co₂', 'carbon', 'ppm', 'air quality', 'ventilat']):
            metric = 'co2'
            cmap = 'Greens'
        
        fig = plot_heatmap(metric_name=metric, cmap=cmap)
        if fig is not None:
            return fig, "heatmap"
        return "No data for heatmap.", "text"

    # Plot requests – correctly indented inside the function
    if any(word in ql for word in ['plot', 'bar', 'chart', 'graph', 'visualise', 'visualize', 'show', 'top rooms', 'top 10', 'ranking']):
        metric = 'Occupancy'
        agg = 'max'
        color = 'skyblue'
        ylabel = 'Max Occupancy (people)'

        # Force average if user mentions it
        if any(w in ql for w in ['average', 'avg', 'mean']):
            agg = 'mean'

        # Metric detection (expanded for CO2)
        if any(w in ql for w in ['temp', 'temperature', 'hot', 'heat']):
            metric = 'temp'
            agg = 'mean'
            color = 'orange'
            ylabel = 'Avg Temperature (°C)'

        elif any(w in ql for w in ['co2', 'co₂', 'carbon', 'air quality', 'ppm', 'ventilat']):
            metric = 'co2'
            agg = 'mean'  # always mean for CO2
            color = 'green'
            ylabel = 'Avg CO₂ (ppm)'

        # Special case for average occupancy
        if 'occupancy' in ql and agg == 'mean':
            ylabel = 'Avg Occupancy (people)'

        fig = plot_top_rooms_chart(
            metric_name=metric,
            agg=agg,
            top_n=10,
            color=color,
            ylabel=ylabel
        )

        if fig is not None:
            return fig, "plot"
        return "No data for plot", "text"


    # Fallback if nothing matched
    return "Query not recognized by analyzer.", "text"

# === RESPONSE AGENT – THIS IS THE FIXED VERSION ===
def response_agent(result, query):
    ql = query.lower().strip()

    # Heatmap display
    if isinstance(result, tuple) and len(result) == 2 and result[1] == "heatmap":
        return "**Hour-of-Day × Day-of-Week Heatmap** (average values):", result[0]

    # 1. Plot case
    if isinstance(result, tuple) and len(result) == 2 and result[1] == "plot":
        return "Here is the requested bar chart:", result[0]

    # 2. CONTINUOUS HIGH CO₂ PERIODS
    if (isinstance(result, tuple) and len(result) == 2 and result[1] == "continuous_co2_table") or \
       (isinstance(result, pd.DataFrame) and 'duration_hours' in result.columns and 'avg_temp' not in result.columns):
        
        df_periods = result[0] if isinstance(result, tuple) else result
        
        if df_periods.empty:
            st.markdown("No continuous high CO₂ periods found (≥1000 ppm for ≥3 consecutive hours).")
            return ""
        
        df_display = df_periods[['room', 'start_time', 'end_time', 'duration_hours']].copy()
        df_display['start_time'] = df_display['start_time'].dt.strftime('%Y-%m-%d %H:%M')
        df_display['end_time'] = df_display['end_time'].dt.strftime('%Y-%m-%d %H:%M')
        df_display = df_display.rename(columns={
            'room': 'Room',
            'start_time': 'Start Time',
            'end_time': 'End Time',
            'duration_hours': 'Duration (hours)'
        })
        
        st.markdown("**🔥 Longest continuous high CO₂ periods** (≥1000 ppm for ≥3 consecutive hours)")
        st.dataframe(
            df_display.sort_values('Duration (hours)', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        st.caption("Sorted by longest duration first")
        st.info("Tip: Very long periods indicate ventilation issues.")
        return ""

    # 3. CONTINUOUS HIGH TEMPERATURE PERIODS
    if (isinstance(result, tuple) and len(result) == 2 and result[1] == "continuous_temp_table") or \
       (isinstance(result, pd.DataFrame) and 'duration_hours' in result.columns and 'avg_temp' in result.columns):
        
        df_periods = result[0] if isinstance(result, tuple) else result
        
        if df_periods.empty:
            st.markdown("No continuous high temperature periods found (≥24°C for ≥3 consecutive hours).")
            return ""
        
        df_display = df_periods[['room', 'start_time', 'end_time', 'duration_hours', 'max_temp', 'avg_temp']].copy()
        df_display['start_time'] = df_display['start_time'].dt.strftime('%Y-%m-%d %H:%M')
        df_display['end_time'] = df_display['end_time'].dt.strftime('%Y-%m-%d %H:%M')
        df_display = df_display.rename(columns={
            'room': 'Room',
            'start_time': 'Start',
            'end_time': 'End',
            'duration_hours': 'Hours',
            'max_temp': 'Max °C',
            'avg_temp': 'Avg °C'
        })
        
        st.markdown("**🔥 Longest continuous high temperature periods** (≥24°C for ≥3 consecutive hours)")
        st.dataframe(
            df_display.sort_values('Hours', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        st.caption("Sorted by longest duration • Shows max & average temperature")
        st.info("Long periods may indicate cooling system or sensor issues.")
        return ""

    # 4. Average temperature >21°C
    if isinstance(result, str) and '°C' in result and \
       any(x in ql for x in ['average', 'avg', 'mean', 'usually', 'on average']):
        return (
            "**Rooms usually above 21°C on average** (sorted hottest → least hot):\n\n"
            f"{result}\n\n"
            "Tip: PES Floor 0 Reception is often significantly hotter — possible AC issue?"
        )

    # 5. Under-utilized rooms
    if isinstance(result, str) and 'people' in result and \
       any(x in ql for x in ['under', 'low', 'utili', 'utiliz']):
        return (
            "**Under-utilized rooms** (avg < 5 people, least → most used):\n\n"
            f"{result}\n\n"
            "Tip: Good candidates for energy savings."
        )

    # 6. Simple text answers
    if isinstance(result, str):
        return result

    return str(result)

def multi_agent_chain(query):
    """
    Simple multi-agent pipeline:
    1. Retriever → gets filtered data
    2. Analyst → does calculations & logic
    3. Visualizer → creates plot/heatmap if needed
    4. Formatter → builds beautiful final response
    """
    # Agent 1: Retriever
    current_df, filter_msg = retriever_agent(query)
    st.session_state.current_df = current_df  # update session state
    
    # Agent 2: Analyst
    analysis_result = analyzer_agent(query)
    
    # Determine result type
    if isinstance(analysis_result, tuple):
        analysis_content, rtype = analysis_result
    else:
        analysis_content = analysis_result
        rtype = "text"
    
    # Agent 3: Visualizer
    figure = None
    if rtype in ["plot", "heatmap"] and isinstance(analysis_content, plt.Figure):
        figure = analysis_content
        vis_msg = "Visualization generated successfully."
    else:
        vis_msg = "No visualization needed for this query."
    
    # Agent 4: Formatter (final writer)
    final_text = response_agent(analysis_content, query)
    
    # Add a small agentic touch: insight or note
    if "continuous" in query.lower() or "heatmap" in query.lower():
        final_text += "\n\n**Agent Insight**: This temporal analysis helps identify patterns for energy optimization and comfort improvement."
    
    # Return everything for chat display
    return final_text, "figure_ready" if figure else "text_only", figure

# === RAG SETUP ===
@st.cache_resource
def setup_rag():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunks = [f"Room: {r['display_name']} (ID: {r['geometry_id']})" for _, r in hr_df.iterrows()]
    
    for m in ['Occupancy', 'temp', 'co2']:
        txt, _ = get_top_rooms_current(m, 10)
        chunks.append(txt)
    
    hist = []
    for room in df['display_name'].unique():
        rd = df[df['display_name'] == room]
        occ = rd[rd['metric_name'] == 'Occupancy']['value'].mean() if 'Occupancy' in rd['metric_name'].values else 0
        tmp = rd[rd['metric_name'] == 'temp']['value'].mean() if 'temp' in rd['metric_name'].values else 0
        c2 = rd[rd['metric_name'] == 'co2']['value'].mean() if 'co2' in rd['metric_name'].values else 0
        hist.append(f"{room} | Avg Occ: {occ:.1f} | Avg Temp: {tmp:.1f}°C | Avg CO2: {c2:.1f} ppm")
    
    all_chunks = chunks + hist
    emb = model.encode(all_chunks, normalize_embeddings=True)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb.astype('float32'))
    return model, idx, all_chunks

model, index, chunks = setup_rag()


# === SIDEBAR ===
with st.sidebar:
    st.title("Controls")

    # LLM Mode Selection (keep your existing)
    st.markdown("### LLM Mode")
    use_local = st.checkbox(
        "Use Local Ollama (llama3.2:3b - faster & offline)",
        value=ollama_available,
        disabled=not ollama_available
    )
    if use_local and not ollama_available:
        st.warning("Ollama not detected. Install from https://ollama.com and run 'ollama run llama3.2:3b'")
        use_local = False
    if use_local:
        st.success("Using local Llama 3.2 3B (offline)")
        st.caption(
            "Tip: If local mode freezes:\n"
            "1. Restart Ollama (`ollama serve`)\n"
            "2. Refresh page (F5) or click button"
        )
        if st.button("♻️ Restart Local & Refresh"):
            st.rerun()
    else:
        st.info("Using Hugging Face cloud (Gemma-2-9b-it)")

    # ── DATE FILTER ────────────────────────────────────────────────
    st.markdown("### Date Filter")
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()

    start_date = st.date_input(
        "Start date",
        value=min_date if not st.session_state.filtered else st.session_state.current_df['timestamp'].min().date(),
        min_value=min_date,
        max_value=max_date
    )
    end_date = st.date_input(
        "End date",
        value=max_date if not st.session_state.filtered else st.session_state.current_df['timestamp'].max().date(),
        min_value=min_date,
        max_value=max_date
    )

    if st.button("Apply Date Filter"):
        if start_date > end_date:
            st.error("Start date must be before or equal to End date")
        else:
            filtered = filter_df_by_date(str(start_date), str(end_date))
            if filtered is not None and not filtered.empty:
                st.session_state.current_df = filtered.copy()
                st.session_state.filtered = True
                st.session_state.filter_message = f"Data filtered to {start_date} → {end_date} ({len(filtered):,} rows)"
                st.rerun()
            else:
                st.warning("No data in selected date range — using full data")

    # Reset button (keep your existing)
    if st.button("Reset Filter to Full Data"):
        st.session_state.filtered = False
        st.session_state.current_df = df.copy()
        st.session_state.filter_message = "Reset to full data!"
        st.rerun()

# === CURRENT PERIOD HEADER ===
if st.session_state.filtered:
    s = st.session_state.current_df['timestamp'].min().date()
    e = st.session_state.current_df['timestamp'].max().date()
    txt = f"**Showing data from {s} to {e}**" if s != e else f"**Showing data for {s}**"
    note = "(filtered period)"
else:
    txt = "**Showing full available data**"
    note = f"(latest: {df['timestamp'].max().strftime('%Y-%m-%d %H:%M UTC')})"

st.markdown(f"<h3 style='text-align:center;color:#1E88E5;'>{txt}</h3>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;'>{note}</p>", unsafe_allow_html=True)
st.markdown("---")


# === DYNAMIC TABLES ===
st.markdown("### Current Snapshot (Latest in selected period)")
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(get_top_rooms_current('Occupancy', 5)[0])
    with c2: st.markdown(get_top_rooms_current('temp', 5)[0])
    with c3: st.markdown(get_top_rooms_current('co2', 5)[0])

st.markdown("### Top Anomalies (in selected period)")
with st.container():
    c4, c5, c6 = st.columns(3)
    with c4: st.markdown(make_anomaly_table('Occupancy', get_anomaly_rooms('Occupancy')))
    with c5: st.markdown(make_anomaly_table('temp', get_anomaly_rooms('temp')))
    with c6: st.markdown(make_anomaly_table('co2', get_anomaly_rooms('co2')))


# === TOP 5 BAR CHARTS ===
st.markdown("### Top 5 Rooms Visual Summary (Selected Period)")
if st.session_state.current_df.empty:
    st.info("No data in the selected period for charts.")
else:
    col1, col2, col3 = st.columns(3, gap="small")
    
    with col1:
        st.markdown("**Top 5 Busiest Rooms (Max Occupancy)**")
        fig1 = plot_top_rooms_chart('Occupancy', agg='max', color='skyblue', ylabel='Max Occupancy (people)')
        if fig1 is not None:
            st.pyplot(fig1, use_container_width=True)
    
    with col2:
        st.markdown("**Top 5 Hottest Rooms (Mean Temperature)**")
        fig2 = plot_top_rooms_chart('temp', agg='mean', color='orange', ylabel='Avg Temp (°C)')
        if fig2 is not None:
            st.pyplot(fig2, use_container_width=True)
    
    with col3:
        st.markdown("**Top 5 Highest CO₂ Rooms (Mean)**")
        fig3 = plot_top_rooms_chart('co2', agg='mean', color='green', ylabel='Avg CO₂ (ppm)')
        if fig3 is not None:
            st.pyplot(fig3, use_container_width=True)


# === CHAT INTERFACE ===
st.markdown("### 💬 Full Agentic RAG Assistant! (Week B)")

chat_input = st.chat_input(
    "Ask anything... e.g. 'Hottest room right now?' | 'continuous high co2' | "
    "'heatmap occupancy' | 'heatmap temperature' | 'between 2025-04-01 and 2025-04-15' | "
    "'Plot average CO2' | 'under utilised rooms' | 'long continuous high temperature'"
)

if chat_input:
    with st.chat_message("user"):
        st.markdown(chat_input)
    
    with st.chat_message("assistant"):
        start_time = time.time()  # Start timing here — always!
        
        with st.spinner("Thinking..."):
            try:
                text, flag, fig = multi_agent_chain(chat_input)
                
                # Existing filter message handling...
                if any(phrase in text for phrase in ['Filtered to', 'No data in date range', 'Showing full available data']):
                    st.info(text.split('\n')[0] if '\n' in text else text)
                
                st.markdown(text)
                
                if flag == "figure_ready" and fig is not None:
                    st.pyplot(fig, use_container_width=True, clear_figure=True)
            
            except Exception as e:
                st.warning(f"Agent chain error: {str(e)}")
                st.info("Trying simple text fallback...")
                
                try:
                    # ... (your existing RAG fallback code) ...
                    
                    response = generate_llm_response(
                        messages=msgs,
                        temperature=0.65,
                        max_tokens=350,
                        stream_to_ui=True
                    )
                    st.markdown(response)
                
                except Exception as rag_e:
                    st.error(f"RAG fallback also failed: {str(rag_e)}")
                    st.info("Please check Ollama/HF connection or try simpler question.")
        
        # Always show time at the end of the assistant message
        end_time = time.time()
        response_time = end_time - start_time
        st.caption(f"**Response time: {response_time:.3f} seconds** ({'Local Ollama' if use_local else 'Hugging Face Cloud'})")

# === RESET ===
if st.button("Reset Filter to Full Data"):
    st.session_state.filtered = False
    st.session_state.current_df = df.copy()
    st.session_state.filter_message = "Reset to full data!"
    st.rerun()

st.success("Complete RAG Assistant Working! In shaa Allah distinction 🌟")
st.caption("HF Inference | Gemma-2-9b-it | Embeddings: all-MiniLM-L6-v2 | Local: Llama 3.2 3B")