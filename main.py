import locale
import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from time import sleep

dtypes = ['boolean', 'Int8', 'Int16', 'Int32', 'Int64', 'Float32',
          'Float64', 'datetime64', 'object']
numeric_dtypes = ['Int8', 'Int16', 'Int32', 'Int64', 'Float32', 'Float64']
int_dtypes = ['int8', 'int16', 'int32', 'int64',
              'Int8', 'Int16', 'Int32', 'Int64']

data = {
    "has_data": False,
    "raw": pd.DataFrame(),
    "clean": pd.DataFrame(),
    "summary": pd.DataFrame(),
    "columns": {
        "all": [],
        "continuous": [],
        "categorical": [],
        "numerical": [],
        "continuous_numerical": [],
    },
    "config": {
        "index": None,
        "drop_columns": [],
        "column_configs": {},
    }
}
page = {
    "data_tab": st.empty(),
    "raw_tab": st.empty(),
    "summary_tab": st.empty(),
    "config_tab": st.empty(),
    "data_file": st.empty(),
    "side":{
        "continuous":{
            "tab": st.empty(),
            "x_axis": {
                "select": st.empty(),
                "select_many": st.empty()
            },
            "y_axis": {
                "select": st.empty()
            }
        },
        "categorical":{
            "tab": st.empty(),
            "x_axis": {
                "select": st.empty(),
                "select_many": st.empty()
            },
            "y_axis": {
                "select": st.empty()
            }
        }
    },
    "config": {
        "index": st.empty(),
        "drop_columns": st.empty(),
    }
}
summary_config = {
    "genre": "Date Genre",
    "types": "Types",
    "unique": "Unique Values",
    "percent_null": st.column_config.NumberColumn(
        "Percentage Empty",
        format="percent",
    )
}


def read_session_data():
    # Initialise values from session_state
    if st.session_state.get('index_column'):
        data["config"]['index'] = st.session_state['index_column']

    if st.session_state.get('drop_columns'):
        data["config"]['drop_columns'] = st.session_state['drop_columns']

    if st.session_state.get('btn_continuous'):
        st.session_state['data_type'] = "Continuous"
        if st.session_state.get('x_axis_continuous'):
            st.session_state['x_axis_cols'] = st.session_state['x_axis_continuous']
    elif st.session_state.get('btn_categorical'):
        st.session_state['data_type'] = "Categorical"
        if st.session_state.get('x_axis_categorical'):
            st.session_state['x_axis_cols'] = st.session_state['x_axis_categorical']

    if st.session_state.get('column_config'):
        data["config"]["column_configs"] = st.session_state["column_config"]


def clear_session_data():
    # Clear values from session_state
    if st.session_state.get('index_column'):
        data["config"]['index'] = None
        st.session_state.pop('index_column')
    if st.session_state.get('drop_columns'):
        data["config"]['drop_columns'] = []
        st.session_state.pop('drop_columns')
    if st.session_state.get('x_axis_cols'):
        st.session_state.pop('x_axis_cols')
    if st.session_state.get('column_config'):
        data["config"]["column_configs"] = {}
        st.session_state.pop('column_config')


def draw_graph(chart_data, x_axis, y_axis, graph_type):
    """ Draws the graphs for the csv data.
    :param chart_data: (pandas.DataFrame) Dataframe containing the csv data.
    :param x_axis: (str) Name of the x-axis column.
    :param y_axis: (str) Name of the y-axis column.
    :param graph_type: (str) The type of graph to be drawn (histogram, scatter, etc.).
    """
    x_data = chart_data.get(x_axis)
    y_data = chart_data.get(y_axis)
    if graph_type == "histogram":
        if y_data is None:
            plot = px.histogram(chart_data, x=x_data)
        else:
            plot = px.histogram(chart_data, x=x_data, y=y_data, histfunc='avg')
        sleep(0.1)
        st.plotly_chart(plot, use_container_width=True)
    else:
        plot = px.scatter(chart_data, x=x_data, y=y_data)
        sleep(0.1)
        st.plotly_chart(plot, use_container_width=True)
    return plot


def convert_column_types(df):
    column_configs = data["config"]["column_configs"]
    df = df.convert_dtypes()
    for col in column_configs:
        dtype = column_configs[col]["dtype"]
        if dtype is not None:
            if df[col].dtype in ['object','string'] and dtype in numeric_dtypes:
                df[col] = df[col].str.replace(",", "")
            try:
                df[col] = df[col].astype(dtype)
            except ValueError:
                st.error(f"Could not convert '{col}' to '{dtype}'", icon="🚨")

    for col in df.columns:
        if df[col].dtype in ['object','string']:
            try:
                df[col] = pd.to_datetime(df[col])
            except ValueError:
                pass
    return df


def clean_dataset(raw):
    config = data["config"]
    column_configs = config["column_configs"]

    if config["index"] is not None:
        raw.set_index(config["index"], drop=True, inplace=True)

    clean_data = raw.copy()

    for col in column_configs:
        col_config = column_configs[col]
        dtype = clean_data[col].dtype
        if col_config["dtype"] is not None:
            dtype = col_config["dtype"]
        fill_value = None
        if col_config["fill_value"]:
            fill_value = get_fill_value(clean_data[col],
                                        col_config["fill_value"])

        if fill_value is not None:
            try:
                if dtype is not None and dtype in int_dtypes:
                    fill_value = int(fill_value)
            except ValueError:
                st.error(f"Could not convert '{fill_value}' to '{dtype}'",)
            clean_data[col] = clean_data[col].fillna(fill_value)

    if len(config["drop_columns"]) > 0:
        clean_data.drop(config["drop_columns"], axis=1, inplace=True)

    return raw, clean_data

def get_fill_value(data_series, method):
    try:
        if method == "mean":
            return data_series.mean()
        elif method == "mode":
            return data_series.mode()
        elif method == "median":
            return data_series.median()
        else:
            return method
    except TypeError:
        return None


def load_data(data_file):
    if data_file is not None:
        raw_data = pd.DataFrame()
        try:
            has_data = False
            file_type = data_file.name.split(".")[-1]
            if file_type == "csv":
                raw_data = pd.read_csv(data_file, encoding_errors='ignore')
                has_data = True
            elif file_type in ["xlsx", "xls"]:
                raw_data = pd.read_excel(data_file)
                has_data = True
            data["has_data"] = has_data
            data["raw"] = raw_data
        except UnicodeDecodeError:
            print(f"{data_file.name} could not be decoded.")
        except Exception as e:
            print(f"{data_file.name} could not be read.", e)

        if not data["has_data"]:
            return

        raw_data = convert_column_types(raw_data)
        raw_data, clean_data = clean_dataset(raw_data)
        data["clean"] = clean_data
        data["raw"] = raw_data

        row_count = len(clean_data)
        types = pd.Series(raw_data.dtypes, name='types')
        counts = pd.Series(raw_data.nunique(), name='unique')
        nulls = pd.Series(raw_data.isnull().sum() / row_count,
                          name='percent_null')
        summary = pd.concat([types, counts, nulls], axis=1)
        continuous_limit = min(25, len(clean_data) // 10)
        summary["genre"] = np.where(summary['unique'] > continuous_limit,
                                    "Continuous", "Categorical")
        data["summary"] = summary

        active_summary = summary.loc[list(clean_data.columns.values)]
        columns =  data["columns"]
        columns["all"] = raw_data.columns.to_list()
        columns["numerical"] = clean_data.select_dtypes(
            include=['number', 'datetime']).columns.to_list()
        columns["continuous"] = active_summary.loc[
            active_summary["genre"] == "Continuous"].index.tolist()
        columns["categorical"] = active_summary.loc[
            active_summary["genre"] == "Categorical"].index.tolist()
        columns["continuous_numerical"] = [col for col in columns["numerical"]
                                           if col in columns["continuous"]]


def set_x_axis_columns(data_type, columns):
    st.session_state['data_type'] = data_type
    st.session_state['x_axis_cols'] = columns


def create_select_boxes():
    columns = data["columns"]
    continuous = page["side"]["continuous"]
    with continuous["tab"]:
        cont_form = st.form(key="continuous_form")
        continuous_numerical = columns["continuous_numerical"]
        x_continuous = continuous["x_axis"]
        x_continuous_select = cont_form.multiselect("X-Axis (Select up to 8)",
                                                    continuous_numerical,
                                                    key="x_axis_continuous",
                                                    max_selections=8)
        x_continuous["select_many"] = x_continuous_select
        y_continuous = continuous["y_axis"]
        y_continuous_select = cont_form.selectbox("Y-Axis (Select One)",
                                                  continuous_numerical,
                                                  key="y_axis_continuous",
                                                  index=0)
        y_continuous["select"] = y_continuous_select
        cont_form.form_submit_button("Draw Graphs",
                                     type="primary",
                                     key="btn_continuous")

    categorical = page["side"]["categorical"]
    with categorical["tab"]:
        cat_form = st.form(key="categorical_form")
        categorical_columns = columns["categorical"]
        x_categorical = categorical["x_axis"]
        x_categorical_select = cat_form.multiselect("X-Axis",
                                                    categorical_columns,
                                                    key="x_axis_categorical")
        x_categorical["select_many"] = x_categorical_select
        y_categorical = categorical["y_axis"]
        y_categorical["select"] = cat_form.selectbox("Y-Axis",
                                                     continuous_numerical,
                                                     key="y_axis_categorical",
                                                     index=0)
        cat_form.form_submit_button("Draw Graphs",
                                    type="primary",
                                    key="btn_categorical")


def create_graphs():
    if st.session_state.get('x_axis_cols'):
        data_type = st.session_state['data_type']
        x_axis_cols = st.session_state['x_axis_cols']
        st.session_state.pop('x_axis_cols')
        y_continuous = page["side"]["continuous"]["y_axis"]["select"]
        y_categorical = page["side"]["categorical"]["y_axis"]["select"]
        y_axis_val = y_continuous if data_type == 'Continuous' else y_categorical
        graph_type = "histogram" if data_type == 'Categorical' else "scatter"
        clean_data = data["clean"]
        if len(x_axis_cols) > 3:
            col1, col2 = st.columns(2, gap="large")
            for index, x_option in enumerate(x_axis_cols):
                if index % 2 == 0:
                    with col1:
                         draw_graph(clean_data, x_option, y_axis_val, graph_type)
                else:
                     with col2:
                         draw_graph(clean_data, x_option, y_axis_val, graph_type)
        else:
            for index, x_option in enumerate(x_axis_cols):
                draw_graph(clean_data, x_option, y_axis_val, graph_type)


def get_color_indicator(val):
    if val == 0:
        return "\U0001F7E9"
    elif 0 < val < 25:
        return "\U0001F7E8"
    elif 25 <= val <= 50:
        return "\U0001F7E7"
    else:
        return "\U0001F7E5"


def format_drop_options(col):
    option_text = col
    percent_null = data["summary"].loc[col]["percent_null"]
    if percent_null is not None:
        percentage = math.ceil(percent_null * 100)
        color = get_color_indicator(percentage)
        option_text = f"{color} {col} - {percentage}% null"
    return option_text


def config_section():
    elements = page["config"]
    index_options = data["clean"].columns
    ix = None
    index_col = data["config"]["index"]
    if index_col and index_col not in index_options:
        ix = 0
        index_options = index_options.insert(ix, index_col)

    with page["config_tab"]:
        config_form = st.form(key="config_form", enter_to_submit=False,
                              border=True)

    with config_form:
        col_index, col_drop = st.columns(2)
        with col_index:
            st.subheader(":material/key_vertical: Index")
            elements["index"] = st.selectbox("Select the index for the dataset",
                                             index_options, key="index_column",
                                             index=ix)

        with col_drop:
            selected_cols = data["config"]['drop_columns']
            st.subheader(":material/delete_forever: Drop Columns")
            elements["drop_columns"] = st.multiselect("Columns to drop from the dataset",
                                                      data["columns"]["all"],
                                                      default=selected_cols,
                                                      key="drop_columns",
                                                      format_func=format_drop_options)

        st.form_submit_button("Apply", type="primary",
                                       key="config_form_submit")
    with page["config_tab"]:
        config_cols = st.container(border=True)
        with config_cols:
            st.subheader(":material/amend: Configure Columns")

            cols = st.columns([2, 2, 2, 2, 1])
            cols[0].write("Column")
            cols[1].write("Data Type")
            cols[2].write("Fill Nulls")
            cols[3].write("Description")
            cols[4].write("Action")
            column_configs = data["config"]["column_configs"]
            options = data["clean"].columns.tolist()
            unused_options = [opt for opt in options if opt not in column_configs]
            for column in column_configs:
                col_config = column_configs[column]
                create_column_config_row(cols, options, column, col_config)

            create_column_config_row(cols, unused_options)


def create_column_config_row(cols, options, column=None, values=None):
    fill_options = ["0", "mean", "median", "mode"]
    has_values = values is not None
    prefix = column if column is not None else ""
    column = options.index(column) if column is not None else None

    if has_values and values["dtype"] is not None:
        dtype = dtypes.index(values["dtype"])
    else:
        dtype = None

    if has_values and values["fill_value"] is not None:
        if values["fill_value"] not in fill_options:
            fill_options.append(values["fill_value"])
        fill_value = fill_options.index(values["fill_value"])
    else:
        fill_value = None

    description = values["description"] if values is not None else None
    action1 = ":material/refresh:" if values is not None else ":material/add:"
    prevent_col_change = values is not None
    col_select = cols[0].selectbox("Column", options, index=column,
                                   label_visibility="collapsed",
                                   key=prefix + "_column_select",
                                   disabled=prevent_col_change,
                                   format_func=format_drop_options)
    type_select = cols[1].selectbox("Data Type", dtypes, index=dtype,
                                    help="Select Data Type for the Column",
                                    label_visibility="collapsed",
                                    key=prefix + "_type_select")
    fill_select = cols[2].selectbox("Fill Nulls", fill_options,
                                    help="Select Data Type for the Column",
                                    index=fill_value, accept_new_options=True,
                                    label_visibility="collapsed",
                                    key=prefix + "_fill_select")
    desc_text = cols[3].text_input("Description", label_visibility="collapsed",
                                   placeholder="Enter column description...",
                                   value=description,
                                   key=prefix + "_desc_text")
    col_add, col_del = cols[4].columns(2, gap="small", width=100)
    col_add.button(action1, type="secondary", key=prefix + "_action1",
                   on_click=set_column_config,
                   args=[col_select, type_select, fill_select, desc_text])
    if values is not None:
        col_del.button(":material/delete:", type="secondary", key=prefix + "_action2",
                       on_click=delete_column_config,
                       args=[col_select])


def set_column_config(col_name, data_type, fill_value, description):
    if col_name is not None:
        col_config = {"dtype": data_type,
                      "fill_value": fill_value, "description": description}
        data["config"]["column_configs"][col_name] = col_config
    st.session_state["column_config"] = data["config"]["column_configs"]

def delete_column_config(col_name):
    data["config"]["column_configs"].pop(col_name)
    st.session_state["column_config"] = data["config"]["column_configs"]


def file_uploader():
    with st.sidebar:
        st.subheader(":material/settings: Configuration")
        data_file = st.file_uploader(":material/upload: Upload a CSV file",
                                    type=['csv', 'xls', 'xlsx'],
                                    on_change=clear_session_data)
        page["data_file"] = data_file

    load_data(data_file)


def main_section():
    st.title(":material/analytics: Data Analysis, Cleaning, and Visualiser")
    st.write("Create graphical visualisations from data files")

    data_tab, raw_tab, summary_tab, config_tab = st.tabs(
        [":material/data_table: Clean Data",
         ":material/data_table: Raw Data",
         ":material/data_object: Summary",
         ":material/build: Config"])
    page["data_tab"] = data_tab
    page["raw_tab"] = raw_tab
    page["summary_tab"] = summary_tab
    page["config_tab"] = config_tab

    if data["has_data"]:
        data_tab.write(data["clean"])
        raw_tab.write(data["raw"])
        page["summary_tab"].dataframe(data["summary"],
                                      column_config=summary_config)


read_session_data()

st.set_page_config(
    page_title="CSV Data File Visualiser",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS to set the width of the sidebar
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 408px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

file_uploader()

main_section()

if len(data["raw"]) > 0:
    st.divider()
    tab_cont, tab_cat = st.sidebar.tabs(["Continuous", "Categorical"])
    page["side"]["continuous"]["tab"] = tab_cont
    page["side"]["categorical"]["tab"] = tab_cat
    create_select_boxes()

    st.sidebar.divider()
    config_section()

create_graphs()
