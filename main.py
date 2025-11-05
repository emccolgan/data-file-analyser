import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from time import sleep

dtypes = ['boolean', 'Int8', 'Int16', 'Int32', 'Int64', 'Float32',
          'Float64', 'datetime64', 'string', 'object', 'category']
numeric_dtypes = ['int8', 'int16', 'int32', 'int64', 'float32', 'float64',
                  'Int8', 'Int16', 'Int32', 'Int64', 'Float32', 'Float64']
int_dtypes = ['int8', 'int16', 'int32', 'int64', 'Int8', 'Int16', 'Int32', 'Int64']
chart_types = ['scatter', 'heatmap', 'histogram', 'line', 'bar', 'box plot']
agg_funcs = {"Count (Unique Values)": "count", "Sum": "sum", "Mean": "mean",
             "Median": "median", "Standard Deviation": "std",
             "Variance": "var", "Max": "max", "Min": "min"}
hist_funcs = {"Count": "count", "Sum": "sum", "Mean": "avg",
              "Minimum": "min", "Maximum": "max"}
color_scales = ['Solid'] + px.colors.named_colorscales()

data = {
    "has_data": False,
    "filename": None,
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
    "correlation_tab": st.empty(),
    "config_tab": st.empty(),
    "data_file": st.empty(),
    "chart_type": None,
    "chart_display": None,
    "agg_func": None,
    "color_scale": "Solid",
    "variables":{
        "dependent": {
            "select": st.empty(),
            "values": []
        },
        "independent": {
            "select": st.empty(),
            "values": []
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
    "unique": "Cardinality (Unique Values)",
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

    dependent_var = st.session_state.get('dependent_var')
    if dependent_var:
        page["variables"]["dependent"]["values"] = dependent_var
        st.session_state['dependent_cols'] = dependent_var

    independent_var = st.session_state.get('independent_var')
    if independent_var:
        page["variables"]["independent"]["values"] = independent_var

    if st.session_state.get('column_config'):
        data["config"]["column_configs"] = st.session_state["column_config"]

    if st.session_state.get('chart_type'):
        page["chart_type"] = st.session_state["chart_type"]

    if st.session_state.get('chart_display'):
        page["chart_display"] = st.session_state["chart_display"]

    if st.session_state.get('agg_func'):
        page["agg_func"] = st.session_state["agg_func"]

    if st.session_state.get('color_scale'):
        page["color_scale"] = st.session_state["color_scale"]

    data_file = st.session_state.get('data_file')
    if data_file:
        data["filename"] = data_file.name


def clear_session_data():
    # Clear values from session_state
    if st.session_state.get('index_column'):
        data["config"]['index'] = None
        st.session_state.pop('index_column')
    if st.session_state.get('drop_columns'):
        data["config"]['drop_columns'] = []
        st.session_state.pop('drop_columns')
    if st.session_state.get('dependent_cols'):
        st.session_state.pop('dependent_cols')
    if st.session_state.get('column_config'):
        data["config"]["column_configs"] = {}
        st.session_state.pop('column_config')


def aggregate_dataset(dataset, group, agg_func):
    if agg_func == "mean":
        return dataset.groupby(group, as_index=False).mean()
    elif agg_func == "median":
        return dataset.groupby(group, as_index=False).median()
    elif agg_func == "mode":
        return dataset.groupby(group, as_index=False).mode()
    elif agg_func == "std":
        return dataset.groupby(group, as_index=False).std()
    elif agg_func == "var":
        return dataset.groupby(group, as_index=False).var()
    elif agg_func == "max":
        return dataset.groupby(group, as_index=False).max()
    elif agg_func == "min":
        return dataset.groupby(group, as_index=False).min()
    elif agg_func == "sum":
        return dataset.groupby(group, as_index=False).sum()
    elif agg_func == "count":
        return dataset.groupby(group, as_index=False).nunique()
    else:
        return dataset


def get_axis_label(label, agg=""):
    if agg:
        full_label = agg.title() + " of " + label.replace("_", " ").title()
    else:
        full_label = label.replace("_", " ").title()
    return full_label

def draw_graph(chart_data, x_axis, y_axis, graph_type, color_scale):
    """ Draws the graphs for the csv data.
    :param chart_data: (pandas.DataFrame) Dataframe containing the csv data.
    :param x_axis: (str) Name of the x-axis column.
    :param y_axis: (str) Name of the y-axis column.
    :param graph_type: (str) The type of graph to be drawn (histogram, scatter, etc.).
    :param color_scale: (str) The color scale to use for the graph
    """
    plot = None
    key = f"{graph_type}_{x_axis}_{y_axis}"
    st.subheader(f"{x_axis} vs. {y_axis}")
    use_color_sequence = color_scale != "Solid"
    if graph_type == "scatter":
        if use_color_sequence:
            plot = px.scatter(chart_data, x=x_axis, y=y_axis, color=y_axis,
                              color_continuous_scale=color_scale)
        else:
            plot = px.scatter(chart_data, x=x_axis, y=y_axis)
        plot.update_xaxes(title_text=get_axis_label(x_axis))
        if isinstance(y_axis, str):
            plot.update_yaxes(title_text=get_axis_label(y_axis))
        sleep(0.1)
        st.plotly_chart(plot, use_container_width=True, key=key)
    elif graph_type == "heatmap":
        if use_color_sequence:
            plot = px.density_heatmap(chart_data, x=x_axis, y=y_axis,
                                      color_continuous_scale=color_scale)
        else:
            plot = px.density_heatmap(chart_data, x=x_axis, y=y_axis)
        plot.update_xaxes(title_text=get_axis_label(x_axis))
        if isinstance(y_axis, str):
            plot.update_yaxes(title_text=get_axis_label(y_axis))
        sleep(0.1)
        st.plotly_chart(plot, use_container_width=True, key=key)
    elif graph_type == "histogram":
        agg_name = page.get("agg_func")
        agg_func = hist_funcs.get(agg_name) if agg_name else 'avg'
        y_label = agg_name if isinstance(y_axis, list) else get_axis_label(y_axis,
                                                                           agg_name)
        if y_axis is None:
            plot = px.histogram(chart_data, x=x_axis, histfunc=agg_func)
            plot.update_xaxes(title_text=x_axis.title())
            plot.update_yaxes(title_text=agg_name.title())
        else:
            plot = px.histogram(chart_data, x=x_axis, y=y_axis, histfunc=agg_func)
            plot.update_xaxes(title_text=x_axis.title())
            plot.update_yaxes(title_text=y_label)
        sleep(0.1)
        st.plotly_chart(plot, use_container_width=True, key=key)
    elif graph_type == "line":
        agg_name = page.get("agg_func")
        agg_func = agg_funcs.get(agg_name) if agg_name else 'mean'
        agg_data = aggregate_dataset(chart_data, x_axis, agg_func)
        y_label = agg_name if isinstance(y_axis, list) else get_axis_label(y_axis,
                                                                           agg_name)
        plot = px.line(agg_data, x=x_axis, y=y_axis)
        plot.update_xaxes(title_text=x_axis.title())
        plot.update_yaxes(title_text=y_label)
        sleep(0.1)
        st.plotly_chart(plot, use_container_width=True, key=key)
    elif graph_type == "bar":
        agg_name = page.get("agg_func")
        agg_func = agg_funcs.get(agg_name) if agg_name else 'mean'
        agg_data = aggregate_dataset(chart_data, x_axis, agg_func)
        y_label = agg_name if isinstance(y_axis, list) else get_axis_label(y_axis,
                                                                           agg_name)
        plot = px.bar(agg_data, x=x_axis, y=y_axis)
        plot.update_xaxes(title_text=x_axis.title())
        plot.update_yaxes(title_text=y_label)
        sleep(0.1)
        st.plotly_chart(plot, use_container_width=True, key=key)
    elif graph_type == "box plot":
        plot = px.box(chart_data, x=x_axis, y=y_axis)
        plot.update_xaxes(title_text=get_axis_label(x_axis))
        if isinstance(y_axis, str):
            plot.update_yaxes(title_text=get_axis_label(y_axis))
        sleep(0.1)
        st.plotly_chart(plot, use_container_width=True, key=key)
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
            except (ValueError, TypeError):
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
                raw_data = pd.read_csv(data_file, thousands=',',
                                       parse_dates=True,
                                       infer_datetime_format=True,
                                       encoding_errors='ignore')
                has_data = True
            elif file_type in ["xlsx", "xls"]:
                raw_data = pd.read_excel(data_file, parse_dates=True)
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


def create_select_boxes(chart_form):
    columns = data["columns"]
    var_data = page["variables"]
    chart_type = page["chart_type"]
    max_charts = 8 if chart_type == "scatter" else None
    all_columns = columns["all"]
    independent_var = var_data["independent"]
    independent_var_select = chart_form.selectbox("Independent Variable",
                                                 all_columns,
                                                 key="independent_var",
                                                 index=0)
    independent_var["select"] = independent_var_select
    dependent_var = var_data["dependent"]
    default_dependents = dependent_var["values"]
    if max_charts is not None:
        default_dependents = default_dependents[:max_charts]
    dependent_var_select = chart_form.multiselect("Dependent Variables",
                                                 all_columns, default=default_dependents,
                                                 key="dependent_var",
                                                 max_selections=max_charts)
    dependent_var["select"] = dependent_var_select
    chart_form.form_submit_button("Draw Graphs",
                                 type="primary",
                                 key="btn_var_data")


def create_graphs():
    if st.session_state.get('dependent_cols'):
        dependent_cols = st.session_state['dependent_cols']
        st.session_state.pop('dependent_cols')
        independent_var = page["variables"]["independent"]["select"]
        independent_val = independent_var
        graph_type = page["chart_type"]
        clean_data = data["clean"]
        chart_display = page["chart_display"]
        color_scale = page["color_scale"]
        multiple_series = True if chart_display == "Multiple Series" else False
        if multiple_series:
            draw_graph(clean_data, independent_val, dependent_cols,
                       graph_type, color_scale)
        else:
            if len(dependent_cols) > 3:
                col1, col2 = None, None
                for index, dependent_val in enumerate(dependent_cols):
                    if col1 is None:
                        col1, col2 = st.columns(2, border=True)
                    if index % 2 == 0:
                        with col1:
                             draw_graph(clean_data, independent_val,
                                        dependent_val, graph_type, color_scale)
                    else:
                        with col2:
                             draw_graph(clean_data, independent_val,
                                        dependent_val, graph_type, color_scale)
                        col1, col2 = None, None
            else:
                for index, y_option in enumerate(dependent_cols):
                    draw_graph(clean_data, independent_val, y_option,
                               graph_type, color_scale)


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


def create_correlation():
    chart_title = "Correlation between numerical features"
    # Filter only numerical data
    numeric_data = data["clean"].select_dtypes(include=numeric_dtypes)
    corr = numeric_data.corr().round(2)
    fig = px.imshow(corr, text_auto=True, width=700, height=600, labels={
        "x":"X Feature",
        "y":"Y Feature",
        "color":"Correlation"
    }, color_continuous_scale=px.colors.diverging.RdBu)
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        title={
            "text": chart_title,
            "x": 0.5,
            "xanchor": "center",
            "font":{"size":24}
        })
    st.plotly_chart(fig, use_container_width=False)

def config_section():
    elements = page["config"]
    index_options = data["clean"].columns
    ix = None
    index_col = data["config"]["index"]
    if index_col and index_col not in index_options:
        ix = 0
        index_options = index_options.insert(ix, index_col)

    with page["correlation_tab"]:
        create_correlation()

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


def format_chart_type_options(col):
    option_text = col.capitalize()
    if col == "scatter":
        option_text += " (max. of 8 variables)"
    return option_text


def main_section():
    st.title(":material/analytics: Data Analysis, Cleaning, and Visualiser")
    st.write("Create graphical visualisations from data files")

    if data["filename"] is not None:
        st.subheader(data["filename"])

    (data_tab, raw_tab, summary_tab,
     correlation_tab, config_tab) = st.tabs(
        [":material/data_table: Clean Data",
         ":material/data_table: Raw Data",
         ":material/data_object: Summary",
         ":material/key_visualizer: Correlation",
         ":material/build: Config"])
    page["data_tab"] = data_tab
    page["raw_tab"] = raw_tab
    page["summary_tab"] = summary_tab
    page["correlation_tab"] = correlation_tab
    page["config_tab"] = config_tab

    if data["has_data"]:
        data_tab.write(data["clean"])
        raw_tab.write(data["raw"])
        page["summary_tab"].dataframe(data["summary"],
                                      column_config=summary_config)


def sidebar_config():
    with st.sidebar:
        st.subheader(":material/dataset: Dataset")
        data_file = st.file_uploader(":material/upload: Upload a data file",
                                     type=['csv', 'xls', 'xlsx'],
                                     on_change=clear_session_data,
                                     key="data_file")
        page["data_file"] = data_file

    load_data(data_file)

    if len(data["raw"]) > 0:
        form_box = st.sidebar.container(border=True)
        with form_box:
            st.subheader(":material/chart_data: Chart")

            chart_type = st.selectbox("Chart Type", chart_types,
                                      key="chart_type", index=0,
                                      format_func=format_chart_type_options)

            if chart_type in ["histogram", "bar", "line"]:
                if chart_type == "histogram":
                    aggregates = hist_funcs.keys()
                else:
                    aggregates = agg_funcs.keys()
                st.selectbox("Aggregate Function", aggregates,
                             key="agg_func", index=0)

            if chart_type in ["scatter", "heatmap"]:
                st.selectbox("Color Scale", color_scales,
                             key="color_scale", index=0)

            chart_form = st.form(key="all_data_form", border=False)
            chart_form.segmented_control("Display",
                                 ["Multiple Graphs", "Multiple Series"],
                                 default="Multiple Graphs",
                                 selection_mode="single",
                                 key="chart_display")

        create_select_boxes(chart_form)

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

sidebar_config()

main_section()

if len(data["raw"]) > 0:
    config_section()

create_graphs()
