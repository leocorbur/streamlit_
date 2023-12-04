# Import the required libraries
import streamlit as st
import lasio as ls
import pandas as pd
import io
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

def read_las_file(u_file):
    las_file_contents = u_file.read()
    las_file_contents_str = las_file_contents.decode("utf-8")
    las_file_buffer = io.StringIO(las_file_contents_str)
    las = ls.read(las_file_buffer)
    df = las.df()
    df.reset_index(inplace=True)
    df.index = df.index + 1
    return df

def display_main_data(df):
    st.header('Main Data')
    st.subheader('Header')
    st.write(df.head())
    st.subheader('Tail')
    st.write(df.tail())
    st.subheader('Statistics')
    st.write(df.describe())
    dim = st.radio('Data dimension:', ('Rows', 'Columns'), horizontal=True)
    if dim == 'Rows':
        st.write('Number of rows: ', df.shape[0])
    else:
        st.write('Number of columns:', df.shape[1])

def display_missingno_plots(df):
    st.subheader('A nullity matrix')
    msno.matrix(df)
    st.pyplot()

def select_columns(df):
    st.subheader('Column selection')
    selected_columns = st.multiselect('Select at least 03 columns you want to interact with, and include a depth column for log visualization purposes:', 
                                      ['All columns'] + list(df.columns))
    if 'All columns' in selected_columns:
        selected_columns = df.columns 
    df_filtered = df[selected_columns]
    return df_filtered, selected_columns

def handle_null_values(df_filtered, selected_columns):
    st.subheader('What do you want to do with null values?')
    st.text('Please choose one. Delete null values or imputation methods.')
    st.text('By default, the imputation methods are applied automatically')

    delete_nulls = st.checkbox('Delete rows with null values in any column')
    if delete_nulls:
        df_filtered = df_filtered.dropna()
    else:
        for column in selected_columns:
            imputation_method = st.selectbox(f'Select imputation method for {column}:', ['Mean', 'Median', 'Specific Value'])
            
            if imputation_method == 'Mean':
                imputer = SimpleImputer(strategy='mean')
                df_filtered[column] = imputer.fit_transform(df_filtered[[column]])
            elif imputation_method == 'Median':
                imputer = SimpleImputer(strategy='median')
                df_filtered[column] = imputer.fit_transform(df_filtered[[column]])
            elif imputation_method == 'Specific Value':
                specific_value = st.number_input(f'Enter the specific value for {column}:')
                df_filtered[column] = df_filtered[column].fillna(specific_value)

    return df_filtered

def display_selected_data(df_filtered):
    st.title('Selected Data')
    st.write(df_filtered.head())
    dimension = st.radio('Dimension of filtered df:', ('Rows', 'Columns'), horizontal=True)
    if dimension == 'Rows':
        st.write('Number of rows: ', df_filtered.shape[0])
    else:
        st.write('Number of columns:', df_filtered.shape[1])

def display_boxplots(df_filtered):
    st.title('Boxplots')
    red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')

    # Allow the user to select logarithmic columns
    log_columns = st.multiselect('Select logarithmic columns:', df_filtered.columns)

    # Create box plots using Streamlit and Matplotlib
    fig, axs = plt.subplots(1, len(df_filtered.columns), figsize=(30, 10))

    for i, ax in enumerate(axs.flat):
        ax.boxplot(df_filtered.iloc[:, i], flierprops=red_circle)
        ax.set_title(df_filtered.columns[i], fontsize=20, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)

        # Check if column names match the expected logarithmic columns
        if df_filtered.columns[i] in log_columns:
            ax.semilogy()

    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(fig)

def plot_well_logs(df, x_columns, y_column, log_scale_columns=None, xlims=None, ylim=None, color_palette='colorblind'):

    if log_scale_columns is None:
        log_scale_columns = []

    if xlims is None:
        xlims = {}

    if ylim is None:
        ylim = (df[y_column].max(), df[y_column].min())

    fig, axes = plt.subplots(1, len(x_columns), figsize=(10, 10), sharey=True) 

    # Obtener la paleta de colores
    colors = sns.color_palette(color_palette, n_colors=len(x_columns))

    for i, column in enumerate(x_columns):
        ax = axes [i]

        ax.plot(df[column], y_column, data=df, color=colors[i], linewidth=0.5) 
        ax.set_xlim(xlims.get(column, (df[column].min(), df[column].max())))

        if column in log_scale_columns:
            ax.set_xscale('log') 

        ax.set_ylim(ylim)
        ax.grid()

        # Mover las etiquetas del eje x al top
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        ax.set_xlabel(column)

        if i > 0:
            plt.setp(ax.get_yticklabels(), visible=False)

    fig.subplots_adjust(wspace=0.05)

    return fig, axes

def display_log_data_viz(df_filtered):
    st.title('Log Data Viz')
    if not df_filtered.empty:
        default_x_columns = df_filtered.columns[1:3].tolist()

        x_columns = st.multiselect('Select at least 2 columns for the X axis, do not select a depth column:', df_filtered.columns,key='x_columns',
                                   default=default_x_columns)
        y_column = st.selectbox('Select a depth column for the Y axis:', df_filtered.columns, key='y_column')
        log_scale_columns = st.multiselect('Select logarithmic columns:', df_filtered.columns, key='log_scale_columns')

        xlims = {}
        ylim = ()

        for x_column in x_columns:
            st.write(f'You have selected: {x_column}')
            min_value = st.number_input(f'Enter the minimum value for {x_column}:', value=round(df[x_column].min(), 2), step=0.01)
            max_value = st.number_input(f'Enter the maximum value for {x_column}:', value=round(df[x_column].max(), 2), step=0.01)
            xlims[x_column] = (min_value, max_value)
            st.write(f'Selected range for {x_column}: {min_value} to {max_value}')

        st.write(f'You have selected: {y_column}')
        min_value = st.number_input(f'Enter the minimum value for {y_column}:', value=round(df[y_column].min(), 2), step=0.01)
        max_value = st.number_input(f'Enter the maximum value for {y_column}:', value=round(df[y_column].max(), 2), step=0.01)
        ylim = (max_value, min_value)
        st.write(f'Selected range for {y_column}: {min_value} to {max_value}')

        fig, axes = plot_well_logs(df_filtered, x_columns, y_column, log_scale_columns, xlims, ylim)
        st.pyplot(fig)

# Streamlit App
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('File Data Explorer')
st.text('This is a web app to allow exploration of las format file')

# Setup
u_file = st.file_uploader('Upload a las file format')

if u_file is not None:
    df = read_las_file(u_file)
    display_main_data(df)
    display_missingno_plots(df)

    df_filtered, selected_columns = select_columns(df)
    

    if not df_filtered.empty:
        df_filtered = handle_null_values(df_filtered, selected_columns)
        display_selected_data(df_filtered)
        display_boxplots(df_filtered)
        display_log_data_viz(df_filtered)







    