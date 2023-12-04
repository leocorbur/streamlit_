# Import the required Libraries
import streamlit as st
import lasio as ls
import pandas as pd
import io
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

def read_las_file(u_file):
    las_file_contents = u_file.read()
    las_file_contents_str = las_file_contents.decode("utf-8")
    las_file_buffer = io.StringIO(las_file_contents_str)
    las = ls.read(las_file_buffer)
    df = las.df()
    df.reset_index(inplace=True)
    df.index = df.index + 1
    return df

# Function for data exploration
def explore_data(df):
    st.header('Main Data')

    # Display first and last rows
    st.subheader('First Rows')
    st.write(df.head())
    st.subheader('Last Rows')
    st.write(df.tail())
    st.subheader('Statistics')
    st.write(df.describe())
    dim = st.radio('Data dimension:', ('Rows', 'Columns'), horizontal=True)
    if dim == 'Rows':
        st.write('Number of rows: ', df.shape[0])
    else:
        st.write('Number of columns:', df.shape[1])
    
    # Missingno
    st.subheader('A nullity matrix')
    msno.matrix(df)
    st.pyplot()

    # Columns selection
    st.subheader('Column selection')
    #selected_columns = st.multiselect('Select at least 03 columns you want to interact with, and include a depth column for log visualization purposes:', ['All columns'] + list(df.columns))
    selected_columns = st.multiselect('Select at least 03 columns you want to interact with, and include a depth column for log visualization purposes:', 
                                      list(df.columns))
    
    # Verify if "All columns" is in selected columns
    #if 'All columns' in selected_columns:
        #selected_columns = df.columns 

    # Filter the original DataFrame based on the selected columns
    df_filtered = df[selected_columns]

    if selected_columns:
        # Remove rows with null values or impute values
        st.subheader('What do you want to do with missing values?')
        operation_choice = st.radio('Choose operation:', ['Remove rows containing missing values', 'Impute missing values'], index=1)

        if operation_choice == 'Remove rows containing missing values':
            df_filtered = df_filtered.dropna()
            
        else:
            # Imputation of values
            imputation_method = st.selectbox('Select global imputation method:', ['Mean', 'Median', 'Specific Value', 'Zero'],
                                         help="Choose a method to fill missing values for all selected columns.")

            if imputation_method == 'Zero':
                df_filtered[selected_columns] = df_filtered[selected_columns].fillna(0)
            else:
                for column in selected_columns:
                    if imputation_method == 'Mean':
                        imputer = SimpleImputer(strategy='mean')
                        df_filtered[column] = imputer.fit_transform(df_filtered[[column]])
                    elif imputation_method == 'Median':
                        imputer = SimpleImputer(strategy='median')
                        df_filtered[column] = imputer.fit_transform(df_filtered[[column]])
                    elif imputation_method == 'Specific Value':
                        specific_value = st.number_input(f'Enter the specific value for {column}:')
                        df_filtered[column] = df_filtered[column].fillna(specific_value)


        # Display the filtered DataFrame
        st.title('Selected Data')
        st.write(df_filtered.head())

    return df_filtered, selected_columns

# Function for boxplot
def boxplot(df_filtered):
    # Configuration for the red circle in the boxplot
    red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')

    # Create the Streamlit application
    st.title('Boxplots')

    # Allow the user to select logarithmic columns
    log_columns = st.multiselect('Select logarithmic columns:', df_filtered.columns)

    # Create box plots 
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

    # Get the color palette
    colors = sns.color_palette(color_palette, n_colors=len(x_columns))

    for i, column in enumerate(x_columns):
        ax = axes [i]

        ax.plot(df[column], y_column, data=df, color=colors[i], linewidth=0.5) 
        ax.set_xlim(xlims.get(column, (df[column].min(), df[column].max())))

        if column in log_scale_columns:
            ax.set_xscale('log') 

        ax.set_ylim(ylim)
        ax.grid()

        # Move x-axis labels to the top
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


# Config Setup
st.set_page_config(page_title="LAS Explorer", page_icon="📊", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Add a title and intro text
st.title('LAS File Data Explorer')
st.text('Welcome to the LAS File Data Explorer web app. Upload a LAS file to get started.')

# Sidebar setup
st.sidebar.title('Instructions')
st.sidebar.write('1. Upload a LAS file.')
st.sidebar.write('2. Choose an option from the sidebar navigation.')
u_file = st.sidebar.file_uploader('Upload a las file format')

# Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select what you want to display:', ['Explore Data', 'Box Plot', 'Log Data Viz'])

# Check if file has been uploaded
if u_file is not None:
    st.success('File uploaded successfully!')
    df = read_las_file(u_file)

    # Capture the filtered DataFrame from the explore_data function
    df_filtered, selected_columns = explore_data(df)
else:
    st.warning('Please upload a LAS file to begin.')

# Navigation options
if options == 'Explore Data':
    pass  # No need to explicitly call the function as it's already invoked within the condition above
elif options == 'Box Plot':
    boxplot(df_filtered)
elif options == 'Log Data Viz':
    display_log_data_viz(df_filtered)

# Footer with additional information or links
st.sidebar.markdown('---')
st.sidebar.subheader('Additional Information')
st.sidebar.write('For more information, please contact me at leocorbur@gmail.com or via ' 
                 '[LinkedIn](https://www.linkedin.com/in/leonelcortez/). ' 
                 'Also, I invite you see my lastest projest on [GitHub](https://github.com/leocorbur).')