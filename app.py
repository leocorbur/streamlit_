# Import the required libraries
import streamlit as st
import lasio as ls
import pandas as pd
import io
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

st.set_option('deprecation.showPyplotGlobalUse', False)


# Add a title and intro text
st.title('File Data Explorer')
st.text('This is a web app to allow exploration of las format file')

# Setup
u_file = st.file_uploader('Upload a las file format')


# Check if file has been uploaded

if u_file is not None:
    # Create a string-like object
    las_file_contents = u_file.read()
    las_file_contents_str = las_file_contents.decode("utf-8")
    las_file_buffer = io.StringIO(las_file_contents_str)

    # Read the LAS file
    las = ls.read(las_file_buffer)
    df = las.df()
    df.reset_index(inplace=True)
    df.index = df.index + 1


if u_file:
        st.header('Main Data')

        # Display first and last rows
        st.subheader('First rows')
        st.write(df.head())
        st.subheader('Last rows')
        st.write(df.tail())

        st.subheader('Statistics')
        st.write(df.describe())

        # Numbers of rows and columns
        dim = st.radio('Data dimension:', ('Rows', 'Columns'), horizontal=True)
        if dim == 'Rows':
            st.write('Number of rows: ', df.shape[0])
        else:
            st.write('Number of columns:', df.shape[1])

        # Missingno
        st.subheader('A nullity matrix')
        msno.matrix(df)
        st.pyplot()

        st.subheader('Nullity by column')
        msno.bar(df)
        st.pyplot()

        # Columns selection
        st.subheader('Column selection')
        selected_columns = st.multiselect('Select the columns you want to interact with:', 
                                                ['All columns'] + list(df.columns))
        
        # Verify if "All columns" is in selected columns
        if 'All columns' in selected_columns:
            selected_columns = df.columns 

        # Filter the original DataFrame based on the selected columns
        df_filtered = df[selected_columns]

        # Interactions df_filtered
        if not df_filtered.empty:
            st.header('First rows of selected columns')
            st.write(df_filtered.head())

            # Remove rows with null values in any column
            delete_nulls = st.checkbox('Delete rows with null values in any column')
            if delete_nulls:
                df_filtered = df_filtered.dropna()
            else:
                # Imputation of values
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

            # Display the filtered DataFrame
            st.write(df_filtered.head())

            # Dimension selection
            dimension = st.radio('Dimension of filtered df:', ('Rows', 'Columns'), horizontal=True)
            if dimension == 'Rows':
                st.write('Number of rows: ', df_filtered.shape[0])
            else:
                st.write('Number of columns:', df_filtered.shape[1])

            
            # Boxplot
            
            red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')
            st.title('Boxplots')

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


else:
    st.text('To begin please upload a las format file')

