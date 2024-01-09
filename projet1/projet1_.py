import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from geopy.geocoders import Nominatim
import requests
import plotly.express as px
from scipy.stats import mode, skew
from io import StringIO
import streamlit as st
import plotly.express as px
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

def generate_summary_and_plot33(df, columns=None):
    if columns is None:
        summary_data = []
        for col in df.columns:
            numeric_values = pd.to_numeric(df[col], errors='coerce')

            mean = numeric_values.mean()
            median_value = numeric_values.median()
            mode_result = mode(numeric_values)
            mode_value = mode_result.mode
            mode_count = mode_result.count

            skewness = skew(numeric_values)
            if pd.isnull(skewness):
                symmetry = 'Asymétrie non identifiée'
            elif skewness > 0:
                symmetry = 'Asymétrie positive'
            elif skewness < 0:
                symmetry = 'Asymétrie négative'
            else:
                symmetry = 'Symétrique'

            summary_data.append([col, mean, median_value, mode_value, mode_count, skewness, symmetry])

        summary_df = pd.DataFrame(summary_data, columns=['Colonne', 'Moyenne', 'Médiane', 'Mode', 'Occurrences_Mode', 'Asymétrie', 'Nature_Asymetrie'])
        return summary_df
    else:
        # Calculate summary statistics for a specific attribute
        mean = df[columns].mean()
        median_value = df[columns].median()
        mode_result = mode(df[columns])
        mode_value = mode_result.mode
        mode_count = mode_result.count

        skewness = skew(df[columns])
        if pd.isnull(skewness):
            symmetry = 'Asymétrie non identifiée'
        elif skewness > 0:
            symmetry = 'Asymétrie positive'
        elif skewness < 0:
            symmetry = 'Asymétrie négative'
        else:
            symmetry = 'Symétrique'

        # Create summary DataFrame for the specific attribute
        summary_df = pd.DataFrame([[columns, mean, median_value, mode_value, mode_count, skewness, symmetry]],
                                  columns=['Colonne', 'Moyenne', 'Médiane', 'Mode', 'Occurrences_Mode', 'Asymétrie', 'Nature_Asymetrie'])

        # Plotting
        # fig = px.histogram(
        # df,
        # x=columns,
        # nbins=20,
        # marginal='rug',
        # title=f'Distribution of {columns}',
        # color_discrete_sequence=['darkblue']
        # )

        # # Update the layout for dark mode and customize other features
        # fig.update_layout(
        #     template='plotly_dark',
        #     xaxis_title=columns,
        #     yaxis_title='Count',
        #     showlegend=False  # To hide the legend
        # )
        # fig.show()
        # sns.set(style="darkgrid", rc={"grid.color": "red"})
        plt.figure(figsize=(8, 6))
        sns.histplot(df[columns], kde=True, color='darkblue', label='Distribution')
        plt.axvline(x=mean, color='red', linestyle='--', label='Moyenne')
        plt.axvline(x=median_value, color='green', linestyle='--', label='Médiane')
        if mode_value is not None:
            plt.axvline(x=mode_value, color='orange', linestyle='--', label='Mode')

        plt.title(f'Distribution of {columns}')
        plt.xlabel(columns)
        plt.legend()
        plt.show()

        return  plt ,summary_df
    
def parse_info(info_str):
    lines = info_str.split('\n')
    data = []
    for line in lines[3:]:
        if line.strip() == '':
            continue
        columns = line.split(maxsplit=3)
        data.append(columns)
    
    num_columns = len(data[0]) if data else 0

    columns = [f"Column_{i}" for i in range(num_columns)]
    
    # Create DataFrame
    df_info = pd.DataFrame(data, columns=columns)
    df_info = df_info.drop([0, 1])
    df_info = df_info.rename(columns={'Column_0': 'num', 'Column_1': 'Attribute', 'Column_2': 'Valeurs', 'Column_3': 'Type'})
    return df_info

def plot_histogram(dataframe, column):
    plt.figure(figsize=(8, 6))
    sns.histplot(dataframe[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
    return plt

def plot_scatter_with_correlation(dataframe, column1, column2):
    plt.figure(figsize=(8, 6))
    
    # Calculate the correlation between the two columns
    correlation = dataframe[column1].corr(dataframe[column2])
    
    # Create a scatter plot with a single point
    plt.scatter(dataframe[column1], dataframe[column2], label=f'Correlation: {correlation:.2f}', c='blue')
    
    plt.title(f'Scatter plot: {column1} vs {column2}')
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.legend()
    plt.show()
    return plt
    
def display_outliers(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)][column]
    print(f"Outliers in {column}: {len(outliers)}")
    print(outliers)

def plot_scatter(dataframe, column1, column2):
    plt.figure(figsize=(8, 6))
    plt.scatter(dataframe[column1], dataframe[column2])
    plt.title(f'Scatter plot: {column1} vs {column2}')
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show()
    
def isnull(df):
    null_counts = {}
    for att in df.columns:
        count = df[att].isnull().sum()
        null_counts[att] = count
    return null_counts
        
def boxplot_with_outliers(dataframe, column):
    plt.figure(figsize=(8, 6))
    sns.boxplot(dataframe[column])
    plt.title(f'Boxplot of {column}')
    plt.show()
    return plt
    
def generate_summary_and_plot(df, x=True, column=None):
    if x:
        summary_data = []

        for col in df.columns:
            mean = df[col].mean()
            median_value = df[col].median()
            mode_result = mode(df[col])
            mode_value = mode_result.mode
            mode_count = mode_result.count

            skewness = skew(df[col])
            if pd.isnull(skewness):
                symmetry = 'Asymétrie non identifiée'
            elif skewness > 0:
                symmetry = 'Asymétrie positive'
            elif skewness < 0:
                symmetry = 'Asymétrie négative'
            else:
                symmetry = 'Symétrique'

            summary_data.append([col, mean, median_value, mode_value, mode_count, skewness, symmetry])

        # Create summary DataFrame for all attributes
        summary_df = pd.DataFrame(summary_data, columns=['Colonne', 'Moyenne', 'Médiane', 'Mode', 'Occurrences_Mode', 'Asymétrie', 'Nature_Asymetrie'])

        return summary_df

    else:
        # Calculate summary statistics for a specific attribute
        mean = df[column].mean()
        median_value = df[column].median()
        mode_result = mode(df[column])
        mode_value = mode_result.mode
        mode_count = mode_result.count

        skewness = skew(df[column])
        if pd.isnull(skewness):
            symmetry = 'Asymétrie non identifiée'
        elif skewness > 0:
            symmetry = 'Asymétrie positive'
        elif skewness < 0:
            symmetry = 'Asymétrie négative'
        else:
            symmetry = 'Symétrique'

        # Create summary DataFrame for the specific attribute
        summary_df = pd.DataFrame([[column, mean, median_value, mode_value, mode_count, skewness, symmetry]],
                                  columns=['Colonne', 'Moyenne', 'Médiane', 'Mode', 'Occurrences_Mode', 'Asymétrie', 'Nature_Asymetrie'])

        # Plotting
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True, color='skyblue', label='Distribution')
        plt.axvline(x=mean, color='red', linestyle='--', label='Moyenne')
        plt.axvline(x=median_value, color='green', linestyle='--', label='Médiane')
        if mode_value is not None:
            plt.axvline(x=mode_value, color='orange', linestyle='--', label='Mode')

        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.legend()
        plt.show()

        return summary_df

def heatmap(df):
    # Replace non-numeric values with NaN
    df_numeric = df.apply(pd.to_numeric, errors='coerce')

    f, ax = plt.subplots(figsize=(20, 10))
    corr = df_numeric.corr("pearson")
    
    # Use bool instead of np.bool
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax, annot=True)
    
    return f

def replace_outliers_with_regression(df, max_iterations=10):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    df_copy = df.copy()
    for _ in range(max_iterations):
        outliers_exist = False
        for column_name in df_copy.select_dtypes(include='number').columns:
            column = df_copy[column_name]
            Q1 = column.quantile(0.25)
            Q3 = column.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (column < lower_bound) | (column > upper_bound)

            if outliers.any():
                outliers_exist = True

                # Separate data into non-outliers and outliers
                non_outliers = column[~outliers].values.reshape(-1, 1)
                outliers_indices = column[outliers].index.values.reshape(-1, 1)

                # Create features and target
                X_train, X_test, y_train, _ = train_test_split(
                    non_outliers, non_outliers, test_size=0.2, random_state=42
                )
                # Train a linear regression model
                model = LinearRegression()
                model.fit(X_train, y_train)
                # Predict values for outliers
                predicted_values = model.predict(outliers_indices)
                df_copy.loc[outliers, column_name] = predicted_values.flatten()

        if not outliers_exist:
            break  # Break the loop if no outliers were found in the last iteration

    return df_copy

def replace_missing(df, method='Mode'):
    for column in df.columns:
        if df[column].isnull().any():
            if df[column].dtype == 'object':
                if method != 'Mode':
                    print('Cette méthode ne peut pas être utilisée pour les attributs non numériques')
                else:
                    df[column] = df[column].fillna(df[column].mode()[0])
            else:
                if method == 'Mode':
                    df[column] = df[column].fillna(df[column].mode()[0])
                elif method == 'Median':
                    df[column] = df[column].fillna(df[column].median())
                elif method == 'Mean':
                    df[column] = df[column].fillna(df[column].mean())
                else:
                    print("Une erreur s'est produite : Méthode inconnue.")
    return df

def detect_redundant_col(df):
    red_col = []

    for i in range(len(df.columns) - 1):
        for j in range(i + 1, len(df.columns)):
            if df.iloc[:, i].equals(df.iloc[:, j]):
                red_col.append(df.columns[j])

    # Vertical check (columns)
    for col in df.columns:
        if df[col].nunique() == 1: 
            red_col.append(col)

    red_col = list(set(red_col))  
    return red_col

def remove_redundant_ligne(df):
    nombre_lignes_avant = df.shape[0]
    df=df.drop_duplicates()
    nombre_lignes_apret = df.shape[0]
    diff=nombre_lignes_avant -nombre_lignes_apret
    return df, diff

def remove_redundant_column(df):
    nombre_lignes_avant = df.shape[1]
    df=df.T.drop_duplicates().T
    nombre_lignes_apret = df.shape[1]
    diff=nombre_lignes_avant -nombre_lignes_apret
    return df, diff

def reduction_par_correlation(data, seuil):
    cor_mat = data.corr().abs()
    upper = cor_mat.where(np.triu(np.ones(cor_mat.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > seuil)]
    if "Fertility" in to_drop:
        to_drop.remove("Fertility")
    data = data.drop(columns=to_drop)
    return data, to_drop

def min_max(df):
    normalized_df = df.copy()

    for column in normalized_df.columns:
        if normalized_df[column].dtype in ['int64', 'float64']:
            min_val = normalized_df[column].min()
            max_val = normalized_df[column].max()
            normalized_df[column] = (normalized_df[column] - min_val) / (max_val - min_val)

    return normalized_df

def replace_outliers(df, method='median', max_iterations=10):
    df_copy = df.copy()

    for _ in range(max_iterations):
        outliers_exist = False 
        for column_name in df_copy.select_dtypes(include='number').columns:
            column = df_copy[column_name]
            Q1 = column.quantile(0.25)
            Q3 = column.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (column < lower_bound) | (column > upper_bound)
            if method == 'mode':
                replacement = column.mode()[0]
                if outliers.any():
                    outliers_exist = True
                    df_copy.loc[outliers, column_name] = replacement
            elif method == 'median':
                replacement = column.median()
                if outliers.any():
                    outliers_exist = True
                    df_copy.loc[outliers, column_name] = replacement
            elif method == 'mean':
                replacement = column.mean()
                if outliers.any():
                    outliers_exist = True
                    df_copy.loc[outliers, column_name] = replacement
            elif method == 'Q1':
                if outliers.any():
                    outliers_exist = True
                    df_copy.loc[outliers, column_name] = Q1
            elif method == 'Q3':
                if outliers.any():
                    outliers_exist = True
                    df_copy.loc[outliers, column_name] = Q3
            elif method == 'delete':
                if outliers.any():
                    outliers_exist = True
                    df_copy = df_copy[~outliers]      
        if not outliers_exist:
            break  # Break the lomin_maxop if no outliers were found in the last iteration
    
    return df_copy

def count_outliers(df):
    outliers_count = []
    for column_name in df.columns:
        if df[column_name].dtype=="object" or df[column_name].dtype=="string":
            outliers_count.append({'Attribute': column_name, 'Number of Outliers': 0})
        else:
            Q1 = df[column_name].quantile(0.25)
            Q3 = df[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)][column_name]
            outliers_count.append({'Attribute': column_name, 'Number of Outliers': len(outliers)})

    return pd.DataFrame(outliers_count)

def z_score(df):
    normalized_df = df.copy()

    for column in normalized_df.columns:
        if normalized_df[column].dtype in ['int64', 'float64']:
            mean = normalized_df[column].mean()
            std = normalized_df[column].std()
            normalized_df[column] = (normalized_df[column] - mean) / std

    return normalized_df

def Normalisation_log(df):
    normalized_df = df.copy()
    for column in normalized_df.columns:
        if normalized_df[column].dtype in ['int64', 'float64']:
            normalized_df[column] = np.log1p(normalized_df[column])

    return normalized_df

def racine_carre(df):
    normalized_df = df.copy()
    for column in normalized_df.columns:
        if normalized_df[column].dtype in ['int64', 'float64']:
            normalized_df[column] = np.sqrt(normalized_df[column])

    return normalized_df

def robuste_scalaire(df):
    from sklearn.preprocessing import RobustScaler
    robust_scaler = RobustScaler()
    return pd.DataFrame(robust_scaler.fit_transform(df), columns=df.columns)

def info_data(dataset):
    ##affichage du dataset
    if isinstance(dataset, str):
        df=pd.read_csv(dataset)
    else:
        df=dataset
    df['P'] = pd.to_numeric(df['P'], errors='coerce')
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    null = pd.DataFrame(list(isnull(df).items()), columns=['Attributes', 'Null Count'])
    return df,df.shape,df.head(),info_str,df.describe(),null

def Boite_Mous(X):
    DATA_TRIE = sorted(X)
    MEDIAN = np.percentile(DATA_TRIE, 75)
    Q1 = np.percentile(DATA_TRIE, 25)
    Q3 = np.percentile(DATA_TRIE, 75)
    iqr = Q3 - Q1
    Born_INF = Q1 - 1.5 * iqr
    BORN_SUPP = Q3 + 1.5 * iqr
    Boxplot = {
        "Born_Inf": Born_INF,
        "Q1": Q1,
        "Median": MEDIAN,
        "Q3": Q3,
        "Born_Supp": BORN_SUPP,
        "Outliers": [x for x in DATA_TRIE if x < Born_INF or x > BORN_SUPP]
    }
    return pd.DataFrame(Boxplot)

def analyse(df,col,col2):
    #mesure de tendace de toutes les attributs 

    summary_result = generate_summary_and_plot(df,False, col)
    print(summary_result)
    ##boite a moustache 
    boxplot_with_outliers(df,col)
    display_outliers(df,col)
    # Construire un histogramme et visualiser la distribution des données.
    plot_histogram(df,col)
    #Construire et afficher des diagrammes de dispersion des données et en déduire les corrélations entre les propriétés du sol
    plot_scatter_with_correlation(df, col, col2)

def update_dates(row, first_date_year, jan_count):
    months = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    
    if pd.isnull(row):
        return row, first_date_year, jan_count

    first_date_year = int(first_date_year)  # Convert to integer

    if "/" in row:
        date = datetime.strptime(row, '%m/%d/%Y')
    else:
        parts = row.split('-')
        if len(parts) == 3:
            date = datetime.strptime(row, '%Y-%d-%m')
        elif len(parts) == 2:
            day, month = parts
            month_number = months.get(month)
            if month_number is not None and month_number == '01':
                if jan_count == 10:
                    first_date_year += 1  # Increment the year by 1 after the third consecutive January
                    jan_count = 0
                else:
                    jan_count += 1  # Increment January count
            else:
                jan_count = 0  # Reset January count
            date = datetime.strptime(f"{day}-{month_number}-{first_date_year}", '%d-%m-%Y')

    return date.strftime('%Y-%m-%d'), str(first_date_year), jan_count

def update_data(data):
    # Read the data into DataFrame
    if isinstance(data, str):
        df2=pd.read_csv(data)
    else:
        return data
    data = df2.sort_values('time_period')
    
    first_row_date = data.loc[0, 'Start date']
    first_date_year = datetime.strptime(first_row_date, '%m/%d/%Y').year
    jan_count = 0

    for column in ['Start date', 'end date']:
        for index, row in data.iterrows():
            date = row[column]
            data.at[index, column], first_date_year, jan_count = update_dates(date, first_date_year, jan_count)

    # Resort the DataFrame without resetting the index before returning
    data['Start date'] = pd.to_datetime(data['Start date'])
    data['end date'] = pd.to_datetime(data['end date'])
    data = data.sort_index()
    
    return data

def animated_plot(df, zone, plot_type):
    data_zone = df[df['zcta'] == zone]

    data_zone['Start date'] = pd.to_datetime(data_zone['Start date'])

    if plot_type == 'weekly':
        resample_freq = 'W'
        title_suffix = 'hebdomadaire'
    elif plot_type == 'monthly':
        resample_freq = 'M'
        title_suffix = 'mensuel'
    elif plot_type == 'yearly':
        resample_freq = 'Y'
        title_suffix = 'annuel'
    else:
        raise ValueError("Invalid plot_type. Choose from 'weekly', 'monthly', or 'yearly'.")

    # Resample data for animation
    data_resampled = data_zone.resample(resample_freq, on='Start date').agg({
        'case count': 'sum',
        'positive tests': 'sum'
    })

    # Create animated plot using Plotly Express
    fig = px.line(
        data_resampled,
        x=data_resampled.index,
        y=['case count', 'positive tests'],
        labels={'value': 'Nombre', 'variable': 'Type'},
        title=f'Évolution {title_suffix} des tests et des cas COVID-19 pour la zone {zone}',
        animation_frame=data_resampled.index,
        category_orders={'variable': ['case count', 'positive tests']},
        template='plotly_dark'
    )

    # Show the plot
    st.plotly_chart(fig)

def line(df, zone, plot_type='weekly'):
    data_zone = df[df['zcta'] == zone]
    
    data_zone['Start date'] = pd.to_datetime(data_zone['Start date'])
    
    if plot_type == 'weekly':
        data_resampled = data_zone.resample('W', on='Start date').agg({
            'case count': 'sum',
            'positive tests': 'sum'
        })
        title_suffix = 'hebdomadaire'
    elif plot_type == 'monthly':
        data_resampled = data_zone.resample('M', on='Start date').agg({
            'case count': 'sum',
            'positive tests': 'sum'
        })
        title_suffix = 'mensuel'
    elif plot_type == 'yearly':
        data_resampled = data_zone.resample('Y', on='Start date').agg({
            'case count': 'sum',
            'positive tests': 'sum'
        })
        title_suffix = 'annuel'
    else:
        raise ValueError("Invalid plot_type. Choose from 'weekly', 'monthly', or 'yearly'.")

    # Create a single plot
    plt.figure(figsize=(12, 6))

    # Plot data
    plt.plot(data_resampled.index, data_resampled['case count'], label='Tests COVID-19 Nbr cas')
    plt.plot(data_resampled.index, data_resampled['positive tests'], label='Tests COVID-19 pos test')
    
    # Set title and labels
    plt.title(f'Évolution {title_suffix} des tests et des cas COVID-19 pour la zone {zone}')
    plt.xlabel('Date')
    plt.ylabel('Nombre')
    plt.legend()

    # Show the plot
    return plt

def plot_positive_cases_distribution(data):
    # Filter positive cases
    positive_cases = data[data['positive tests'] > 0]
 
    # Grouper les données par zone et par année, en comptant les cas positifs
    grouped_data = positive_cases.groupby([pd.Grouper(key='Start date', freq='Y'), 'zcta'])['positive tests'].sum().unstack()

    # Créer un graphique à barres empilées pour chaque zone et chaque année
    grouped_data.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Distribution des cas COVID positifs par zone et par année')
    plt.xlabel('Annee ')
    plt.ylabel('Nombre de cas positifs')
    plt.legend(title='zone')
    return plt

def ZCTA_TO_ZONE(DATA):
    unique_values = DATA['zcta'].unique()
    key = "hAMnp5MY9VrrrdjOmHX1BmVYoJpnp02c"
    locations = []
    for zip_code in unique_values:
        url = f'https://www.mapquestapi.com/geocoding/v1/address?postalCode={zip_code}&key={key}'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get("results", [])
            if data:
                for result in data:
                    location = result.get("locations", [])[0]
                    city = location.get("adminArea5", "")
                    country = location.get("adminArea1", "")                 
                    specific_info = {
                        "Zip Code": zip_code,
                        "City": city,
                        "Country": country
                    }
                    locations.append(specific_info)
            else:
                st.warning("No location")
        else:
            st.warning(f"Failed to make the geocoding request for ZIP code {zip_code}. Status Code: {response.status_code}")
    mapping_dict = {item['Zip Code']: str(item['Zip Code'])+","+item['City']+","+ item['Country'] for item in locations}
    DATA['zcta'] = DATA['zcta'].map(mapping_dict)
    
    return DATA

def animated_Plot(data):
    # Sort 'Start date' within each 'zcta' group
    # data['Start date'] = pd.to_datetime(data['Start date'])
    # data = data.groupby('zcta').apply(lambda x: x.sort_values('Start date')).reset_index(drop=True)
    #la valeur pour grouper doit etre la valeur pour trie
    grouped_df = data.groupby('time_period')
    valid_groups = grouped_df.filter(lambda x: len(x) == 7)
    data = data[data['time_period'].isin(valid_groups['time_period'])]
    data = data.sort_values(by='time_period')
    fig = px.bar(data, x='zcta', y='case count', color='zcta', animation_frame='time_period',
                 labels={'zcta': 'ZCTA', 'case count': 'Case Count'},
                 title='Case Count Distribution Over Years',
                 template='plotly_dark')
    fig.update_layout(yaxis=dict(range=[0, 800]))

    # Specify the order of animation frames
    fig.update_layout(
        xaxis=dict(categoryorder='array', categoryarray=data['zcta'].unique())
    )
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]=2000
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"]=1000
   
    return fig

def animated_Plot2(data):
    grouped_df = data.groupby('time_period')
    valid_groups = grouped_df.filter(lambda x: len(x) == 7)
    data = data[data['time_period'].isin(valid_groups['time_period'])]
    data = data.sort_values(by='time_period')
    grouped_by_zcta = data.groupby('zcta')

    data = grouped_by_zcta.apply(lambda group: group.sort_values('time_period'))
    
    st.write(data)
    color_discrete_map = {
        'case count': 'blue',
        'positive tests': 'green',
        'test count': 'red'
    } 
    fig = px.bar(data, 
                 x='zcta', 
                 y=['case count', 'positive tests', 'test count'],
                 color='variable',  # 'variable' will be the name of the new column created by px.bar
                 animation_frame='time_period',
                 labels={'zcta': 'ZCTA', 'value': 'Count', 'variable': 'Attribute'},
                 title='Case Count Distribution Over Years',
                 template='plotly_dark',
                 color_discrete_map=color_discrete_map)
    
    fig.update_layout(yaxis=dict(range=[0, 2000]))
    
    # Specify the order of categories on the x-axis
    fig.update_layout(
        xaxis=dict(categoryorder='array', categoryarray=data['zcta'].unique())
    )
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]=2000
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"]=1000
   
    return fig

def plot_distribution_by_zone(data):
    df_case_count = data.groupby('zcta').agg({'case count': 'sum'}).reset_index()
    df_positive_tests = data.groupby('zcta').agg({'positive tests': 'sum'}).reset_index()

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot for case count
    sns.barplot(data=df_case_count, x='zcta', y='case count', palette='viridis', ax=axes[0])
    axes[0].set_title('Distribution du nombre total de cas confirmés par zone')
    axes[0].set_xlabel('Zone')
    axes[0].set_ylabel('Nombre total de cas confirmés')
    axes[0].tick_params(axis='x', rotation=45)

    # Plot for positive tests
    sns.barplot(data=df_positive_tests, x='zcta', y='positive tests', palette='viridis', ax=axes[1])
    axes[1].set_title('Distribution du nombre total de tests positifs par zone')
    axes[1].set_xlabel('Zone')
    axes[1].set_ylabel('Nombre total de tests positifs')
    axes[1].tick_params(axis='x', rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Return the plt object
    return plt

def plot_population_test_relation(data):
    plt.figure(figsize=(8, 6))
    plt.scatter(data['population'], data['test count'])
    plt.title('Relation entre la population et le nombre de tests effectués')
    plt.xlabel('Population')
    plt.ylabel('Nombre de tests effectués')
    plt.grid(True)
    return plt

def plot_top_zones_impacted(data, column, top_n=5):
    # Group data by zone and calculate the sum of the specified column for each zone
    total_per_zone = data.groupby('zcta')[column].sum()
    
    # Get the top N zones by the total of the specified column
    top_zones = total_per_zone.nlargest(top_n)

    # Plotting the top N impacted zones
    plt.figure(figsize=(10, 6))
    top_zones.plot(kind='bar', color='skyblue' if column == 'case count' else 'salmon')
    plt.title(f'Top {top_n} Zones Most Impacted by Total Number of {column.replace(" ", "")}')
    plt.xlabel('Zone (zcta)')
    plt.ylabel(f'Total {column}')
    return plt

def coeff_asy(dict):
    mode = max(dict["mode"]) if type(dict["mode"]) == list  else dict["mode"]
    if round(dict["mean"]) == round(dict["median"]) == np.round(mode):
        return "Distribution symetrique"
    elif dict["mean"] < dict["median"] < mode:
        return "Distribution d'asymetrie negative"
    elif dict["mean"] > dict["median"] > mode:
        return "Distribution d'asymetrie positive"
    else:
        return "Distribution non identifie"

def generate_summary_and_plot2(df, columns=None):
    dict={}
    summary_data = []
    if columns is None:
        # Calculate summary statistics for all attributes
        for col in df.columns:
            # Convert to numeric, replacing non-numeric values with NaN
            numeric_values = pd.to_numeric(df[col], errors='coerce')

            dict["mean"] = numeric_values.mean()
            dict["median"] = numeric_values.median()
            mode_result = mode(numeric_values)
            dict['mode'] = mode_result.mode
            mode_count = mode_result.count

            dict["symetrie"] = coeff_asy(dict)
            

            summary_data.append([col, dict["mean"], dict["median"], dict['mode'], mode_count,  dict["symetrie"]])

        summary_df = pd.DataFrame(summary_data, columns=['Colonne', 'Moyenne', 'Médiane', 'Mode', 'Occurrences_Mode',  'Nature_Asymetrie'])
        return summary_df
    else:
        # Calculate summary statistics for a specific attribute
        dict["mean"] = df[columns].mean()
        dict["median"] = df[columns].median()
        mode_result = mode(df[columns])
        dict['mode'] = mode_result.mode
        mode_count = mode_result.count
        dict["symetrie"] = coeff_asy(dict)
        summary_data.append([columns, dict["mean"], dict["median"], dict['mode'], mode_count,  dict["symetrie"]])
        summary_df = pd.DataFrame(summary_data, columns=['Colonne', 'Moyenne', 'Médiane', 'Mode', 'Occurrences_Mode',  'Nature_Asymetrie'])
        plt.figure(figsize=(8, 6))
        sns.histplot(df[columns], kde=True, color='darkblue', label='Distribution')
        plt.axvline(x=dict["mean"], color='red', linestyle='--', label='Moyenne')
        plt.axvline(x=dict["median"], color='green', linestyle='--', label='Médiane')
        if dict['mode'] is not None:
            plt.axvline(x=dict['mode'], color='orange', linestyle='--', label='Mode')
        plt.title(f'Distribution of {columns}')
        plt.xlabel(columns)
        plt.legend()
        plt.show()
        return  plt ,summary_df
    
def reduction_par_faible_variance(data, seuil):
    features = data.drop(columns=["Fertility"])

    variances = features.var()
    low_variance_features = variances[variances < seuil].index

    reduced_data = data.drop(columns=low_variance_features)

    return reduced_data, low_variance_features   