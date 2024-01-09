import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import mode, skew
from io import StringIO
from datetime import datetime

def generate_summary_and_plot2(df, columns=None):
    if columns is None:
        # Calculate summary statistics for all attributes
        summary_data = []

        for col in df.columns:
            # Convert to numeric, replacing non-numeric values with NaN
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

        # Create summary DataFrame for all attributes
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
        plt.figure(figsize=(8, 6))
        sns.histplot(df[columns], kde=True, color='skyblue', label='Distribution')
        plt.axvline(x=mean, color='red', linestyle='--', label='Moyenne')
        plt.axvline(x=median_value, color='green', linestyle='--', label='Médiane')
        if mode_value is not None:
            plt.axvline(x=mode_value, color='orange', linestyle='--', label='Mode')

        plt.title(f'Distribution of {columns}')
        plt.xlabel(columns)
        plt.legend()
        plt.show()

        return  plt ,summary_df # Return both the summary DataFrame and the plot
def parse_info(info_str):
    lines = info_str.split('\n')
    data = []
    for line in lines[3:]:
        if line.strip() == '':
            continue
        columns = line.split(maxsplit=3)
        data.append(columns)
    
    # Determine the number of columns dynamically
    num_columns = len(data[0]) if data else 0
    
    # Create column names
    columns = [f"Column_{i}" for i in range(num_columns)]
    
    # Create DataFrame
    df_info = pd.DataFrame(data, columns=columns)
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
        # Calculate summary statistics for all attributes
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

    # Horizontal check (columns)
    for i in range(len(df.columns) - 1):
        for j in range(i + 1, len(df.columns)):
            if df.iloc[:, i].equals(df.iloc[:, j]):
                red_col.append(df.columns[j])

    # Vertical check (columns)
    for col in df.columns:
        if df[col].nunique() == 1:  # Utiliser nunique pour compter les valeurs uniques
            red_col.append(col)

    red_col = list(set(red_col))  # Supprimer les doublons détectés
    return red_col

def remove_redundant_columns(df):
    red_col = detect_redundant_col(df)
    #col
    df_cleaned = df.drop(columns=red_col)
    #les lignes
    df_cleaned.drop_duplicates(inplace=True)
    return df_cleaned
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
        outliers_exist = False  # Flag to check if outliers are still present

        for column_name in df_copy.select_dtypes(include='number').columns:
            column = df_copy[column_name]
            Q1 = column.quantile(0.25)
            Q3 = column.quantile(0.75)

            if method == 'mode':
                replacement = column.mode()[0]
            elif method == 'median':
                replacement = column.median()
            elif method == 'mean':
                replacement = column.mean()
            elif method == 'iqr':
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (column < lower_bound) | (column > upper_bound)
                if outliers.any():
                    outliers_exist = True
                    df_copy[column_name] = df_copy[column_name].replace(column[outliers], column.median())
            elif method == 'delete':
                outliers = (column < Q1 - 1.5 * (Q3 - Q1)) | (column > Q3 + 1.5 * (Q3 - Q1))
                if outliers.any():
                    outliers_exist = True
                    df_copy = df_copy[~outliers]
            else:
                print(f"Invalid method for column {column_name}. Please choose from 'mode', 'median', 'mean', 'iqr', or 'delete'.")
                continue

        if not outliers_exist:
            break  # Break the loop if no outliers were found in the last iteration

    return df_copy
def count_outliers(df):
    outliers_count = []

    for column_name in df.columns:
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
def info_data(dataset):
    ##affichage du dataset
    df=pd.read_csv(dataset)
    df['P'] = pd.to_numeric(df['P'], errors='coerce')
    # print(df.head())
    # ##le nombre de ligne et colonnes
    # print(df.shape)
    # df.info()
    # df.describe()
    # isnull(df)
    # Capture the output of df.info() into a string
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    return df,df.shape,df.head(),info_str,df.describe(),isnull(df)

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


# Function to update dates
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
    df2 = pd.read_csv(data)
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
import pandas as pd
import matplotlib.pyplot as plt

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


def plot_distribution_by_zone(data):
    # Aggregate data by zone
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


