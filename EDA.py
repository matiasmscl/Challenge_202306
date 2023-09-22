from scipy.cluster import hierarchy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.manifold import TSNE
import umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def convert_to_numeric(x):
    if isinstance(x, int):
        return x
    elif isinstance(x, float):
        return int(x) if x.is_integer() else x
    elif isinstance(x, str):
        try:
            return int(x)
        except ValueError:
            try:
                float_x = float(x)
                return int(float_x) if float_x.is_integer() else float_x
            except ValueError:
                return x
    else:
        return x

def eda_plots(df):
    for column in df.columns:
        column_data = df[column].dropna()
        if np.issubdtype(column_data.dtype, np.number):
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.hist(column_data, bins=10)
            plt.title(f'Histogram for {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            value_counts = column_data.value_counts()
            plt.subplot(1, 2, 2)
            plt.bar(value_counts.index, value_counts.values)
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.title(f'Value Counts of {column}')
            plt.show()
            plt.figure()
            if column_data.dtype == 'float' or column_data.dtype == 'int':
                plt.boxplot(column_data)
                plt.title(f'Box Plot for {column}')
                plt.ylabel(column)
                plt.tight_layout()
            else:
                string_lengths = column_data.astype(str).apply(len)
                plt.hist(string_lengths, bins=20)
                plt.xlabel('String Length')
                plt.ylabel('Frequency')
                plt.title(f'String Length Distribution of {column}')
            plt.show()
        elif len(column_data.unique()) > 1:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.countplot(data=df, x=column)
            plt.title(f'Count Plot for {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.subplot(1, 2, 2)
            value_counts = column_data.value_counts()
            labels = value_counts.index
            plt.pie(value_counts, labels=labels, autopct='%1.1f%%')
            plt.title(f'Pie Chart for {column}')
            plt.axis('equal')
            plt.tight_layout()
            plt.show()



def generate_CORR(df):
    numerical_columns = df.select_dtypes(include=np.number).columns
    if len(numerical_columns) <= 1:
        print("No enough columns found in the DataFrame.")
        return
    # Calculate correlation matrix
    correlation_matrix = df[numerical_columns].corr()
    # Display correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    
def generate_PCA(df, column_categorica = 'RandomVar'):
    df=df.drop_duplicates()
    numerical_columns = df.select_dtypes(include=np.number).columns
    if len(numerical_columns) <= 1:
        print("No enough columns found in the DataFrame.")
        return
    # Calculate correlation matrix
    correlation_matrix = df[numerical_columns].corr()
    # Check if any numerical column has missing values
    if df[numerical_columns].isnull().any().any():
        print("Numerical columns contain missing values. Please handle missing values before performing PCA.")
        return
    correlaciones=0
    cuentas=0
    for j in range(len(correlation_matrix)-1):
        for k in range(j+1, len(correlation_matrix)):
            cuentas=cuentas+1
            if abs(correlation_matrix.iloc[j, k]) == 1:
                correlaciones=correlaciones+1
    if correlaciones>= cuentas:
        print("Todas Columnas Correlacionadas.")
        return
    df = df.loc[df[numerical_columns].dropna().index]
    df_copy = df.copy()
    df_copy = df_copy.drop(columns=[column_categorica])
    numerical_columns = df_copy.select_dtypes(include=np.number).columns
    print(df[column_categorica].unique())
    for n_components in range(2, min(5, len(numerical_columns))):
        print('----------')
        print(n_components)
        if n_components> len(df_copy) or n_components> len(numerical_columns):
            print("Pocas Columnas o filas.")
            break
        # Perform PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(df_copy[numerical_columns])
        # Create PCA plot
        plt.figure(figsize=(8, 6))
        if n_components == 2:
            plt.figure(figsize=(8, 6))
            for category in df[column_categorica].unique():
                mask = df[column_categorica] == category
                if category == 1:
                    marker = '*'
                elif category == 2:
                    marker = 'x'
                else:
                    marker = 'o'
                plt.scatter(principal_components[mask, 0], principal_components[mask, 1], marker=marker, label=f'Category {category}')
            plt.title('PCA Plot (2 Components)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend()
            plt.show()
        elif n_components == 3:
            plt.figure(figsize=(8, 6))
            for category in df[column_categorica].unique():
                mask = df[column_categorica] == category
                if category == 1:
                    marker = '*'
                elif category == 2:
                    marker = 'x'
                else:
                    marker = 'o'
                plt.scatter(principal_components[mask, 0], principal_components[mask, 1], c=principal_components[mask, 2],
                            marker=marker, label=f'Category {category}')
            plt.title('PCA Plot (3 Components)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.colorbar(label='Principal Component 3')
            plt.legend()
            plt.show()
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            for category in df[column_categorica].unique():
                mask = df[column_categorica] == category
                if category == 1:
                    marker = '*'
                elif category == 2:
                    marker = 'x'
                else:
                    marker = 'o'
                ax.scatter3D(principal_components[mask, 0], principal_components[mask, 1], principal_components[mask, 2],
                             marker=marker, label=f'Category {category}')
            ax.set_title('PCA Plot (3 Components)')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            ax.legend()
            plt.show()
        elif n_components == 4:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for category in df[column_categorica].unique():
                mask = df[column_categorica] == category
                if category == 1:
                    marker = '*'
                elif category == 2:
                    marker = 'x'
                else:
                    marker = 'o'
                scatter = ax.scatter3D(
                    principal_components[mask, 0], principal_components[mask, 1], principal_components[mask, 2],
                    c=principal_components[mask, 3], cmap='coolwarm', marker=marker, label=f'Category {category}')
            ax.set_title('PCA Plot (4 Components)')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            cbar = fig.colorbar(scatter, ax=ax, label='Principal Component 4')
            ax.legend()
            plt.show()


def generate_tSNE(df, column_categorica='RandomVar'):
    df = df.drop_duplicates()
    numerical_columns = df.select_dtypes(include=np.number).columns
    if len(numerical_columns) <= 1:
        print("Not enough numerical columns found in the DataFrame.")
        return
    # Check if any numerical column has missing values
    if df[numerical_columns].isnull().any().any():
        print("Numerical columns contain missing values. Please handle missing values before performing t-SNE.")
        return
    df_copy = df.copy()
    df_copy = df_copy.drop(columns=[column_categorica])
    numerical_columns = df_copy.select_dtypes(include=np.number).columns
    if len(numerical_columns) < 2:
        print("t-SNE requires at least two numerical columns.")
        return
    if len(df_copy) < 2:
        print("t-SNE requires at least two data points.")
        return
    try:
        for n_components in range(2, min(4, len(numerical_columns))):
            if n_components > len(df_copy) or n_components > len(numerical_columns):
                print("Not enough data points or numerical columns for t-SNE with {} components.".format(n_components))
                break
            # Perform t-SNE
            tsne = TSNE(n_components=n_components, init='random', learning_rate=200.0)
            tsne_components = tsne.fit_transform(df_copy[numerical_columns])
            # Create t-SNE plot
            if n_components == 2:
                plt.figure(figsize=(8, 6))
                for category in df[column_categorica].unique():
                    mask = df[column_categorica] == category
                    if category == 1:
                        marker = '*'
                    elif category == 2:
                        marker = 'x'
                    else:
                        marker = 'o'
                    plt.scatter(tsne_components[mask, 0], tsne_components[mask, 1], marker=marker, label=f'Category {category}')
                plt.title('t-SNE Plot (2 Components)')
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.legend()
                plt.show()
            elif n_components == 3:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                for category in df[column_categorica].unique():
                    mask = df[column_categorica] == category
                    if category == 1:
                        marker = '*'
                    elif category == 2:
                        marker = 'x'
                    else:
                        marker = 'o'
                    ax.scatter3D(tsne_components[mask, 0], tsne_components[mask, 1], tsne_components[mask, 2],
                                 marker=marker, label=f'Category {category}')
                ax.set_title('t-SNE Plot (3 Components)')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.set_zlabel('Component 3')
                ax.legend()
                plt.show()
    except ValueError:
        print(ValueError)

        
def generate_UMAP(df, column_categorica='RandomVar'):
    df = df.drop_duplicates()
    numerical_columns = df.select_dtypes(include=np.number).columns
    if len(numerical_columns) <= 1:
        print("Not enough numerical columns found in the DataFrame.")
        return
    # Check if any numerical column has missing values
    if df[numerical_columns].isnull().any().any():
        print("Numerical columns contain missing values. Please handle missing values before performing UMAP.")
        return
    df_copy = df.copy()
    df_copy = df_copy.drop(columns=[column_categorica])
    numerical_columns = df_copy.select_dtypes(include=np.number).columns
    if len(numerical_columns) < 2:
        print("UMAP requires at least two numerical columns.")
        return
    if len(df_copy) < 2:
        print("UMAP requires at least two data points.")
        return
    try:
        for n_components in range(2, min(4, len(numerical_columns))):
            if n_components > len(df_copy) or n_components > len(numerical_columns):
                print("Not enough data points or numerical columns for UMAP with {} components.".format(n_components))
                break
            # Perform UMAP
            reducer = umap.UMAP(n_components=n_components)
            umap_components = reducer.fit_transform(df_copy[numerical_columns])
            # Create UMAP plot
            if n_components == 2:
                plt.figure(figsize=(8, 6))
                for category in df[column_categorica].unique():
                    mask = df[column_categorica] == category
                    if category == 1:
                        marker = '*'
                    elif category == 2:
                        marker = 'x'
                    else:
                        marker = 'o'
                    plt.scatter(umap_components[mask, 0], umap_components[mask, 1], marker=marker, label=f'Category {category}')
                plt.title('UMAP Plot (2 Components)')
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.legend()
                plt.show()
            elif n_components == 3:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                for category in df[column_categorica].unique():
                    mask = df[column_categorica] == category
                    if category == 1:
                        marker = '*'
                    elif category == 2:
                        marker = 'x'
                    else:
                        marker = 'o'
                    ax.scatter3D(umap_components[mask, 0], umap_components[mask, 1], umap_components[mask, 2],
                                 marker=marker, label=f'Category {category}')
                ax.set_title('UMAP Plot (3 Components)')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.set_zlabel('Component 3')
                ax.legend()
                plt.show()
    except ValueError as e:
        print(e)



def calculate_statistics(dataframe):
    statistics = {
        'Column': [],
        'Count': [],
        'Mean': [],
        'Variance': [],
        'Standard Deviation': [],
        'Minimum': [],
        '25th Percentile': [],
        '50th Percentile (Median)': [],
        '75th Percentile': [],
        'Maximum': [],
        'Mode': [],
        'Elementos Unicos': [],
        'Skewness': [],
        'Kurtosis': [],
        'Range': [],
        'IQR(Rango Intercuartílico)': [],
        'Median Absolute Deviation': [],
        'Outliers': []
    }
    numerical_columns = dataframe.select_dtypes(include=[np.number, bool])
    for column in numerical_columns:
        column_data = dataframe[column]
        statistics['Column'].append(column)
        statistics['Count'].append(column_data.count())
        statistics['Mean'].append(column_data.mean())
        statistics['Variance'].append(column_data.var())
        statistics['Standard Deviation'].append(column_data.std())
        statistics['Minimum'].append(column_data.min())
        statistics['25th Percentile'].append(column_data.quantile(0.25))
        statistics['50th Percentile (Median)'].append(column_data.median())
        statistics['75th Percentile'].append(column_data.quantile(0.75))
        statistics['Maximum'].append(column_data.max())
        statistics['Mode'].append(column_data.mode().values[0] if not column_data.empty and column_data.mode().values.size > 0 else np.nan)
        statistics['Elementos Unicos'].append(column_data.nunique())
        statistics['Skewness'].append(column_data.skew())
        statistics['Kurtosis'].append(column_data.kurtosis())
        statistics['Range'].append('Max: {} Min: {}'.format(column_data.max(), column_data.min()))
        statistics['IQR(Rango Intercuartílico)'].append(column_data.quantile(0.75) - column_data.quantile(0.25))
        statistics['Median Absolute Deviation'].append((column_data - column_data.mean()).abs().mean())
        outliers = len(column_data[
            (column_data < column_data.quantile(0.25) - 1.5 * column_data.quantile(0.75)) |
            (column_data > column_data.quantile(0.75) + 1.5 * column_data.quantile(0.75))
        ])
        statistics['Outliers'].append(outliers)

    statistics_df = pd.DataFrame(statistics)
    return statistics_df

def identify_outliers(df, threshold=99):
    numerical_columns = df.select_dtypes(include=np.number).columns
    for column in numerical_columns:
        column_data = df[column].dropna()
        if not column_data.empty:
            # Calculate the percentiles
            lower_percentile = 50 - threshold / 2
            upper_percentile = 50 + threshold / 2
            # Calculate the lower and upper bounds for outliers
            lower_bound = column_data.quantile(lower_percentile / 100)
            upper_bound = column_data.quantile(upper_percentile / 100)
            # Identify and flag potential outliers
            outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
            outliers['Outlier_Flag'] = True
            # Print the identified outliers
            if outliers.empty:
                print("No outliers found in the column.")
            else:
                print(f"Identified {len(outliers)} potential outlier(s) in the column: {column}")
                print(outliers)

def missing_data_visualization(df):
    missing_data = df.isnull()
    missing_columns = missing_data.sum()
    missing_columns = missing_columns[missing_columns > 0]
    if len(missing_columns) > 0:
        plt.figure(figsize=(10, 6))
        sns.heatmap(missing_data.transpose(), cmap='binary', cbar=False)
        plt.title('Missing Data Visualization')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()
    else:
        print("No missing data found in the dataframe.")

def clean_strings(df):
    df_cleaned = df.copy()
    df_cleaned=df_cleaned.drop_duplicates()
    to_drop=[]
    for column in df_cleaned.columns:
        if len(df_cleaned[column].dropna())==0 or len(df_cleaned[column].unique())<2:
            to_drop=to_drop+[column]
    df_cleaned=df_cleaned.drop(columns=to_drop)
    to_drop=[]
    for column in df_cleaned.columns:
        try:
            if df_cleaned[column].dtype == 'object' or df_cleaned[column].dtype == 'string':
                df_cleaned[column] = df_cleaned[column].str.replace(' ', '').str.upper()
        except:
            to_drop=to_drop+[column]
    df_cleaned=df_cleaned.drop(columns=to_drop)
    return df_cleaned

def create_heatmap_with_dendrogram(df):
    numerical_columns = df.select_dtypes(include=np.number).columns
    if len(numerical_columns) == 0:
        print("No numerical columns found. Please ensure your dataframe contains numerical data.")
        return
    if df[numerical_columns].isnull().any().any():
        print("Numerical columns contain missing values. Please handle missing values before performing PCA.")
        return
    # Handle missing values by dropping rows with NaN
    df_cleaned = df.dropna()
    if df_cleaned.empty:
        print("After removing missing values, the dataframe is empty. Please ensure you have enough data for analysis.")
        return
    # Calculate the correlation matrix with numeric columns only
    correlation_matrix = df_cleaned.corr()
    # Generate the dendrogram
    try:
        dendrogram = hierarchy.dendrogram(hierarchy.linkage(correlation_matrix.values, method='ward'),
                                          labels=correlation_matrix.columns)
    except ValueError as e:
        print("Error generating dendrogram:", str(e))
        return
    # Reorder the columns based on the dendrogram
    reordered_columns = [correlation_matrix.columns[i] for i in dendrogram['leaves']]
    reordered_correlation_matrix = correlation_matrix[reordered_columns].reindex(reordered_columns)
    # Plot the heatmap with reordered correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(reordered_correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap with Dendrogram')
    plt.show()
    
def add_random_category(df, column_name = 'RandomVar'):
    # Generate random categorical values
    categories = np.random.choice([0, 1, 2], size=len(df))
    # Add the new column to the DataFrame
    df[column_name] = categories
    return df

def normalize_data(df, columns=None, excepcion='RandomVar'):
    if columns is None:
        columns = df.drop(columns=[excepcion]).select_dtypes(include=np.number).columns
    for column in columns:
        min_value = df[column].min()
        max_value = df[column].max()
        df[column] = (df[column] - min_value) / (max_value - min_value)
    return df


def TodasFunciones(df):    
    if len(df)<=1:
        print('Muy pocas filas')
        return
    if len(df.columns)<=1:
        print('Muy pocas columnas')
        return
    df=clean_strings(df)
    df=add_random_category(df)
    df=normalize_data(df)
    generate_UMAP(df)
    generate_tSNE(df)
    generate_PCA(df)
    create_heatmap_with_dendrogram(df)
    eda_plots(df)
    identify_outliers(df)
    missing_data_visualization(df)
    return calculate_statistics(df)