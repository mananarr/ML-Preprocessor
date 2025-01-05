# Example usage: python preprocess.py dataset.csv
# Keep the script in the same folder as the dataset.

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler,StandardScaler,normalize
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.model_selection import KFold
from scipy.stats import zscore, skew, shapiro
import category_encoders as ce
import sys

# Load your dataset
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Total rows: {df.shape[0]}, Total columns: {df.shape[1]}\n")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Describe the data and provide insights
def data_insights(df):
    print("\n### Data Insights ###")
    print("Total Columns:", len(df.columns))
    
    
    # Identifying scale differences
    print("\nScale Differences between columns:")
    print(df.describe().transpose())
    
    # Categorical and Numerical Column Segregation
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("\nTotal count of Categorical Columns:", len(categorical_cols))
    print("\nCategorical Columns:", categorical_cols)
    print("\n\nTotal count of Numerical Columns:", len(numerical_cols))
    print("\nNumerical Columns:", numerical_cols)

    # Standard deviation analysis
    std_dev = df[numerical_cols].std().sort_values(ascending=False)
    print("\nNumerical Columns with highest and lowest standard deviation:")
    print(std_dev)

    return categorical_cols, numerical_cols, std_dev

def label_encoder(df, col, order):
    print("Label Encoding............")
    label_encoder = LabelEncoder()
    if order == []:
        df[col] = label_encoder.fit_transform(df[col])
    else:
        label_encoder.fit(order)
        df[col] = label_encoder.transform(df[col])
    return df

def one_hot_encoding(df, col):
    print("One Hot Encoding...........")
    encoded_columns = pd.get_dummies(df[col], dtype=int)
    df = df.join(encoded_columns).drop(col, axis=1)
    return df

def binary_encoding(df, col):
    print("Binary Encoding..........")
    encoder = ce.BinaryEncoder(cols=[col])
    encoded_df = encoder.fit_transform(df[col])
    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(columns=[col])
    return df

def target_encoding(df, col, target_name, folds):
    print("Target Encoding..........")
    print(f"NOTE: The '{col}' will be converted to '{col}_encoded'")
    
    # K-Fold setup
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    
    # Create a column for storing the target-encoded values, initialized with NaN (to handle floats)
    df_encoded = df.copy()
    df_encoded[col + '_encoded'] = np.nan  # Ensure the column is of float type
    
    # Perform K-Fold Cross-Validation
    for i, (train_index, valid_index) in enumerate(kf.split(df)):
        print(f"\nFold {i+1}:")
        
        # Split the data into training and validation sets
        train_df, valid_df = df.iloc[train_index], df.iloc[valid_index]
        
        # Initialize Target Encoder
        encoder = ce.TargetEncoder(cols=[col])
        
        # Fit the encoder on the training set
        encoder.fit(train_df[col], train_df[target_name])
        print(f"Fitting encoder on training data, fold {i+1}")
        
        # Transform the validation set
        transformed_valid = encoder.transform(valid_df[col])
        print(f"Transformed validation set for fold {i+1}")
        
        # Update the encoded column in the df_encoded DataFrame
        df_encoded.loc[valid_index, col + '_encoded'] = transformed_valid[col].values
    
    # Drop the original column that was encoded
    df_encoded = df_encoded.drop(columns=[col])
    
    # Check final output
    print(df_encoded.head())
    
    return df_encoded

# Preprocess Categorical Columns
def preprocess_categorical(df, categorical_cols):
    for col in categorical_cols:
        unique_values = df[col].unique()
        n_unique_values = df[col].nunique()
        print(f"\nColumn: {col} | Unique Values: {n_unique_values}", unique_values)
        choice = 0
        while choice not in [1, 2, 3, 4]:
            try:
                choice = int(input("Please select one of the following encoding methods for this column:\n"
                                   "1. Label Encoding\n"
                                   "2. One Hot Encoding\n"
                                   "3. Binary Encoding\n"
                                   "4. Target (Mean) Encoding\n"))
                if choice not in [1, 2, 3, 4]:
                    print("Invalid input. Please enter a number between 1 and 4.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        if choice == 1:
            order_input = input("Explicit order (or just press Enter if you want this do be done alphabetically): ").strip()
            if  order_input == "":
                order = []
            else:
                order = [item.strip() for item in order_input.split(",")]
            df = label_encoder(df, col, order)
        elif choice == 2:
            df = one_hot_encoding(df, col)
        elif choice == 3:
            df = binary_encoding(df, col)
        elif choice == 4:
            target = input("Enter the target feature name: ")
            folds = int(input("Enter the count of folds for K-fold Cross Validation: "))
            df = target_encoding(df, col, target, folds)
        else:
            print("Invalid input. Please enter a number between 1 and 4.")


    return df


def handle_null_values(df):
    # Handle Missing Values
    for col in df.columns:
        missing_values = df[col].isnull().sum()
        missing_percentage = (missing_values / len(df[col])) * 100
        print(f"\nColumn: {col} | Missing Values: {missing_values} | {missing_percentage:.2f}% Missing")
    
        if missing_values > 0:
            if df[col].dtype in ['int64', 'float64']:  # Numerical column
                print("Select a method to handle missing values:")
                print("1. Remove rows with missing values")
                print("2. Numerical imputation (fill missing values with 0, mean, or median)")
                print("3. KNN imputation")
                method = int(input("Enter the number of the method you'd like to use: "))

                if method == 1:
                    df = df.dropna(subset=[col])
                    print(f"Rows with missing values in column {col} have been removed.")
                elif method == 2:
                    print("Choose an imputation method:")
                    print("1. Fill with 0")
                    print("2. Fill with mean")
                    print("3. Fill with median")
                    impute_method = int(input("Enter the number of the imputation method: "))

                    if impute_method == 1:
                        df[col] = df[col].fillna(0)
                    elif impute_method == 2:
                        df[col] = df[col].fillna(df[col].mean())
                    elif impute_method == 3:
                        df[col] = df[col].fillna(df[col].median())
                elif method == 3:
                    imputer = KNNImputer(n_neighbors=5)
                    df[col] = imputer.fit_transform(df[[col]])
                    print(f"Missing values in column {col} have been filled.")
                else:
                    print("Invalid option selected.")
        
            elif df[col].dtype == 'object':  # Categorical column
                print("Select a method to handle missing values:")
                print("1. Remove rows with missing values")
                print("2. Categorical imputation (fill missing values with most frequent)")
                method = int(input("Enter the number of the method you'd like to use: "))

                if method == 1:
                    df = df.dropna(subset=[col])
                    print(f"Rows with missing values in column {col} have been removed.")
                elif method == 2:
                    df[col] = df[col].fillna(df[col].mode()[0])
                    print(f"Missing values in column {col} have been filled with the most frequent value.")
                else:
                    print("Invalid option selected.")
    return df

# Outlier detection
def detect_outliers(df):
    print("### Outlier Detection Using Z-Score and Standard Deviation ###\n")
    
    print("NOTE: Avoid outlier handling if you are using TREE Based algos or if outliers are practical and significant\n")
    columns = list(df.columns)
    print("Which columns do you want to exclude from the outlier detection process?")
    print("Enter numbers separated by commas(1,2,3,...), press 'Enter' to include all columns, or type 'skip' to skip outlier management entirely.\n")
    for i, col in enumerate(columns, 1):
        print(f"{i}. {col}")
    
    exclude_input = input("\nEnter your choice: ").strip().lower()
    if exclude_input == 'skip':
        print("\nOutlier management skipped for all columns.")
        return df
    # If no columns are excluded, continue with all columns
    if exclude_input:
        exclude_indices = [int(x.strip()) - 1 for x in exclude_input.split(',')]
        numerical_cols = [col for i, col in enumerate(columns) if i not in exclude_indices]
    else:
        numerical_cols = columns  # All columns are used if no input is provided

    outlier_indices = {}

    print("\nChecking outliers with z_score method")
    for col in numerical_cols:
        z_scores = np.abs(zscore(df[col]))
        
        outliers = (z_scores > 3)

        outlier_count = np.sum(outliers)
        outlier_indices[col] = df[outliers].index.tolist()
        
        print(f"\nColumn: {col}")
        print(df[col][outliers])
        print(f"Number of outliers: {outlier_count}")
        print(f"Outlier indices: {outlier_indices[col]}")

        if outlier_count > 0:
            print("\nCapping the outliers: ")
            # Capping: replace outliers with the min/max non-outlier values
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
            print(f"Capped the outliers in column {col}.")
        else:
            print("Zero outliers present, no need for handling.")

    print("\nNumerical DataFrame after outlier handling:")
    print(df.head())
    return df


# Helper function to display method choices
def display_method_choices():
    print("\n**1. Standardization (MinMax or Z-score)** *(Use when the columns have similar scales but different distributions which is not necessarily proportional, for eg: height in cm (0–200) and weight in kg (0–150) -  Support Vector Machines (SVM), Principal Component Analysis (PCA), Logistic Regression, k-Means Clustering.)*")
    print("**2. Normalization (L1/L2)** *(Use when there is a significant difference in the scales of the columns, for eg: income in dollars (thousands to millions) and age in years (10s to 100s) -  Neural Networks, k-Nearest Neighbors (k-NN), Gradient Descent-based models.)*")
    print("**3. Log Transform** *(If your data has some very large numbers compared to most of the other values, making the chart look stretched out on one side, for eg:  Income data where a few people earn millions while most earn thousands - Linear Regression, Decision Trees, Random Forests)*")
    choice = input("\nSelect a scaling method (1/2/3): ")
    return choice

# Helper function to display scaling choices
def display_standardization_choices():
    print("\n1. MinMax Standardization (Use when features have different units or ranges, and you need to normalize them to a specific range (e.g., 0 to 1), especially for algorithms like neural networks or gradient descent-based models.)")
    print("2. Z-score Standardization (Use when features have different means and variances, and you need to standardize them to have a mean of 0 and a standard deviation of 1, especially for algorithms like SVM, PCA, or k-means clustering.)")
    choice = input("\nSelect a Standardization method (1/2): ")
    return choice

# Helper function to display normalization choices
def display_normalization_choices():
    print("\n1. L1 Normalization (Use when you want to make features smaller and less affected by outliers, good for sparse data like text.)")
    print("2. L2 Normalization (Use when you want to evenly scale features but still keep the impact of bigger values, useful for models like regression.)")
    choice = input("\nSelect a normalization method (1/2): ")


# Helper function to handle Log Transformation
def apply_log_transform(df, col):
    if (df[col] < 0).any():
        df[col] = (df[col] - df[col].min() + 1).transform(np.log)
        print(f"Applied log transform to column {col} with `- min() + 1` adjustment.")
    elif (df[col] == 0).any():
        df[col] = (df[col] + 1).transform(np.log)
        print(f"Applied log transform to column {col} with `+1` adjustment.")
    else:
        df[col] = df[col].transform(np.log)
        print(f"Applied standard log transform to column {col}.")
    return df

def display_suggests(df,col):
    skewness = skew(df[col])
    shapiro_stat, shapiro_p = shapiro(df[col])
    data_range = df[col].max() - df[col].min()
    std_dev_mean_ratio = (df[col].std() / df[col].mean()) * 100
    print(f"Skewness: {skewness:.2f}")
    print(f"Range: {data_range:.2f}")
    print(f"Standard Deviation as % of Mean: {std_dev_mean_ratio:.2f}%")
    print(f"Shapiro-Wilk Test for Normality: p-value={shapiro_p:.3f}")
    if abs(skewness) > 1:
        print("Recommendation: Apply Log Transform (Highly skewed data).")
    elif data_range > 100:
        print("Recommendation: Apply Min-Max Standardization (Large range).")
    elif shapiro_p > 0.05:
        print("Recommendation: Apply Standardization (Data is close to normal).")
    elif std_dev_mean_ratio > 30:
        print("Recommendation: Apply L2 Normalization (Balances feature magnitudes).")
    else:
        print("Recommendation: Consider Normalization or Standardization based on use case.")
    print("\n")

def preprocess_numerical(df, numerical_cols):
    print("\nHow would you like to proceed with scaling of numerical columns:")
    print("1. All columns at once using a single method")
    print("2. Individual methods for individual columns")
    choice = input("Select an approach from above (1/2): ")

    if choice == '1':
        # Handle all columns with a single method
        method_choice = display_method_choices()
        if method_choice == '1':  # Standardization
            standardization_choice = display_standardization_choices()
            scaler = MinMaxScaler() if standardization_choice == '1' else StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            print(f"Applied {'MinMax scaling' if standardization_choice == '1' else 'Z-score normalization'} to all numerical columns.")
        elif method_choice == '2':  # Normalization
            norm_choice = display_normalization_choices()
            norm = 'l1' if norm_choice == '1' else 'l2'
            df[numerical_cols] = normalize(df[numerical_cols], norm=norm)
            print(f"Applied {'L1' if norm_choice == '1' else 'L2'} normalization to all numerical columns.")
        elif method_choice == '3':  # Log Transform
            for col in numerical_cols:
                df = apply_log_transform(df, col)
            print(f"Applied log transform to all numerical columns.")
        else:
            print("Invalid choice. No changes applied.")

    elif choice == '2':
        for col in numerical_cols:
            print(f"\nColumn: {col}\n")
            display_suggests(df,col)
            method_choice = display_method_choices()
            if method_choice == '1':
                standardization_choice = display_standardization_choices()
                scaler = MinMaxScaler() if standardization_choice == '1' else StandardScaler()
                df[col] = scaler.fit_transform(df[col].to_frame(name=col))
                print(f"Applied {'MinMax scaling' if standardization_choice == '1' else 'Z-score normalization'} to column {col}.")
            elif method_choice == '2':  # Normalization
                norm_choice = display_normalization_choices()
                norm = 'l1' if norm_choice == '1' else 'l2'
                df[col] = normalize(pd.DataFrame(df[col]), norm=norm)
                print(f"Applied {'L1' if norm_choice == '1' else 'L2'} normalization to column {col}.")
            elif method_choice == '3':  # Log Transform
                df = apply_log_transform(df, col)
            else:
                print(f"Invalid choice for column {col}. No changes applied.")

    else:
        print("Invalid approach selected. No changes applied.")

    print("\nDataFrame after numerical preprocessing:")
    print(df.head())
    return df


# Save the preprocessed dataset to a new CSV file
def save_preprocessed_dataset(df, file_path):
    output_file = file_path.replace('.csv', '_preprocessed.csv')
    df.to_csv(output_file, index=False)
    print(f"\nPreprocessed dataset saved to {output_file}")

def rename_columns(df):
    rename_cols = input("\nEnter column renames (original1:new1,original2:new2,.....), or press Enter to skip: ").strip()
    if rename_cols:  # If input is not empty
        rename_dict = {}
        
        for pair in rename_cols.split(','):
            try:
                orig, new = pair.split(':')
                orig = orig.strip()
                new = new.strip()
                if orig in df.columns:
                    rename_dict[orig] = new
                else:
                    raise ValueError(f"{orig} is NOT a valid column name.")
            except ValueError as e:
                print(e)

        # Rename the DataFrame
        df = df.rename(columns=rename_dict)
        print("\nUpdated DataFrame:\n", df)
    else:
        print("Dataframe not affected")
    return df


# Main function with preprocessing
def preprocess_dataset(file_path):
    df = load_dataset(file_path)
    if df is None:
        return

    del_cols = input("\nDo you want to remove any columns? (insert comma separated values col1,col2,..), or press Enter to skip: ")
    if del_cols != "":
        delete_cols = del_cols.split(",")
        df = df.drop(columns=delete_cols)

    df = rename_columns(df)

    df = handle_null_values(df.copy())

    print("\nSample of data:\n", df.head())
    print("\nColumns:\n", df.columns.tolist())


    # Get initial data insights
    categorical_cols, numerical_cols, std_dev = data_insights(df)
    
    # Preprocess Categorical Columns
    print("\nEntering preprocessing for Categorical columns")
    df1 = preprocess_categorical(df[categorical_cols].copy(), categorical_cols)

    # categorical_cols, numerical_cols, std_dev = data_insights(df)
    df = detect_outliers(df[numerical_cols].copy())

    df_cat_outliers = pd.concat([df1, df], axis=1)
    print("Your whole Dataframe looks like this so far.......\n")
    print(df_cat_outliers.head())

    # Preprocess Numerical Columns
    print("\nEntering preprocessing for Numerical columns")
    print(f"\n Numerical columns: {numerical_cols}\n")
    df2 = preprocess_numerical(df[numerical_cols].copy(), numerical_cols)
    
    print("\n### Preprocessing Completed ###")
    
    #Contact num and cat cols
    df = pd.concat([df1, df2], axis=1)
    print("\nScale Differences between columns:")
    print(df.describe().transpose())
    
    # # Standard deviation analysis
    std_dev = df.std().sort_values(ascending=False)
    print("\nNumerical Columns with highest and lowest standard deviation:")
    print(std_dev)
    print("Final dataset shape:", df.shape)
    
    # # Save the final preprocessed dataset to a new CSV file
    save_preprocessed_dataset(df, file_path)
    return df

file_path = sys.argv[1]
preprocessed_data = preprocess_dataset(file_path)
