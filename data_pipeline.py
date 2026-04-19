import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger("Data_Pipeline")

def load_data(file_path):
    logger.info("Loading Dataset")
    df = pd.read_csv(file_path)
    return df


def basic_data_overview(df):
    pd.set_option('display.width', None)
    print(df.head(20))

    logger.info("============ Basic Functions ============")
    logger.info("Information About Data:")
    print(df.info())

    logger.info("Statistical operations:")
    print(df.describe().round(2))

    logger.info("Rows & Columns of Data:")
    print(df.shape)

    logger.info("Columns of Data:")
    print(df.columns)

    logger.info("Data Types:")
    print(df.dtypes)

    logger.info("Display Index Range:")
    print(df.index)

def clean_and_preprocess_data(df):
    logger.info("============ Cleaning Data ============")

    logger.info("Number of duplicate rows:")
    logger.info(df.duplicated().sum()) # = 0

    logger.info("Number of Missing Values:")
    print(df.isnull().sum()) # = 0

    #sns.heatmap(df.isnull(), annot=True)
    #plt.title("Missing Values in Data")
    #plt.show()

    logger.info("============ Data Preprocessing ============")

    logger.info('Converting TotalCharges to numeric and filling missing values...')
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    logger.info('Convert all "Yes/No" values to 0/1')
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]

    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    logger.info('Replacing "No phone service" with "No" in MultipleLines...')
    df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

    logger.info('Replacing "No internet service" with "No" in service columns...')
    replace_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in replace_cols:
        df[col] = df[col].replace({"No internet service": "No"})

    logger.info("Processed Dataset Preview:")
    print(df.head(20))

    logger.info("Final Data Types:")
    print(df.dtypes)

    df['NumServices'] = (
        df[['PhoneService', 'InternetService', 'StreamingMovies', 'StreamingTV']]
        .apply(lambda x: sum([
            1 if x['PhoneService'] == 1 else 0,
            1 if x['InternetService'] != 'No' else 0,
            1 if x['StreamingMovies'] == 1 else 0,
            1 if x['StreamingTV'] == 1 else 0
        ]), axis=1)
    )

    skew_value = df['tenure'].skew()
    logger.info(f"Skewness of tenure:{skew_value}")

    sns.histplot(df['tenure'], kde=True)
    plt.title("Distribution of tenure Before Treatment Skew")
    plt.show()

    skew_value = df['MonthlyCharges'].skew()
    logger.info(f"Skewness of MonthlyCharges:{skew_value}")

    sns.histplot(df['MonthlyCharges'], kde=True)
    plt.title("Distribution of MonthlyCharges Before Treatment Skew")
    plt.show()

    skew_value = df['TotalCharges'].skew()
    logger.info("Skewness of TotalCharges:", skew_value)

    sns.histplot(df['TotalCharges'], kde=True)
    plt.title("Distribution of TotalCharges Before Treatment Skew")
    plt.show()

    # we found :
    #Skewness of tenure: 0.2395397495619829 ---> natural
    #Skewness of MonthlyCharges: -0.22052443394398033 ----> natural
    #Skewness of TotalCharges: 0.9633155974592842 ----> (Moderately Skew) : Sqrt

    df['TotalCharges'] = np.sqrt(df['TotalCharges']) # -----> 0.3

    treat_skew_TotalCharges = df['TotalCharges'].skew()
    logger.info(f"Treatment Skew of TotalCharges:{treat_skew_TotalCharges}")
    sns.histplot(df['TotalCharges'], kde=True)
    plt.title("Distribution of TotalCharges After Treatment Skew (Sqrt)")
    plt.show()

    cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    print(df[cols].describe().round(2))

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        logger.info(f"--- {col} ---")
        logger.info(f"IQR: {IQR:.2f}")
        logger.info(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
        logger.info(f"Number of Outliers: {len(outliers)}\n")

    plt.figure(figsize=(15, 5))
    for i, col in enumerate(cols, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(y=df[col], color='skyblue')
        plt.title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()

    # [1] Outlier Detection Analysis:
    # We used the IQR (Interquartile Range) method to detect potential outliers
    # in the numerical features: 'tenure', 'MonthlyCharges', and 'TotalCharges'.

    # [2] Observations:
    # - Tenure: No outliers detected (Values are within the logical range of customer lifespan).
    # - MonthlyCharges: No outliers detected (Pricing follows a consistent distribution).
    # - TotalCharges: After applying the Square Root (sqrt) transformation,
    #   the outliers were successfully handled, resulting in 0 outliers.

    # [3] Conclusion:
    # The numerical features are now normally distributed (skewness ~ 0.3)
    # and free of outliers, which enhances the stability and performance
    # of the CatBoost model.

    return df


def run_data_pipeline(file_path):
    logger.info("============ Starting Data Pipeline ============")

    df = load_data(file_path)
    basic_data_overview(df)
    df_processed = clean_and_preprocess_data(df)

    logger.info("============ Data Pipeline Completed ============")

    return df_processed


if __name__ == "__main__":
    FILE_PATH = r"C:\Users\Hedaya_city\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df_final = run_data_pipeline(FILE_PATH)
