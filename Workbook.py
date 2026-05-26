import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load Dataset
df = pd.read_csv("G:\Pw skills\Assignment 41(python)\employee_productivity_dataset.csv")

print("========== Q1 Handle Missing Values ==========")

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
df["Hours_Worked_Per_Week"] = df["Hours_Worked_Per_Week"].fillna(df["Hours_Worked_Per_Week"].median())
df["Performance_Score"] = df["Performance_Score"].fillna(df["Performance_Score"].mean())

print(df.head())

print("\n========== Q2 Label Encoding ==========")

label_encoder = LabelEncoder()

df["Gender"] = label_encoder.fit_transform(df["Gender"])
df["Department"] = label_encoder.fit_transform(df["Department"])

print(df[["Gender", "Department"]].head())

print("\n========== Q3 One-Hot Encoding ==========")

df = pd.get_dummies(df, columns=["Work_Mode", "Location"])

print(df.head())

print("\nNumber of Columns After One-Hot Encoding:")
print(df.shape[1])

print("\n========== Q4 Normalization (Min-Max Scaling) ==========")

minmax_scaler = MinMaxScaler()

df[["Salary", "Hours_Worked_Per_Week"]] = minmax_scaler.fit_transform(
    df[["Salary", "Hours_Worked_Per_Week"]]
)

print(df[["Salary", "Hours_Worked_Per_Week"]].head())

print("\n========== Q5 Standardization (Scaling) ==========")

standard_scaler = StandardScaler()

df[["Age", "Projects_Completed"]] = standard_scaler.fit_transform(
    df[["Age", "Projects_Completed"]]
)

print(df[["Age", "Projects_Completed"]].head())

print("\n========== Q6 Compare Scaling Methods ==========")

salary_original = pd.read_csv("G:\Pw skills\Assignment 41(python)\employee_productivity_dataset.csv")["Salary"]

comparison_df = pd.DataFrame()

comparison_df["Original_Salary"] = salary_original

comparison_df["MinMax_Salary"] = MinMaxScaler().fit_transform(
    salary_original.values.reshape(-1, 1)
)

comparison_df["Standard_Salary"] = StandardScaler().fit_transform(
    salary_original.values.reshape(-1, 1)
)

print(comparison_df.head())

print("\n========== Q7 Build Preprocessing Pipeline ==========")

categorical_columns = ["Gender", "Department"]
numerical_columns = ["Age", "Salary", "Hours_Worked_Per_Week",
                     "Performance_Score", "Projects_Completed"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ]
)

print("Pipeline Created Successfully")

print("\n========== Q8 Apply Pipeline ==========")

X = pd.read_csv("G:\Pw skills\Assignment 41(python)\employee_productivity_dataset.csv")

transformed_data = preprocessor.fit_transform(X)

print("Shape of Final Dataset:")
print(transformed_data.shape)

print("\nTransformed Dataset:")
print(transformed_data)

print("\n========== Q9 Conceptual Question ==========")

print("Scaling is important because it brings all features to a similar range,")
print("which improves machine learning model performance and accuracy.")

print("\n========== Q10 Conceptual Question ==========")

print("Categorical data is converted into numerical form because machine")
print("learning algorithms work with numerical values, not text data.")
