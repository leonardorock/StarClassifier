import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame as df
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, classification_report

# loading csv data, replacing empty data with NaN
df = pd.read_csv("input/stars.csv", na_values=' ')

# displaying some details
print(df.head())
print(df.describe())

# copying data set
df2 = df.copy()

# removing NaN
df2.dropna(inplace=True)
print(df2.isna().sum())

# Temperature mean and median without NaN
temperature_mean_without_nan = df2['Temperature'].mean()
temperature_median_without_nan = df2['Temperature'].median()
print(temperature_mean_without_nan, temperature_median_without_nan)

# Absolute Magintude mean and median without NaN
absolute_magnitude_mean_without_nan = df2['A_M'].mean()
absolute_magnitude_median_without_nan = df2['A_M'].median()
print(absolute_magnitude_mean_without_nan, absolute_magnitude_median_without_nan)

# adjusting df with median
df['Temperature'].fillna(temperature_median_without_nan, inplace=True)
df['A_M'].fillna(absolute_magnitude_median_without_nan, inplace=True)

# Temperature & Absolute Magnitude median after replacing 'missing' data
temperature_median = df['Temperature'].median()
absolute_magnitude_median = df['A_M'].median()
print(temperature_median, absolute_magnitude_median)

# Color & Spectral Class mode without NaN --default
color_mode = df['Color'].mode()[0]
spectral_class_mode = df['Spectral_Class'].mode()[0]
print(color_mode, spectral_class_mode)

# adjusting df with mode
df['Color'].fillna(color_mode, inplace=True)
df['Spectral_Class'].fillna(spectral_class_mode, inplace=True)

# looking for outliers in Absolute Magnitude using ZScore
df['A_M_zscore'] = zscore(df['A_M'])
print(df[['A_M', 'A_M_zscore']])

# removing outliers in Absolute Magnitude
df.drop(df[df.A_M_zscore >= 3].index, inplace=True)
print(df[['A_M', 'A_M_zscore']])

star_color = LabelEncoder()
star_color_types = star_color.fit_transform(df['Color'])
star_color_map = { index: label for index, label in enumerate(star_color.classes_) }

print(star_color_map)

# Adding type classification description to dataset
df_types = pd.DataFrame(
    [
        [0, 'Red Dwarf'],
        [1, 'Brown Dwarf'],
        [2, 'White Dwarf'],
        [3, 'Main Sequence'],
        [4, 'Super Giants'],
        [5, 'Hyper Giants']
    ], 
    index=range(0, 6), 
    columns=['Identifier', 'Description']
)
print(df_types)

df['Type_Description'] = ''
for i_df, r_df in df.iterrows():
    for i_df_types, r_df_types in df_types.iterrows():
        if (r_df['Type'] == r_df_types['Identifier']):
            df.at[i_df, 'Type_Description'] = df_types.at[i_df_types, 'Description']

# removing unnecessary columns from the df
df.drop(columns=['A_M_zscore', "Color", "Spectral_Class", "Type_Description", "Temperature", "A_M"], inplace=True)
print(df)

# creating data frames for decision tree
cols_X = df[["L","R"]]
cols_y = df[['Type']]
df_X = cols_X.copy()
df_y = cols_y.copy()
print(df_X)
print(df_y)

# separating the data set for train and testing
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.3, random_state=42)

# basic parameters
dt_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=1)

# training
dt_classifier = dt_classifier.fit(X_train, y_train)
y_train_prediction = dt_classifier.predict(X_train)
print("Training accuracy using DT:", accuracy_score(y_train, y_train_prediction))
print("Training precision using DT:", precision_score(y_train, y_train_prediction, average='weighted', zero_division=0))

# confusion matrix - for training
print(classification_report(y_train, y_train_prediction, zero_division=0))

# testing
y_test_prediction = dt_classifier.predict(X_test)

# test results
print("Test accuracy using DT:", accuracy_score(y_test, y_test_prediction))
print("Test precision using DT:", precision_score(y_test, y_test_prediction, average='weighted', zero_division=0))

print(df.corr())

tree.plot_tree(dt_classifier, fontsize=8)
plt.show()
