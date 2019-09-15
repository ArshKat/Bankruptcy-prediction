import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_path = os.path.join(os.getcwd(), "..", "input")
target_col = "isBankrupted"

df_train = pd.read_csv(os.path.join(input_path, "train.csv"), index_col="id")
df_test = pd.read_csv(os.path.join(input_path, "test.csv"), index_col="id")

df_train.head()
df_train.describe()

print(df_train[target_col].value_counts())
sns.set_style("whitegrid")
sns.countplot(df_train[target_col])
os.makedirs('../files/', exist_ok=True)
plt.savefig('../files/unbalnced.jpg')


df_train.isnull().any()
df_test.isnull().any()
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median", missing_values=np.nan)),
    ("scaler", StandardScaler()),
])

X = pipe.fit_transform(df_train.drop(target_col, axis=1))
X_test = pipe.transform(df_test)
y = df_train[target_col].values
