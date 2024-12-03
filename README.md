# handling-missing-data
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # To enable IterativeImputer
from sklearn.impute import IterativeImputer

# Example dataset with missing values
data = {'Age': [25, 30, 35, None, 40, None],
        'Salary': [50000, None, 55000, 60000, 65000, None]}

df = pd.DataFrame(data)

# Initialize the IterativeImputer (MICE-like imputation)
imp = IterativeImputer(max_iter=10, random_state=0)

# Perform imputation
imputed_df = imp.fit_transform(df)

# Convert the imputed data back into a DataFrame
imputed_df = pd.DataFrame(imputed_df, columns=df.columns)

# View the imputed dataset
print(imputed_df)
