import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE

data = pd.read_csv('midterm_project/ai4i2020.csv', delimiter=',')
# print(data)
processed_data = data.iloc[:,2:9]
print(processed_data)
has_missing = processed_data.isna().any().any()
print(has_missing)  

H=0
M=0
L=0
S=0
for i in processed_data.iloc[:,0]:
    if i == 'H':
        H += 1
    elif i == 'M':
        M += 1
    elif i == 'L':
        L += 1
    S += 1
prob_H = H/S
prob_M = M/S
prob_L = L/S
S=0
for i in processed_data.iloc[:,0]:
    if i == 'H':
        processed_data.iloc[S, 0] = 0.1
    elif i == 'M':
        processed_data.iloc[S, 0] = 0.6
    elif i == 'L':
        processed_data.iloc[S, 0] = 0.3
    S += 1

# processed_data.to_csv('midterm_project/data.csv', index=False)

X_train = processed_data.iloc[:,:6]
y_train = processed_data.iloc[:,6]

smote = BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
# print(y_train)
df_combined = np.hstack((X_train, y_train.values.reshape(-1, 1)))
print(df_combined.shape)
np.save('midterm_project/ai4i2020_oversample.npy', df_combined)