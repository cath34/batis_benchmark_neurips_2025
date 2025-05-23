import pandas as pd
import numpy as np
import ast 
import ast

df_test  = pd.read_csv("finalized_splits_kenya/test_filtered.csv")
df_test["target"] = df_test["target"].apply(ast.literal_eval)
y_matrix = np.array(list(df_test['target']))
print(y_matrix.shape)
y_mean = y_matrix.mean(axis=0)
print(y_mean.shape)

for i in range(len(df_test)):
    hotspot = df_test.iloc[i]['hotspot_id']
    np.save(f"/Users/cathv/Desktop/PhD/Main Project/Testing_Env/evaluate_results/predictions_kenya/mean_enc/{hotspot}.npy", y_mean)
