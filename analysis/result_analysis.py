import pandas as pd

p = pd.read_csv("output/csv/patients.csv")
c = pd.read_csv("output/csv/conditions.csv")
o = pd.read_csv("output/csv/observations.csv")

hf = c[c["DESCRIPTION"].str.contains("heart failure", case=False, na=False)]
hf_ids = set(hf["PATIENT"].unique())

ef = o[
    o["PATIENT"].isin(hf_ids)
    & o["DESCRIPTION"].str.contains("ejection fraction", case=False, na=False)
]

ef_vals = pd.to_numeric(ef["VALUE"], errors="coerce")

print("Total patients:", len(p))
print("HFpEF patients:", len(hf_ids))
print("HF EF min:", ef_vals.min())
print("HF EF max:", ef_vals.max())
print("HF EF patients:", ef["PATIENT"].nunique())
