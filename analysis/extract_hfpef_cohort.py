import pandas as pd
from pathlib import Path

CSV_DIR = Path("output/csv")
OUT_PATH = Path("output/hfpef_cohort.csv")

patients = pd.read_csv(CSV_DIR / "patients.csv")
conditions = pd.read_csv(CSV_DIR / "conditions.csv")
observations = pd.read_csv(CSV_DIR / "observations.csv")

# 1. Get heart failure patients
hf = conditions[
    conditions["DESCRIPTION"].str.contains("heart failure", case=False, na=False)
].copy()

hf_ids = set(hf["PATIENT"].unique())

print("Heart failure patients:", len(hf_ids))

# 2. Keep observations only for HF patients
obs = observations[observations["PATIENT"].isin(hf_ids)].copy()
obs["DATE"] = pd.to_datetime(obs["DATE"], errors="coerce")
obs["VALUE_NUM"] = pd.to_numeric(obs["VALUE"], errors="coerce")

# 3. Map Synthea observation descriptions to clean column names
wanted = {
    "Body mass index (BMI) [Ratio]": "BMI",
    "Systolic Blood Pressure": "SBP",
    "Diastolic Blood Pressure": "DBP",
    "Heart rate": "HR",
    "Left ventricular Ejection fraction": "EF",
    "Creatinine [Mass/volume] in Blood": "Creatinine_Blood",
    "Creatinine [Mass/volume] in Serum or Plasma": "Creatinine_Serum",
    "Glomerular filtration rate [Volume Rate/Area] in Serum or Plasma by Creatinine-based formula (MDRD)/1.73 sq M": "eGFR",
    "Sodium [Moles/volume] in Blood": "Sodium_Blood",
    "Sodium [Moles/volume] in Serum or Plasma": "Sodium_Serum",
    "Potassium [Moles/volume] in Blood": "Potassium_Blood",
    "Potassium [Moles/volume] in Serum or Plasma": "Potassium_Serum",
    "Oxygen saturation in Arterial blood": "O2_Sat",
}

obs = obs[obs["DESCRIPTION"].isin(wanted.keys())].copy()
obs["VARIABLE"] = obs["DESCRIPTION"].map(wanted)

# 4. Take latest value per patient per variable
latest = (
    obs.sort_values("DATE")
    .groupby(["PATIENT", "VARIABLE"], as_index=False)
    .tail(1)
)

wide = latest.pivot_table(
    index="PATIENT",
    columns="VARIABLE",
    values="VALUE_NUM",
    aggfunc="first"
).reset_index()

# 5. Add demographics
demo = patients[patients["Id"].isin(hf_ids)].copy()

demo["BIRTHDATE"] = pd.to_datetime(demo["BIRTHDATE"], errors="coerce")
demo["DEATHDATE"] = pd.to_datetime(demo["DEATHDATE"], errors="coerce")

# Use current year based on Synthea output reference-ish date.
# If patient is deceased, age at death. Otherwise age in 2026.
reference_date = pd.Timestamp("2026-06-14")

demo["END_DATE"] = demo["DEATHDATE"].fillna(reference_date)
demo["AGE"] = (demo["END_DATE"] - demo["BIRTHDATE"]).dt.days / 365.25

demo = demo.rename(columns={
    "Id": "PATIENT",
    "GENDER": "Sex",
    "RACE": "Race",
    "ETHNICITY": "Ethnicity",
    "INCOME": "Income",
    "HEALTHCARE_EXPENSES": "Healthcare_Expenses",
    "HEALTHCARE_COVERAGE": "Healthcare_Coverage",
})

demo_cols = [
    "PATIENT",
    "AGE",
    "Sex",
    "Race",
    "Ethnicity",
    "Income",
    "Healthcare_Expenses",
    "Healthcare_Coverage",
    "BIRTHDATE",
    "DEATHDATE",
]

cohort = demo[demo_cols].merge(wide, on="PATIENT", how="left")

# 5b. Comorbidity flags (HFpEF is defined as much by its comorbidities as its EF).
# Scan each HF patient's full condition history for the key associated diagnoses.
hf_conditions = conditions[conditions["PATIENT"].isin(hf_ids)]

comorbidity_terms = {
    "Hypertension": "hypertension",
    "Atrial_Fibrillation": "atrial fibrillation",
    "Diabetes": "diabetes mellitus type 2",
    "CKD": "chronic kidney disease",
    "Obesity": "body mass index 30+",
}

comorbid = pd.DataFrame({"PATIENT": list(hf_ids)})
for col, term in comorbidity_terms.items():
    pos = set(
        hf_conditions[
            hf_conditions["DESCRIPTION"].str.contains(term, case=False, na=False)
        ]["PATIENT"].unique()
    )
    comorbid[col] = comorbid["PATIENT"].isin(pos)

cohort = cohort.merge(comorbid, on="PATIENT", how="left")

# 6. Define clean HFpEF cohort:
# Heart failure diagnosis + latest EF >= 50
cohort = cohort[cohort["EF"] >= 50].copy()

# 7. Prefer serum values when available, otherwise blood values
cohort["Creatinine"] = cohort["Creatinine_Serum"].fillna(cohort["Creatinine_Blood"])
cohort["Sodium"] = cohort["Sodium_Serum"].fillna(cohort["Sodium_Blood"])
cohort["Potassium"] = cohort["Potassium_Serum"].fillna(cohort["Potassium_Blood"])

# 7b. Drop physiologically impossible creatinine values.
# Human serum creatinine essentially never exceeds ~15 mg/dL, even in end-stage
# kidney disease; higher values are generator runaways and would skew analysis.
MAX_PLAUSIBLE_CREATININE = 15.0
n_before = len(cohort)
cohort = cohort[
    cohort["Creatinine"].isna() | (cohort["Creatinine"] <= MAX_PLAUSIBLE_CREATININE)
].copy()
print(f"Dropped {n_before - len(cohort)} patients with creatinine > {MAX_PLAUSIBLE_CREATININE} mg/dL")

# 8. Final columns
final_cols = [
    "PATIENT",
    "AGE",
    "Sex",
    "Race",
    "Ethnicity",
    "BMI",
    "SBP",
    "DBP",
    "HR",
    "EF",
    "Creatinine",
    "eGFR",
    "Sodium",
    "Potassium",
    "O2_Sat",
    "Hypertension",
    "Atrial_Fibrillation",
    "Diabetes",
    "CKD",
    "Obesity",
    "Income",
    "Healthcare_Expenses",
    "Healthcare_Coverage",
    "BIRTHDATE",
    "DEATHDATE",
]

cohort = cohort[final_cols]

# 9. Save
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
cohort.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Final HFpEF cohort size:", len(cohort))
print()
print("Missing values per column:")
print(cohort.isna().sum())
print()
print("Summary:")
print(cohort.describe(include="all"))
