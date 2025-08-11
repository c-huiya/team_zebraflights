#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# In[2]:
# 1. Load the original dataset
file_path = r"C:\Users\Chin Zhi Yueh\Desktop\NYP Y3S1\EGT307\Project\team_zebraflights\data\Electric_Vehicle_Population_Data.csv"
original_df = pd.read_csv(file_path)

# In[3]:
# 2. Randomly split 25% for inference
inference_df = original_df.sample(frac=0.25, random_state=42)

# 3. Remaining 75% for your work
df = original_df.drop(inference_df.index)

# In[4]:
# 4. Save inference file
inference_df.to_csv(r"C:\Users\Chin Zhi Yueh\Desktop\NYP Y3S1\EGT307\Project\team_zebraflights\data\inference_dataset.csv", index=False)
df.to_csv(r"C:\Users\Chin Zhi Yueh\Desktop\NYP Y3S1\EGT307\Project\team_zebraflights\data\working_dataset.csv", index=False)

# In[5]:
df.head(5)

# ### Check Data types

# In[6]:
# Column data types
print("\nColumn Data Types:")
print(df.dtypes)

# Missing values per column
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Unique counts per column
print("\nUnique Values per Column:")
print(df.nunique())

# Sample values per column (first 5 unique)
print("\nSample Values per Column:")
for col in df.columns:
    print(f"{col}: {df[col].dropna().unique()[:5]}")

# ## Data Cleaning
# ### Checking for duplicates

# In[7]:
# Show original row count
before_count = len(df)
print(f"Rows before duplicate removal: {before_count}")

# Drop exact full-row duplicates
df.drop_duplicates(inplace=True)

# Drop duplicates based on DOL Vehicle ID
df.drop_duplicates(subset=['DOL Vehicle ID'], inplace=True)

# Show new row count
after_count = len(df)
print(f"Rows after duplicate removal: {after_count}")

# Show how many rows were removed
print(f"Total rows removed: {before_count - after_count}")

# ### Checking for missing values

# In[8]:
missing_counts = df.isna().sum()
missing_percent = (missing_counts / len(df)) * 100

missing_summary = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing %': missing_percent
}).sort_values(by='Missing %', ascending=False)

print(missing_summary)

# In[9]:
df = df.dropna()

# In[10]:
missing_counts = df.isna().sum()
missing_percent = (missing_counts / len(df)) * 100

missing_summary = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing %': missing_percent
}).sort_values(by='Missing %', ascending=False)

print(missing_summary)

# ## Fixing data types

# In[11]:
# === Detect, clean (lowercase), and re-check ===
def report_text_issues(frame):
    for col in frame.select_dtypes(include='object').columns:
        unique_vals = frame[col].dropna().unique()
        lower_map = {}
        for val in unique_vals:
            val_stripped = str(val).strip()
            lower_val = val_stripped.lower()
            lower_map.setdefault(lower_val, set()).add(val_stripped)
        case_issues = {k: v for k, v in lower_map.items() if len(v) > 1}
        if case_issues:
            print(f"Case issues in column '{col}': {case_issues}")

report_text_issues(df)

# In[12]:
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip().str.lower()

report_text_issues(df)

# In[13]:

# --- normalize ALL text columns: remove extra spaces ---
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

# --- specifically for Model: remove ALL spaces ---
if "Model" in df.columns:
    df["Model"] = df["Model"].str.replace(r"\s+", "", regex=True)

# 1) ensure numeric
df["Base MSRP"] = pd.to_numeric(df["Base MSRP"], errors="coerce")

# 2) treat zeros/negatives as missing
df.loc[df["Base MSRP"] <= 0, "Base MSRP"] = np.nan

# 3) drop known outliers (you flagged these)
df = df[~df["Base MSRP"].isin([845000.0, 184400.0])]

# 4) missingness flag
df["Base_MSRP_missing"] = df["Base MSRP"].isna().astype(int)

# 5) hierarchical median imputation
msrp = df["Base MSRP"].copy()
if all(c in df.columns for c in ["Make","Model","Model Year"]):
    msrp = msrp.fillna(df.groupby(["Make","Model","Model Year"])["Base MSRP"].transform("median"))
if all(c in df.columns for c in ["Make","Model"]):
    msrp = msrp.fillna(df.groupby(["Make","Model"])["Base MSRP"].transform("median"))
if "Make" in df.columns:
    msrp = msrp.fillna(df.groupby("Make")["Base MSRP"].transform("median"))
msrp = msrp.fillna(msrp.median())  # final fallback
df["Base MSRP"] = msrp

# In[14]:
df.head()

# In[15]:
for col in df.columns:
    print(f"\n=== {col} ===")
    print(f"Unique count: {df[col].nunique(dropna=True)}")
    print(df[col].dropna().unique()[:10])

# ## Outliers
# In[16]:
def find_outliers(series, upper_bound=None, ignore_zeros=True, lower_tail=False,
                  method="iqr", p_hi=0.99):
    s = pd.to_numeric(series, errors="coerce")
    if ignore_zeros:
        s = s.mask(s == 0)
    s = s.dropna()
    if s.empty:
        return {"count": 0, "percent_col": 0.0, "percent_ds": 0.0,
                "bound_used": None, "max_value": None, "outlier_values": []}

    if method == "iqr":
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            iqr_upper = s.quantile(0.99)
        else:
            iqr_upper = q3 + 1.5 * iqr
        bound = iqr_upper if upper_bound is None else min(iqr_upper, upper_bound)
    elif method == "percentile":
        bound = s.quantile(p_hi) if upper_bound is None else min(s.quantile(p_hi), upper_bound)
    else:
        raise ValueError("method must be 'iqr' or 'percentile'")

    if lower_tail:
        outliers = s[s < bound]
    else:
        outliers = s[s > bound]

    return {
        "count": int(len(outliers)),
        "percent_col": float(len(outliers) / len(s) * 100),
        "percent_ds": float(len(outliers) / len(series) * 100),
        "iqr_upper": float(bound) if method == "iqr" else None,
        "bound_used": float(bound) if np.isfinite(bound) else None,
        "max_value": float(s.max()),
        "outlier_values": outliers.sort_values(ascending=False).head(10).astype(float).tolist()
    }

# Example usage
for col, ub in [("Electric Range", 350), ("Base MSRP", 200_000)]:
    if col in df.columns:
        print(f"\n{col} Outlier Check")
        print(find_outliers(df[col], upper_bound=ub, ignore_zeros=True, method="iqr"))

# In[17]:
# Remove all rows with MSRP values flagged as outliers
if "Base MSRP" in df.columns:
    df["Base MSRP"] = pd.to_numeric(df["Base MSRP"], errors="coerce")
    outlier_values_to_remove = [845000.0, 184400.0]
    df = df[~df["Base MSRP"].isin(outlier_values_to_remove)]

print(find_outliers(df["Base MSRP"], upper_bound=200_000))

# In[18]:
df.shape

# In[19]:
df.head()

# In[20]:
df.to_csv(r"C:\Users\Chin Zhi Yueh\Desktop\NYP Y3S1\EGT307\Project\team_zebraflights\data\clean_dataset.csv", index=False)
