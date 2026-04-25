#!/usr/bin/env python3
"""
BANA290 Assignment 3 — Rust Belt Revival: AI Training Subsidy
Difference-in-Differences Analysis (2018–2025)
"""

# ======================================================================
# PHASE 1 — SCRAPE
# ======================================================================

# --- Use BeautifulSoup to scrape the main index page and extract every
#     href link that points to a regional employment brief ---

import re
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.formula.api as smf
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

BASE_URL = "https://bana290-assignment3.netlify.app"
INDEX_URL = f"{BASE_URL}/"

index_resp = requests.get(INDEX_URL)
index_soup = BeautifulSoup(index_resp.text, "html.parser")

brief_urls = []
for tag in index_soup.find_all("a", href=True):
    href = tag["href"]
    if href.startswith("/briefs/"):
        full_url = BASE_URL + href
        if full_url not in brief_urls:
            brief_urls.append(full_url)

print(f"Discovered {len(brief_urls)} regional brief(s):")
for url in brief_urls:
    print(f"  {url}")

# --- Visit each brief URL, parse its HTML employment table into a
#     Pandas DataFrame, tag with the source URL, and append to a list ---

YEAR_COLS = [str(y) for y in range(2018, 2026)]
raw_frames = []

for url in brief_urls:
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if table is None:
        print(f"  WARNING: No <table> found at {url}")
        continue

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = [
        [td.get_text(strip=True) for td in tr.find_all("td")]
        for tr in table.find_all("tr")[1:]
        if tr.find_all("td")
    ]
    if not rows:
        print(f"  WARNING: Empty table at {url}")
        continue

    df = pd.DataFrame(rows, columns=headers[: len(rows[0])])
    df["_source"] = url
    raw_frames.append(df)
    print(f"  Scraped {len(df)} row(s) from /{url.split('/')[-1]}")

# --- Combine all scraped DataFrames into one raw master table ---

raw_df = pd.concat(raw_frames, ignore_index=True)
print(f"\nRaw combined shape: {raw_df.shape}")
print(raw_df.to_string())


# ======================================================================
# PHASE 2 — CLEAN
# ======================================================================

# --- Normalize heterogeneous column names across brief pages into a
#     single unified schema before any further processing ---

COL_RENAME = {
    "REGION":          "region_raw",
    "County":          "region_raw",
    "STATE_GROUP":     "state_group",
    "District":        "state_group",
    "Region":          "district_note",   # Penn-South uses 'Region' for district
    "PROGRAM_STATUS":  "program_status",
    "Status":          "program_status",
    "ANCHOR_INDUSTRY": "anchor_industry",
    "Industry":        "anchor_industry",
    "PORTAL_NOTE":     "_note",
    "Note":            "_note",
}

clean_df = raw_df.rename(columns=COL_RENAME)

# --- Parse employment strings that mix formats such as "~30.9k",
#     "31.4 thousand", "16,821 jobs", and "34,600" into integers ---

def parse_employment(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    s = s.replace("~", "").replace(",", "").strip()
    s = re.sub(r"\s*jobs\b", "", s, flags=re.IGNORECASE).strip()
    if re.search(r"[kK]$", s):
        return int(round(float(re.sub(r"[kK]$", "", s)) * 1_000))
    if re.search(r"\bthousand\b", s, flags=re.IGNORECASE):
        return int(round(
            float(re.sub(r"\s*thousand\b", "", s, flags=re.IGNORECASE).strip()) * 1_000
        ))
    return int(float(s))

for yr in YEAR_COLS:
    if yr in clean_df.columns:
        clean_df[yr] = clean_df[yr].apply(parse_employment)

# --- Standardize region names to canonical "County, ST" format; fall
#     back to anchor-industry matching for pages that store district
#     names instead of county names in the region column ---

REGION_PATTERNS = {
    r"lucas":    "Lucas County, OH",
    r"stark":    "Stark County, OH",
    r"mahoning": "Mahoning County, OH",
    r"trumbull": "Trumbull County, OH",
    r"erie":     "Erie County, PA",
    r"mercer":   "Mercer County, PA",
    r"lawrence": "Lawrence County, PA",
    r"beaver":   "Beaver County, PA",
}

INDUSTRY_FALLBACK = {
    r"machine shop": "Erie County, PA",
    r"plastic":      "Mercer County, PA",
    r"warehouse":    "Lawrence County, PA",
    r"chemical":     "Beaver County, PA",
}

def normalize_region(row):
    raw = str(row.get("region_raw", "")).lower()
    for pattern, canonical in REGION_PATTERNS.items():
        if re.search(pattern, raw, re.IGNORECASE):
            return canonical
    industry = str(row.get("anchor_industry", "")).lower()
    for pattern, canonical in INDUSTRY_FALLBACK.items():
        if re.search(pattern, industry, re.IGNORECASE):
            return canonical
    return None

clean_df["region"] = clean_df.apply(normalize_region, axis=1)

unresolved = clean_df[clean_df["region"].isna()]
if not unresolved.empty:
    print(f"\nWARNING: {len(unresolved)} row(s) unresolved:")
    print(unresolved[["region_raw", "anchor_industry", "_source"]].to_string())

# --- Reshape from wide format (year columns) to long format so each
#     row represents one county-year observation ---

id_vars = [c for c in ["region", "program_status", "anchor_industry", "_source"]
           if c in clean_df.columns]

long_df = clean_df.melt(
    id_vars=id_vars,
    value_vars=[yr for yr in YEAR_COLS if yr in clean_df.columns],
    var_name="year",
    value_name="employment",
)
long_df["year"] = long_df["year"].astype(int)

# --- Add TREATED and POST_POLICY dummy variables required for DID
#     TREATED    = 1 for Ohio counties (AI subsidy recipients)
#     POST_POLICY = 1 for 2022 onward (year the subsidy was rolled out) ---

TREATED_COUNTIES = {
    "Lucas County, OH", "Stark County, OH",
    "Mahoning County, OH", "Trumbull County, OH",
}

long_df["TREATED"]      = long_df["region"].isin(TREATED_COUNTIES).astype(int)
long_df["POST_POLICY"]  = (long_df["year"] >= 2022).astype(int)
long_df["DID"]          = long_df["TREATED"] * long_df["POST_POLICY"]

long_df = (
    long_df
    .dropna(subset=["employment", "region"])
    .sort_values(["region", "year"])
    .reset_index(drop=True)
)

print(f"\nClean panel shape: {long_df.shape}")
print(long_df[["region", "year", "employment", "TREATED", "POST_POLICY"]].to_string())
