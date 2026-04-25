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
