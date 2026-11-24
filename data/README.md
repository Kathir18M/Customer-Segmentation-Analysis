# Dataset Documentation: Online Retail

## Overview
The dataset used in this project is the **Online Retail** dataset from the UCI Machine Learning Repository.

- **Dataset Name:** Online Retail
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)
- **DOI:** 10.24432/C5BW33
- **Business Domain:** UK-based online gifts and homeware retail
- **Transaction Period:** 01/12/2010 to 09/12/2011 (December 1, 2010 to December 9, 2011)
- **Currency:** GBP (£)

## Raw Data Details
- **File:** `Online Retail.xlsx`
- **Location:** `data/raw/` (ignored by Git)
- **Original Dimensions:** 541,909 rows × 8 columns
- **Original Columns:**
  1. `InvoiceNo` (object): 6-digit transaction identifier. Invoices beginning with 'C' indicate cancellations.
  2. `StockCode` (object): 5-digit product identifier code.
  3. `Description` (object): Product description name.
  4. `Quantity` (int64): The quantity of each product ordered per transaction.
  5. `InvoiceDate` (datetime64[ns]): Transaction timestamp.
  6. `UnitPrice` (float64): Product price per unit in Sterling (£).
  7. `CustomerID` (float64): 5-digit integral customer identifier.
  8. `Country` (object): Name of the country where the customer resides.

## Data Directories
- `data/raw/`: Contains the original unedited `Online Retail.xlsx` dataset. This folder is gitignored to keep repository sizes small.
- `data/processed/`: Contains intermediate processed outputs or cached parquet/feather/csv representations if needed.

## Download Instructions
If `Online Retail.xlsx` is missing, download the dataset archive from:
`https://archive.ics.uci.edu/static/public/352/online+retail.zip`
and extract `Online Retail.xlsx` directly into `data/raw/`.
