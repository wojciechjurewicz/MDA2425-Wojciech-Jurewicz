# Dataset Description - Bluebook for Bulldozers 

This repository contains code and documentation related to the **Bluebook for Bulldozers** dataset. The dataset was provided as part of a Kaggle competition and contains historical sales data for construction equipment, with the goal of predicting future sale prices.


## Dataset Overview

- **Dataset Name**: Bluebook for Bulldozers   
- **Source**: Kaggle competition – [Bluebook for Bulldozers]   
- **Columns**: Includes machine specifications, usage history, and sale-related details

## Dataset Summary

- **Total Rows**: 401,125  
- **Total Columns**: 53   
- **Target Column**: `SalePrice`  
- **Missing Values**: Present in many columns (e.g.,`MachineHoursCurrentMeter`, `Enclosure_Type`, `Blade_Type`, etc.)
  
## Example Columns
 
| Column Name                | Description                                         | Data Type                           |
|----------------------------|-----------------------------------------------------|-------------------------------------|
| `SalesID`                  | Unique identifier for the sale                      | int64                               |
| `SalePrice`                | **Target variable** - price the item was sold for   | float64                             |
| `MachineID`                | Unique identifier for the machine                   | int64                               |
| `ModelID`                  | ID of the machine model                             | int64                               |
| `YearMade`                 | Year the machine was manufactured                   | int64                               |
| `MachineHoursCurrentMeter` | Total machine operating hours at sale               | float64 (many missing)              |
| `UsageBand`                | Equipment usage band (e.g., Low, Medium, High)      | object                              |
| `saleDate`                 | Date of sale                                        | object (string, should be datetime) |
| `ProductGroup`             | Main product group category                         | object                              |
| `Drive_System`             | Drive type (e.g., 4WD, AWD)                         | object, many missing values         |
| `Enclosure`                | Cab type (e.g., OROPS, EROPS)                       | object                              |
| `Blade_Type`, `Steering_Controls`, etc. | Specific equipment configuration       | Many missing values                 |

## Missing Data

Many columns contain missing values. For example:
- `MachineHoursCurrentMeter` – only ~14k non-null out of 401k rows
- Equipment-specific columns (like `Blade_Type`, `Tip_Control`, `Hydraulics`, etc.) – high missingness

Handling missing data appropriately is critical before model training.   