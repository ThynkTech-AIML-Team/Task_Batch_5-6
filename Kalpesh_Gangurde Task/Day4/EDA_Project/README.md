# Titanic EDA Project

## Overview
Exploratory Data Analysis of the Titanic dataset - 100 passengers, analyzing survival patterns and key factors.

## Dataset
- **File:** titanic.csv
- **Records:** 100 passengers
- **Target:** Survived (0=No, 1=Yes)
- **Features:** PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Embarked

## Tools Used
- pandas, numpy, matplotlib, seaborn, jupyter

## What's Included
1. **Titanic_EDA.ipynb** - Complete analysis notebook
2. **titanic.csv** - Dataset (100 records)
3. **README.md** - This file

## EDA Steps
1. Load & explore data (shape, info, describe)
2. Handle missing values (Age → median, Embarked → mode, drop Cabin)
3. Create features (Title from Name, FamilySize, IsAlone)
4. Visualize data (histograms, boxplots, heatmap, bar charts)
5. Extract insights

## Quick Start
```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook
# Open Titanic_EDA.ipynb and run cells
```

## Key Findings
- **Gender:** Female survival ~86%, Male ~19%
- **Class:** 1st class ~63%, 2nd class ~41%, 3rd class ~19.5% survival
- **Age:** Children had higher survival rates
- **Family Size:** Small families survived better
- **Fare:** Higher fares → better survival chances

## Feature Engineering
| Original | New Feature | Reason |
|----------|-------------|--------|
| Name | Title | Extract social status |
| SibSp, Parch | FamilySize | Combine family info |
| FamilySize | IsAlone | Binary family indicator |

## Missing Values Handled
| Column | Missing | Method | Result |
|--------|---------|--------|--------|
| Age | 3 | Median (29.5) | ✓ Fixed |
| Embarked | 2 | Mode (S) | ✓ Fixed |
| Cabin | 76 | Dropped | ✓ Removed |

## Insights
✅ Gender was the strongest survival factor  
✅ Passenger class significantly affected survival  
✅ Younger passengers had better chances  
✅ Family structure mattered for survival  
✅ "Women and children first" policy evident in data  

