# ✏️Data-Science-Job-analysis

## 🖥️About the dataset
![data science](https://user-images.githubusercontent.com/98269318/190690079-59799c12-2afc-4d78-9601-e17af8160ad5.png)

### 🖥️What to analyse from this dataset
  - Count of employment type with respect to work year.

  - Percentage of employment type.
  
  - Distribution of exprience level in terms of percentage.
  
  - Variation of remote ratio with respect to work year.
   
  - Total count of top 10 job titles.
  
  - Total count of top 10 company residence.
  
  - Total count of top 10 employee residence.
  
  - Relation between company locarion and employee residence with respect to work type.
  
  - Variation of job roles with respect to company size.
  
  - Variation of salary with company size and work type.
  
## 🖥️How I approched to solve this dataset
**1.** Imported libraries
``` python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
```
**2.** Loaded the dataset

**3.** Cleaned the data (Data Wrangling)
- Calculated total null values and it's data type

- Replaced **0,50,&100** with **'Onsite','Hybrid',&,'Remote'**

- Replaced **'MI','SE','EN'&,'EX'** with **'Intermediate','Expert','Junior',& 'Director'**

**4.** Started with the visualization (EDA)
- For visualization libraries I used **plotly.express**
- With plotly.express, I plotted histogram, treemap, barplot,& scatterplot.

## 🖥️What I concluded after solving this dataset 
- In **year 2022**, the employees worked more in **full-time** position.
- **96.87%** employees work for the full-time position.
-  Most employees are employed as a **senior level/expert**
-  Due to COVID-19, most employees **worked from home (remote)** & the demand for data analyst/science increased for research purposes like medicenes,start-up, environment, etc, so the count for work type-remote increased every year (2020,2021 & 2022)
-  Most employees work as **data scientist & data engineer**
-  Most companies are located in **US (United States)**
-  Most employees reside in **US (United States)**
-  Most of the **remote** employees work from **different countries**
-  Company of **medium size**, have the most job roles like **Data scientist, Data Analyst & Data engineer**
-  Due to covid pandamic, most employees started working work from home(remote) also **small companies** were providing employees higher salary compare to **large company**

## To view complete data visualization of Data science jobs- [Click Here](https://www.kaggle.com/code/pantanjali/data-science-job-analysis)
