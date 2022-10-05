# Project2-McDonald-s-diet-analysis
## 🖥️About this Dataset
**Context**

McDonald’s in India was started in 1996 in Bandra, Mumbai with one single restaurant. It took almost twelve years to grow one restaurant to 50. Today, McDonald's, that arrived in India without its signature Big Mac (substituted in India by the Maharaja Mac) has about 480 stores all over India providing happy meals to folks and families of India.

**Content**

This dataset provides a nutrition analysis of every menu item on the Indian McDonald's menu, including breakfast, burgeres, fries, salads, soda, coffee and tea, milkshakes, and desserts.

**Acknowledgements**

The menu items and nutrition facts were scraped from the McDonald's website

## 🖥️What I analyzed from this dataset
- Which type pf menu are mostly consumed by the customers?
- How does each nutritions in the menu varie with each other? How these nutritions will affect the health?
- Can a heart or diabetic patient consume these items? If so what type of items they will buy?
- For a normal working person, Is it good to consume meal to have balanced diet?

## 🖥️How I approched to solve this dataset
**1.** Imported libraries for data wrangling & data visualization.
``` Python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
```
**2.** Loaded the dataset with pandas with *read_csv().*

**3.** Checked the missing values in the datasets with *isnull.sum()*, and found one missing value in the column **'Sodium'**. With function **fillna()**, I used method called 'Pad' to fillin the single value in column **'Sodium'**.

(**pad**: *This method is similar to the DataFrame. fillna() method and it fills NA/NaN values using the ffill() method. It returns the DataFrame object with missing values filled or None if inplace=True.*)

**4.** 
