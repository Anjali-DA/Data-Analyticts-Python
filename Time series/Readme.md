# Welcome to Time series analysis
## Aim of this analysis
- Checking the dataset for time series is stationary or non-stationary
- PLotting time series with **acf & pacf**
- Split the dataset to train & test proceed with further analysis

## How did I approcahed to solve this Time series dataset
- Imported the libraries: the most important library is **adfuller** as it is used for checking the stationary an non stationary in time series.
- Dropped the columns as those had the NAN values, which would affect the Time series analysis: *LandMaxTemperature	LandMaxTemperatureUncertainty	LandMinTemperature	LandMinTemperatureUncertainty	LandAndOceanAverageTemperature	LandAndOceanAverageTemperatureUncertainty*
- Converted the column 'dt' to datetime followed by filling up the missing values of columns *LandAverageTemperature & LandAverageTemperatureUncertainty*
- Plotted the LandAverageTemperature by using the matplotlib library. By looking the plot, it can be concluded the plot is stationary- means that the dataset does not vary with seasonal trends.
- Mathematically to check the stationary and non-stationary in time series from year 2005-2015, I used adfuller(ADF) to evaluate the p-value. After running the code the p-value I got **0.0012331195770598107** which is lower than 0.05, thus it rejects the null hypothesis and is in stationary.
- As it is a seasonal data, I did 12 times differencing using **shift(12)**. After the new column of *LandAverageTemperature_12difference*,I used adfuller to find the p-value which is **3.692614581683388e-29** and plotted the correlations by ACF and PACF.
- Created the train & test datasets with datetime function form year 2005 to 2015 and used ARIMA function on these datasets.

## What I concluded from this time series forecasting
- Predicted the datasets with ARIMA & SARIMAX models and plotted the graph to find out the correlation given by these models.
