sequnce of data points collected at equally spaced time intervals 
has temporal dependency 

# components 
trend 
seasonality 
cyclic 
noise 

## ARIMA- AUTO REGRESSIVE INTEGRATED MOVING AVERAGE 
AR-USES PAST VALUES 
I- REMOVE TREND 
MA- ERROR CORRCTION 

## SARIMA (Seasonal ARIMA)

Extension of ARIMA that also handles seasonality.

Best for data with strong seasonal effects (e.g., monthly electricity consumption).

## Prophet (by Facebook)

Easy-to-use forecasting library.

Handles trend + seasonality + holidays automatically.

Great for business forecasting (sales, traffic).

## LSTMs for Time Series

Deep learning approach.

Works well for non-linear & complex time dependencies.

Used in financial forecasting, IoT sensor prediction, etc.

