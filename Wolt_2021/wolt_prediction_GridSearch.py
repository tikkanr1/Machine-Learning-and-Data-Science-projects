# Import required libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error

# Read the data
data = pd.read_csv('orders_autumn_2020.csv', index_col=0)

print(data)

# Change index column to datetime
data.index = pd.to_datetime(data.index)

# Add column to represent weekends to the dataset
data['WEEKEND'] = (data.index.weekday >= 5) * 1

# Helper method to plot correlation matrices
def corr_matrix(data, title):
    corr = data.corr()
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    plt.title(title)
    plt.show()

# Create a correlation matrix to find interesting correlations between columns
corr_matrix(data, 'Correlation matrix for data colummns')

# Helper function to calculate haversine distance between coordinates in km
def haversine_vectorize(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    newlon = lon2 - lon1
    newlat = lat2 - lat1
    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2
    dist = 2 * np.arcsin(np.sqrt(haver_formula ))
    km = 6367 * dist
    return km

# Add column to represent distance to the dataset
data['DISTANCE'] = haversine_vectorize(data['USER_LAT'], data['USER_LONG'], data['VENUE_LAT'], data['VENUE_LONG'])

# Count daily orders and averages
daily_average = data.copy()
daily_average = daily_average.drop(['ACTUAL_DELIVERY_MINUTES - ESTIMATED_DELIVERY_MINUTES',
                                    'ESTIMATED_DELIVERY_MINUTES', 'USER_LAT', 'USER_LONG',
                                    'VENUE_LAT', 'VENUE_LONG'], axis=1)
daily_average['ORDER_COUNT'] = 1
daily_average = daily_average.groupby(pd.Grouper(level='TIMESTAMP', freq="1D"))\
    .agg({'ITEM_COUNT': 'mean', 'ORDER_COUNT': 'sum', 'ACTUAL_DELIVERY_MINUTES': 'mean', 'CLOUD_COVERAGE': 'mean',
          'TEMPERATURE': 'mean', 'WIND_SPEED': 'mean', 'PRECIPITATION': 'mean', 'WEEKEND': 'mean', 'DISTANCE': 'mean'})

# Plot correlation between daily data
corr_matrix(daily_average, "Correlation matrix for daily averages")

# Helper function to highlight weekends
def highlight_weekends(dataframe):
    i = 0
    while i < len(dataframe):
        if(dataframe['WEEKEND'][i] == 1):
            plt.axvspan(dataframe.index[i], dataframe.index[i+1], facecolor='green', alpha=.15)
        i+=1

# Highlight weekends and plot graph
highlight_weekends(daily_average)
plt.title('Clear increase in orders during highlighted weekends')
plt.xlabel('date')
plt.ylabel('orders')
plt.xticks(rotation=45)
plt.plot(daily_average.index, daily_average['ORDER_COUNT'], label="orders")
plt.legend(loc="upper left")
plt.show()


# Plot graph to see correlation between orders and rain
fig, ax1 = plt.subplots()
plt.title('Slight correlation visible between orders and rainy days')
plt.xticks(rotation=45)
ax2 = ax1.twinx()
ax1.plot(daily_average.index, daily_average['ORDER_COUNT'], 'b-')
ax2.plot(daily_average.index, daily_average['PRECIPITATION'], 'g-')
ax1.set_ylabel('Orders', color='b')
ax2.set_ylabel('Precipitation', color='g')
plt.show()


# Predict future orders with linear model

# Define data features and labels and split the data
y = daily_average['ORDER_COUNT']
x = daily_average[['CLOUD_COVERAGE', 'TEMPERATURE', 'WIND_SPEED', 'PRECIPITATION', 'WEEKEND']]
#x = daily_average.drop('ORDER_COUNT', axis=1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, shuffle=False)

# Hyperparameter tuning using brute-force grid search

param_grid = {
        'n_estimators': [250, 500, 750, 1000],
        'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4],
        'max_depth': [3, 4, 6, 8],
        'min_child_weight': [2, 4, 6, 8, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
        }



# xgbregressor as model using all available threads
model = xgb.XGBRegressor(nthread=1)

# Grid Search using selected parameters and model for tuning
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    scoring='neg_mean_squared_error',
                    cv=3,
                    n_jobs=4,
                    verbose=3
                    )

# Fit training data
start = time.time()
grid_fit = grid.fit(xtrain, ytrain)
total = time.time() - start
print("Total time (Intel(R) Xeon(R) CPU E5-1650 v4 @ 3.60GHz): ", total)
#Total time:  1489.928561925888
#Training score:  -1.282776404628142
#MSE: 851.68

#{'colsample_bytree': 0.8, 'gamma': 5, 'learning_rate': 0.3, 'max_depth': 4, 'min_child_weight': 2, 'n_estimators': 250, 'subsample': 1.0}
#Best parameters: {'colsample_bytree': 0.8, 'gamma': 2, 'learning_rate': 0.01, 'max_depth': 6, 'min_child_weight': 2, 'n_estimators': 750, 'subsample': 1.0}
#MSE: 4383.69


# Print the training accuracy score
score = grid_fit.score(xtrain, ytrain)
print("Training score: ", score)

# Make a prediction
pred = grid_fit.predict(xtest)

# Print MSE
mse = mean_squared_error(ytest, pred)
print("MSE: %.2f" % mse)

# Print best parameters
#print('\n Best parameters:', grid_fit.best_params_)

# Plot results
x_len = range(len(ytest))
plt.title("Comparison")
plt.xticks(rotation=45)
plt.plot(ytest.index, ytest, label="original")
plt.plot(ytest.index, pred, label="predicted")
plt.legend()
plt.show()
