from global_land_mask import globe
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

"""
    function euclideanDistance: Finding the distance between two coordinate points.
    param point1:               Point 1.
    param point2:               Point 2.
    return distance:            The euclidean distance between two points.
"""
def euclideanDistance(point1,point2):

    distance = np.linalg.norm(point1 - point2)
    return distance

"""
    function generateNRandomHours:  Generate N random hours.
    param N:                        Number of generated hours.
    return hours:                   Numpy array of generated hours.
"""
def generateNRandomHours(N=10):
    hours = np.array([])
    i=0
    for i in range(N):
        hours = np.append(hours, np.random.randint(24))
    return hours

"""
    function generateNRandomMonths: Generate N random months.
    param N:                        Number of generated months.
    return months:                  Numpy array of generated months.
"""
def generateNRandomMonths(N=10):
    monthList = [9, 10, 11, 12, 1, 2]
    months = np.array([])
    i=0
    for i in range(N):
        months = np.append(months, monthList[np.random.randint(len(monthList))])
        i+=1
    return months

"""
    function generateNRandomCoordinates:    Generate N random valid coordinates.
    param lon_min:                          Minimum value of longitude generated.
    param lon_max:                          Maximum value of longitude generated.
    param lat_min:                          Minimum value of latitude generated.
    param lat_max:                          Maximum value of latitude generated.
    param N:                                Number of generated coordinate points.
    return hours:                           Numpy array of generated coordinate points.
"""
def generateNRandomCoordinates(lon_min = -100, lon_max = 150, lat_min=-20, lat_max=20, N=10):
    coordinates = np.array([])
    i=0
    while i<N:
        lon = round(random.uniform(lon_min, lon_max), 6)
        lat = round(random.uniform(lat_min, lat_max), 6)
        if globe.is_land(lat, lon):
            coordinates = np.append(coordinates, np.array((lat, lon)))
            i+=1
            #print("Appending:", np.array((lat, lon)))
    coordinates = coordinates.reshape(-1, 2)
    return coordinates

"""
    function saveData:  Save dataframe as csv.
    param data:         Dataframe to be saved as csv.
    return 0:           No return.
"""
def saveData(data):
    data.to_csv('public/generated_trainV2.csv', index=False, mode='a', header=False)
    print("Successfully saved data.")
    return 0

"""
    function generateData:      Generate N data points.
    param train:                Training data to be used for neighbour sampling.
    param min_lat:              Minimum value of latitude of data point.
    param max_lat:              Maximum value of latitude of data point.
    param N:                    Number of generated data points.
    param K:                    Number of neighbours.
    return df_new_data:         Dataframe with N generated data points.
"""
def generateData(train, min_lat, max_lat, N=10, K=3):
    coordinates = generateNRandomCoordinates(-155, 155, min_lat, max_lat, N)
    months = generateNRandomMonths(N)
    hours = generateNRandomHours(N)
    print("Created ", N, " new point(s)")
    df_new_data = pd.DataFrame()
    for i in range(N):
        distCoord = []
        for count in range(len(train)):
            distCoord.append(euclideanDistance(coordinates[i], np.array((train.iloc[count]['fact_latitude'], train.iloc[count]['fact_longitude']))))
            # print("Coordinate loop: ", count)

        sortedDistCoord = np.argsort(distCoord)[:(K*99)]
        df_placeholder = train.iloc[sortedDistCoord]
        distMonth = []
        month = np.array((np.sin((months[i]-1) * (2.0 * np.pi / 12)), np.cos((months[i]-1) * (2.0 * np.pi / 12))))
        for count2 in range(len(sortedDistCoord)):
            distMonth.append(euclideanDistance(month, np.array((df_placeholder.iloc[count2]['month_sin'], df_placeholder.iloc[count2]['month_cos']))))
            # print("Month loop: ", count2)

        sortedDistMonth = np.argsort(distMonth)[:(K*49)]
        df_placeholder = df_placeholder.iloc[sortedDistMonth]
        distHour = []
        hour = np.array((np.sin(hours[i] * (2.0 * np.pi / 24)), np.cos(hours[i] * (2.0 * np.pi / 24))))
        for count3 in range(len(sortedDistMonth)):
            distHour.append(euclideanDistance(hour, np.array((df_placeholder.iloc[count3]['hour_sin'], df_placeholder.iloc[count3]['hour_cos']))))
            #print("Hour loop: ", count3)

        sortedDistFinal = np.argsort(distHour)[:K]
        df_placeholder = df_placeholder.iloc[sortedDistFinal].mean(axis=0).to_frame().T

        df_placeholder['fact_latitude'] = coordinates[i][0]
        df_placeholder['fact_longitude'] = coordinates[i][1]
        df_placeholder['month_sin'] = np.sin((months[i] - 1) * (2.0 * np.pi / 12))
        df_placeholder['month_cos'] = np.cos((months[i] - 1) * (2.0 * np.pi / 12))
        df_placeholder['hour_sin'] = np.sin(hours[i] * (2.0 * np.pi / 24))
        df_placeholder['hour_cos'] = np.cos(hours[i] * (2.0 * np.pi / 24))

        df_new_data = pd.concat([df_new_data, df_placeholder])
        if i%10 == 0:
            print("New data generated: ", i+1)

    return df_new_data

df_train = pd.read_csv('public/train_imputed.csv', low_memory=True)
df_test = pd.read_csv('public/test_imputed.csv', low_memory=True)
gen_train = pd.read_csv('public/generated_trainV2.csv', low_memory=True)

columns_to_drop = [
    "Unnamed: 0",
    "index",
    "fact_time",
    "cmc_available",
    "gfs_available",
    "gfs_soil_temperature_available",
    "wrf_available",
    "cmc_timedelta_s",
    "gfs_timedelta_s",
]

new_df_train = df_train.drop(columns_to_drop, axis=1)
new_df_train = new_df_train.dropna()
new_df_test = df_test.drop(columns_to_drop, axis=1)

X = new_df_train.drop('fact_temperature', axis=1)
y = new_df_train["fact_temperature"]

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, test_size=10000)

SMOTE_train = pd.concat([new_df_train, gen_train])

# Subsamples of training data in order to make the data generation faster.
SMOTE_train1 = SMOTE_train.loc[(SMOTE_train['fact_latitude'] >= -15) & (SMOTE_train['fact_latitude'] <= 0)]
SMOTE_train2 = SMOTE_train.loc[(SMOTE_train['fact_latitude'] >= 0) & (SMOTE_train['fact_latitude'] <= 15)]
SMOTE_train3 = SMOTE_train.loc[(SMOTE_train['fact_latitude'] >= -25) & (SMOTE_train['fact_latitude'] <= 25)]

# Generator of new data
try:
    while True:
        test = generateData(SMOTE_train1, -15, 0, 100, 5)
        saveData(test)
        test = generateData(SMOTE_train2, 0, 15, 100, 5)
        saveData(test)
        test = generateData(SMOTE_train3, -25, 25, 50, 5)
        saveData(test)
except KeyboardInterrupt:
    pass
