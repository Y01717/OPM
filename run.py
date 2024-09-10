import numpy as np
import pandas as pd
import os
from keras.utils import set_random_seed
from sklearn.preprocessing import StandardScaler
from model import GCNForecast, load_data, train, test
from tools import euclidean_distance, build_adjacency_matrix
import torch
import torch.nn as nn



df = pd.read_csv('/home/xus23004/Project/OPM/Data/WRF4.2.2_4km_calibrationFile_Eversource_Rainwind_294events_2023-11-17_CT 12.18.32 PM.csv')
df.head()

# set the group label for each data
df = df.iloc[:236350, :]

event_codes = df["eventCode"]
group_labels = np.zeros(len(df), dtype=int)
current_label = 0
previous_code = None
for i, code in enumerate(event_codes):
    if code != previous_code:
        current_label += 1
        previous_code = code
    group_labels[i] = current_label

unique_codes, unique_indices = np.unique(event_codes, return_index=True)
# for code, index in zip(unique_codes, unique_indices):
#     print(f"Group Label: {group_labels[index]}, Event Code: {code}")
df["groupLabel"] = group_labels


# get the actual counts for each event
y = df["countts"].copy()
unique_test_groups = np.unique(group_labels)
actual_counts = np.zeros(len(unique_test_groups)+1)

for group in unique_test_groups:
    group_actual = y[group_labels == group]
    actual_counts[group] = np.int32(np.sum(group_actual))

coordinates = np.array(df[["lonwrf", "latwrf"]][0:815])

threshold = 0.065 # euclidean distance!!!!!

adjacency_matrix = build_adjacency_matrix(coordinates, threshold)
print(adjacency_matrix)

y = df["countts"].copy()
X = df[["sumAFWA_CAPE", "stdAFWA_CAPE", "maxAFWA_CAPE", "stdW850", "sumSSRUN",
        "peakW850", "peakSSRUN", "stdSMOIS4", "peakAFWA_LLWS", "peakGUST", "maxGUST",
        "avgAFWA_RAIN", "stdGUST", "maxW850", "sumAFWA_TOTPRECIP", "maxSSRUN",
        "stdAFWA_RAIN", "stdAFWA_TOTPRECIP", "maxQ2", "sumQ2", "ggt22", "stdSSRUN",
        "maxAFWA_RAIN", "maxAFWA_TOTPRECIP", "stdSMOIS3", "peakQ2", "avgTH2", "sumT2",
        "stdWSPD10MAX", "sumUP_HELI_MAX", "stdAFWA_LLWS", "maxDPT2", "avgDPT2",
        "coggt22", "maxTH2", "LAI", "minT2", "minQ2", "maxWSPD10MAX", "stdPBLH",
        "maxSR", "stdTDIF", "avgHardSDI", "avgHardBA", "minPSFC", "latwrf",
        "maxAFWA_MSLP", "sumAFWA_MSLP", "peakAFWA_MSLP", "minAFWA_MSLP"]]

num_attribs = ["sumAFWA_CAPE", "stdAFWA_CAPE", "maxAFWA_CAPE", "stdW850", "sumSSRUN",
        "peakW850", "peakSSRUN", "stdSMOIS4", "peakAFWA_LLWS", "peakGUST", "maxGUST",
        "avgAFWA_RAIN", "stdGUST", "maxW850", "sumAFWA_TOTPRECIP", "maxSSRUN",
        "stdAFWA_RAIN", "stdAFWA_TOTPRECIP", "maxQ2", "sumQ2", "ggt22", "stdSSRUN",
        "maxAFWA_RAIN", "maxAFWA_TOTPRECIP", "stdSMOIS3", "peakQ2", "avgTH2", "sumT2",
        "stdWSPD10MAX", "sumUP_HELI_MAX", "stdAFWA_LLWS", "maxDPT2", "avgDPT2",
        "coggt22", "maxTH2", "LAI", "minT2", "minQ2", "maxWSPD10MAX", "stdPBLH",
        "maxSR", "stdTDIF", "avgHardSDI", "avgHardBA", "minPSFC", "latwrf",
        "maxAFWA_MSLP", "sumAFWA_MSLP", "peakAFWA_MSLP", "minAFWA_MSLP"]

X = X.fillna(0)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(290, 815, 50)

# Load and preprocess the data
feature_matrix = X  # Load feature matrix (290x815x50)
labels = y  # Load node labels for forecasting (290x815)

x, edge_index, y = load_data(adjacency_matrix, feature_matrix, labels)

# Split the data into training and testing sets
train_size = 200
train_x, test_x = x[:train_size], x[train_size:]
train_y, test_y = y[:train_size], y[train_size:]

# Define the model and optimizer
model = GCNForecast(in_channels=50, hidden_channels=64, out_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.L1Loss()

print(train_x.shape)
print(train_y.shape)


for epoch in range(200):
    loss = train(model, train_x, edge_index, train_y, optimizer, criterion)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

test_loss = test(model, test_x, edge_index, test_y, criterion)
print(f"Test Loss: {test_loss:.4f}")