import numpy as np
import pandas as pd
import pickle

with open("./data/games/game_df_0021500502.pkl", "rb") as f:
	tracking_df = pickle.load(f)

tracking_df["game_clock_min"] = tracking_df["game_clock"].astype(int) % 3600 // 60
tracking_df["game_clock_sec"] = tracking_df["game_clock"].astype(int) % 60
timestring = "{:01d}:{:02d}"
tracking_df["game_clock_timestring"] = tracking_df.apply(
	lambda r: timestring.format(r['game_clock_min'], r['game_clock_sec']),
	axis = 1)

tracking_df["quarter"] = tracking_df["quarter"].astype(int)


q1 = list(tracking_df[tracking_df["quarter"] == 1]["game_clock"])
q1 = [round(i, 2) for i in q1]

sets = []
set_i = []
for i in range(1, len(q1)):

	
	# if q1[i] == round(q1[i-1]-0.04, 2):
	if q1[i-1] - q1[i] <= 0.05:
		set_i.append(q1[i-1])
		set_i.append(q1[i])
	else:
		set_i = list(set(set_i))
		set_i.sort(reverse = True)
		sets.append(set_i)
		set_i = []

# print(tracking_df[(tracking_df["quarter"] == 1) & (tracking_df["game_clock"] <= 4.26)])
print(tracking_df[tracking_df["moment_or_event_idk"] == 105])