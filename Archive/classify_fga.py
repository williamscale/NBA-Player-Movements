import numpy as np
import pandas as pd
import pickle

with open("./data/tracking_pbp_0021500502.pkl", "rb") as f:
	tracking_df = pickle.load(f)

# Seems to be missing at least the first fga of each quarter.
# Doesn't appear to be in tracking data.
# Issue appears to be that tracking data doesn't exist or Game.py isn't capturing it.
# I tracked back all the way through extract_data.py.

# tracking_df["fga_id"] = np.where(
# 	tracking_df["fga_flag"] == 1,
# 	tracking_df.groupby(["quarter", "game_clock_timestring"], sort = False).ngroup(),
# 	None)

tracking_fga_only = tracking_df[tracking_df["fga_flag"] == 1].copy()
tracking_fga_only["fga_id"] = tracking_fga_only.groupby(["quarter", "game_clock_timestring"], sort = False).ngroup()
tracking_fga_only = tracking_fga_only[["quarter", "game_clock_timestring", "fga_id"]].drop_duplicates()

# now calculate trajectory of ball by taking last and first of each group
# need to get rows without fga now to compare?

# def calculate_ball_trajectory(time, ball_x, ball_y):

# 	delta_time = 

tracking_fga = pd.merge(
	tracking_df,
	tracking_fga_only,
	how = "left",
	on = ["quarter", "game_clock_timestring"]
	)
tracking_fga = tracking_fga.replace({np.nan: None})

# are there consecutive intervals in which the ball is travelling towards the hoop?
# the more there are, the higher the probability of it being a shot?
# at some range of speed?
# create variable of angle relative to hoop or something
# create variable of "was previous trajectory going towards hoop"

# split to only look at basket1 for now
# do it after ball speed is calculated though or that will be wrong from skipping datapoints

# def calculate_distance(x1, y1, x2, y2):
# 	return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2)

tracking_fga = tracking_fga.assign(
	ball_distance = lambda x: ((x["ball_x"] - x["ball_x"].shift(1)) ** 2 + (x["ball_y"] - x["ball_y"].shift(1)) ** 2) ** (1 / 2),
	delta_time = lambda x: x["game_clock"].shift(1) - x["game_clock"],
	ball_speed = lambda x: x["ball_distance"] / x["delta_time"]
	)

q2_idx = tracking_fga[tracking_fga["quarter"] == 2].index[0]
q3_idx = tracking_fga[tracking_fga["quarter"] == 3].index[0]
q4_idx = tracking_fga[tracking_fga["quarter"] == 4].index[0]

tracking_fga.loc[[0, q2_idx, q3_idx, q4_idx], ["ball_distance", "delta_time", "ball_speed"]] = None

tracking_fga = tracking_fga[tracking_fga["ball_speed"] <= 100]

basket1 = np.array([4.75, 25])
basket2 = np.array([94 - 4.75, 25])

# tracking_fga["ball_arr"] = np.array([tracking_fga["ball_x"], tracking_fga["ball_y"]])

tracking_fga_b1 = tracking_fga[tracking_fga["ball_x"] <= 47].copy()
tracking_fga_b2 = tracking_fga[tracking_fga["ball_x"] > 47].copy()

# tracking_fga_b1 = tracking_fga_b1.assign(
# 	fga_theta = lambda x: 
# 	# np.degrees(
# 		# np.arccos(
# 			# np.dot(
# 			# 	np.array([x["ball_x"] - basket1[0], x["ball_y"] - basket1[1]]),
# 			# 	np.array([x["ball_x"] - x["ball_x"].shift(1), x["ball_y"] - x["ball_y"].shift(1)])
# 			# 	)
# 			# / 
# 			# (np.linalg.norm(np.array([x["ball_x"] - basket1[0], x["ball_y"] - basket1[1]]))
# 				# * 
# 				np.linalg.norm(np.array([x["ball_x"] - x["ball_x"].shift(1), x["ball_y"] - x["ball_y"].shift(1)]))
# 				# )
# 			# )
# 		# )
# 	# fga_theta = lambda x: np.linalg.norm(np.array([x["ball_x"]-basket1[0], x["ball_y"]-basket1[1]]))
# 	# fga_theta = lambda x: np.linalg.norm(np.array([x["ball_x"], x["ball_y"]]))
# 	# traj_opt = lambda x: np.array([x["ball_x"]-basket1[0], x["ball_y"]-basket1[1]])
# 	# traj_actual = lambda x: np.array([x["ball_x"], x["ball_y"]]) - np.array([x["ball_x"].shift(1), x["ball_y"].shift(1)]),
# 	# fga_theta = lambda x: np.degrees(np.arccos(np.dot(x["traj_opt"], x["traj_actual"]) / (np.linalg.norm(x["traj_opt"]) * np.linalg.norm(x["traj_actual"]))))
# 	)

# # tracking_fga_b1.loc[[0, q2_idx, q3_idx, q4_idx], ["traj_opt", "traj_actual", "fga_theta"]] = None
# print(tracking_fga_b1)
# xyz

def calculate_angle(p1, p2, p3):

	# p2 is basket
	optimal = p1 - p2
	fga = p1 - p3
	# fga = p3 - p2

	cosine_angle = np.dot(optimal, fga) / (np.linalg.norm(optimal) * np.linalg.norm(fga))
	angle_rad = np.arccos(cosine_angle)

	return angle_rad, np.degrees(angle_rad)

theta_rad = []
theta_deg = []

for i in range(1, tracking_fga_b1.shape[0]):
	# print(i)
	if i in [q2_idx, q3_idx, q4_idx]:
		theta_rad.append(1.5)
		theta_deg.append(90)
	else:
		theta_rad_i, theta_deg_i = calculate_angle(
			p1 = np.array([tracking_fga_b1.iloc[(i - 1), 3], tracking_fga_b1.iloc[(i - 1), 4]]),
			p2 = basket1,
			p3 = np.array([tracking_fga_b1.iloc[i, 3], tracking_fga_b1.iloc[i, 4]])
			)
		theta_rad.append(theta_rad_i)
		theta_deg.append(theta_deg_i)

theta_rad.insert(0, 1.5)
theta_deg.insert(0, 90)
# tracking_fga_b1["shot_theta_rad"] = theta_rad
tracking_fga_b1["fga_theta"] = theta_deg

tracking_fga_b1["fga_theta_diff"] = abs(tracking_fga_b1["fga_theta"] - tracking_fga_b1["fga_theta"].shift(1))

# print(tracking_fga_b1["fga_theta"].min(), tracking_fga_b1["fga_theta"].max())

tracking_fga_b1["previous_fga_theta_flag"] = np.where(
	(tracking_fga_b1["fga_theta"].shift(1) <= 2) & (tracking_fga_b1["fga_theta"] <= 2),
	1, 0
	)

# print(tracking_fga_b1[tracking_fga_b1["fga_theta"] <= 2])
test = tracking_fga_b1[(tracking_fga_b1["quarter"] == 1) &
	(tracking_fga_b1["game_clock"] >= 415) & (tracking_fga_b1["game_clock"] <= 440)]

print(test[["game_clock", "game_clock_timestring", "PCTIMESTRING"]])

# classify change of direction?
# create variable of difference in trajectory?
# create variable of who has possession
# offense vs defense.
# clustering model with predictors as
# possessing player, previous possessing player,
# ballspeed, time, change of direction?
# that will cluster passes and shots?
# 

# import matplotlib.pyplot as plt
# plt.hist(tracking_fga_b1["fga_theta"])
# plt.show()
# plt.hist(tracking_fga_b1["fga_theta_diff"])
# plt.show()

import matplotlib.pyplot as plt
plt.scatter(
	x = test["ball_x"],
	y = test["ball_y"]
	)
plt.scatter(x = basket1[0], y = basket1[1], color = "red")
plt.show()