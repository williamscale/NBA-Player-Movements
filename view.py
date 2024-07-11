import numpy as np
import pandas as pd
import pickle

with open("./data/events/events_arr_0021500502.pkl", "rb") as f:
	events_list = pickle.load(f)


variables = [
	"game_clock", "shot_clock",
	"ball_x", "ball_y",
	"player1_team", "player2_team", "player3_team", "player4_team", "player5_team", 
	"player6_team", "player7_team", "player8_team", "player9_team", "player10_team",
	"player1_id", "player2_id", "player3_id", "player4_id", "player5_id",
	"player6_id", "player7_id", "player8_id", "player9_id", "player10_id",
	"player1_x", "player2_x", "player3_x", "player4_x", "player5_x", 
	"player6_x", "player7_x", "player8_x", "player9_x", "player10_x",
	"player1_y", "player2_y", "player3_y", "player4_y", "player5_y",
	"player6_y", "player7_y", "player8_y", "player9_y", "player10_y"
	]

player_x_vars = list(range(24, 33 + 1))
player_y_vars = list(range(34, 43 + 1))

def calculate_distance(x1, y1, x2, y2):
	d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2)
	return d.reshape(x1.shape[0], 1)
	# return np.linalg.norm()

for i in range(0, len(events_list)):

	events_i = events_list[i]
	if events_i.shape[1] != 1:
		for j in range(0, len(player_x_vars)):
			dist_i = calculate_distance(
				x1 = events_i[:, 2],
				y1 = events_i[:, 3],
				x2 = events_i[:, player_x_vars[j]],
				y2 = events_i[:, player_y_vars[j]]
				)

			events_i = np.append(events_i, dist_i, axis = 1)
		events_i = np.append(
			events_i,
			np.full((events_i.shape[0], 1), i),
			axis = 1
			)
		events_list[i] = events_i

variables.extend([
	"player1_d", "player2_d", "player3_d", "player4_d", "player5_d",
	"player6_d", "player7_d", "player8_d", "player9_d", "player10_d",
	"moment_or_event_idk"
	])


events_list_full = [i for i in events_list if i.shape[1] != 1]
events_arr = np.concatenate(events_list_full, axis = 0)
events_df = pd.DataFrame(events_arr, columns = variables)

print(events_df)