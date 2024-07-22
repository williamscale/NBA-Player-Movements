import numpy as np
import pandas as pd
import pickle

with open("./data/events/events_arr_0021500502.pkl", "rb") as f:
	events_list = pickle.load(f)

def calculate_distance(x1, y1, x2, y2):
	d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2)
	return d.reshape(x1.shape[0], 1)

def combine_game(events_list):

	variables = [
		"quarter", "game_clock", "shot_clock",
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

	player_x_vars = list(range(25, 34 + 1))
	player_y_vars = list(range(35, 44 + 1))

	for i in range(0, len(events_list)):

		events_i = events_list[i]

		if events_i.shape[1] != 1:
			# for j in range(0, len(player_x_vars)):
			# 	dist_i = calculate_distance(
			# 		x1 = events_i[:, 3],
			# 		y1 = events_i[:, 4],
			# 		x2 = events_i[:, player_x_vars[j]],
			# 		y2 = events_i[:, player_y_vars[j]]
			# 		)

			# 	events_i = np.append(events_i, dist_i, axis = 1)
			events_i = np.append(
				events_i,
				np.full((events_i.shape[0], 1), i),
				axis = 1
				)
			events_list[i] = events_i

	variables.extend([
		# "player1_d", "player2_d", "player3_d", "player4_d", "player5_d",
		# "player6_d", "player7_d", "player8_d", "player9_d", "player10_d",
		"moment_or_event_idk"
		])


	events_list_full = [i for i in events_list if i.shape[1] != 1]
	events_arr = np.concatenate(events_list_full, axis = 0)
	events_df = pd.DataFrame(events_arr, columns = variables)

	# Some moments/events are overlapping.
	# See quarter==4 & game_clock==679.
	events_df.drop_duplicates(
		subset = ["quarter", "game_clock"],
		keep = "last",
		inplace = True
		)
	events_df.sort_values(
		by = ["quarter", "game_clock"],
		ascending = [True, False],
		inplace = True,
		ignore_index = True
		)

	return events_df

game_df = combine_game(events_list)

# print(game_df[game_df.duplicated(subset=['quarter','game_clock'], keep=False)].sort_values(by=['quarter','game_clock']))
# print(game_df[(game_df["quarter"]==4) & (game_df["game_clock"]== 679)])
# with open("./data/games/game_arr_0021500502" + ".pkl", "wb") as f:
# 	pickle.dump(game_arr, f)

# print(game_df)

with open("./data/games/game_df_0021500502" + ".pkl", "wb") as f:
	pickle.dump(game_df, f)

