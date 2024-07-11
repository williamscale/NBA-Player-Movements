import numpy as np
import pandas as pd
import pickle
from Game import Game

# g = Game(path_to_json = "0021500502.json", event_index = 350)
# g.read_json()

def unnest_event(info):

	e = info.event.__dict__

	m = e["moments"]
	if not m:
		print("No moments logged in event.")
		return np.empty([1, 1]), pd.DataFrame()

	game_clock_m = []
	shot_clock_m = []
	ball_x_m = []
	ball_y_m = []
	player_team_m = []
	player_id_m = []
	player_x_m = []
	player_y_m = []

	for i in m:
		i_dict = i.__dict__

		game_clock_m.append(i_dict["game_clock"])
		shot_clock_m.append(i_dict["shot_clock"])
		ball_x_m.append(i_dict["ball"].__dict__["x"])
		ball_y_m.append(i_dict["ball"].__dict__["y"])

		player_team_m_i = []
		player_id_m_i = []
		player_x_m_i = []
		player_y_m_i = []
		for j in i_dict["players"]:
			j_dict = j.__dict__

			player_team_m_i.append(j_dict["team"].__dict__["id"])
			player_id_m_i.append(j_dict["id"])
			player_x_m_i.append(j_dict["x"])
			player_y_m_i.append(j_dict["y"])

		player_team_m.append(player_team_m_i)
		player_id_m.append(player_id_m_i)
		player_x_m.append(player_x_m_i)
		player_y_m.append(player_y_m_i)
	
	for i in range(0, len(player_team_m)):
		if len(player_team_m[i]) != 10:
			# player_team_m[i].append(0)
			player_team_m[i].extend([0]*(10-len(player_team_m[i])))
			player_id_m[i].extend([0]*(10-len(player_id_m[i])))
			player_x_m[i].extend([0]*(10-len(player_x_m[i])))
			player_y_m[i].extend([0]*(10-len(player_y_m[i])))

	player_team_arr = np.array(player_team_m)
	player_id_arr = np.array(player_id_m)
	player_x_arr = np.array(player_x_m)
	player_y_arr = np.array(player_y_m)

	event_arr_ball = np.column_stack((game_clock_m, shot_clock_m, ball_x_m,
		ball_y_m))
	event_arr = np.concatenate((event_arr_ball, player_team_arr,
		player_id_arr, player_x_arr, player_y_arr),
		axis = 1
		)

	cols = ["game_clock", "shot_clock", "ball_x", "ball_y", "player1_team",
		"player2_team", "player3_team", "player4_team", "player5_team", 
		"player6_team", "player7_team", "player8_team", "player9_team", 
		"player10_team", "player1_id", "player2_id", "player3_id",
		"player4_id", "player5_id", "player6_id", "player7_id",
		"player8_id", "player9_id", "player10_id", "player1_x",
		"player2_x", "player3_x", "player4_x", "player5_x", 
		"player6_x", "player7_x", "player8_x", "player9_x", 
		"player10_x", "player1_y", "player2_y", "player3_y",
		"player4_y", "player5_y", "player6_y", "player7_y",
		"player8_y", "player9_y", "player10_y"]

	event_df = pd.DataFrame(event_arr, columns = cols)

	return event_arr, event_df

game_json = "./data/games/0021500502"
# a, b = unnest_event(g)
# print(a)
# print(b)
events_list = []
event_n = 453 + 1
for i in range(0, event_n):

	print(i)
	g_i = Game(path_to_json = game_json + ".json", event_index = i)
	g_i.read_json()
	arr_i, df_i = unnest_event(g_i)
	# events_list.append(df_i)
	events_list.append(arr_i)

with open("events_arr_0021500502" + ".pkl", "wb") as f:
	pickle.dump(events_list, f)

