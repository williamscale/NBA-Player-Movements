import json
# import numpy as np
import pandas as pd
import pickle
from nba_api.stats.endpoints import playbyplay as pbp

# https://danvatterott.com/blog/2016/06/16/creating-videos-of-nba-action-with-sportsvu-data/

game_pbp_ep = pbp.PlayByPlay(game_id = "0021500013")
game_pbp_df = game_pbp_ep.get_data_frames()[0]

make_events = game_pbp_df[game_pbp_df["EVENTMSGTYPE"] == 1][["EVENTNUM", "HOMEDESCRIPTION", "NEUTRALDESCRIPTION", "VISITORDESCRIPTION"]]
miss_events = game_pbp_df[game_pbp_df["EVENTMSGTYPE"] == 2][["EVENTNUM", "HOMEDESCRIPTION", "NEUTRALDESCRIPTION", "VISITORDESCRIPTION"]]
fga_dict = {str(i): [1] for i in make_events["EVENTNUM"]}
miss_dict = {str(i): [0] for i in miss_events["EVENTNUM"]}
fga_dict.update(miss_dict)

make_events_num = make_events["EVENTNUM"].tolist()
miss_events_num = miss_events["EVENTNUM"].tolist()
make_desc = list(make_events[["HOMEDESCRIPTION", "NEUTRALDESCRIPTION", "VISITORDESCRIPTION"]].values.tolist())
miss_desc = list(miss_events[["HOMEDESCRIPTION", "NEUTRALDESCRIPTION", "VISITORDESCRIPTION"]].values.tolist())

desc_dict = {str(make_events_num[i]): make_desc[i] for i in range(len(make_events_num))}
miss_desc_dict = {str(miss_events_num[i]): miss_desc[i] for i in range(len(miss_events_num))}
desc_dict.update(miss_desc_dict)


json_data = open("./data/games_json/0021500013.json")
tracking = json.load(json_data)["events"]

for i in tracking:
	if i["eventId"] in fga_dict:
		fga_dict[i["eventId"]].append(desc_dict[i["eventId"]])
		fga_dict[i["eventId"]].append(i["moments"])

event_dict = {}
for i in fga_dict:
	quarter = []
	game_clock = []
	shot_clock = []
	ball_x = []
	ball_y = []
	ball_z = []
	home1_id = []
	home1_x = []
	home1_y = []
	home2_id = []
	home2_x = []
	home2_y = []
	home3_id = []
	home3_x = []
	home3_y = []
	home4_id = []
	home4_x = []
	home4_y = []
	home5_id = []
	home5_x = []
	home5_y = []
	away1_id = []
	away1_x = []
	away1_y = []
	away2_id = []
	away2_x = []
	away2_y = []
	away3_id = []
	away3_x = []
	away3_y = []
	away4_id = []
	away4_x = []
	away4_y = []
	away5_id = []
	away5_x = []
	away5_y = []
	for j in fga_dict[i][2]:
		if len(j[5]) == 11:
			quarter.append(j[0])
			game_clock.append(j[2])
			shot_clock.append(j[3])
			ball_x.append(j[5][0][2])
			ball_y.append(j[5][0][3])
			ball_z.append(j[5][0][4])
			home1_id.append(j[5][1][1])
			home1_x.append(j[5][1][2])
			home1_y.append(j[5][1][3])
			home2_id.append(j[5][2][1])
			home2_x.append(j[5][2][2])
			home2_y.append(j[5][2][3])
			home3_id.append(j[5][3][1])
			home3_x.append(j[5][3][2])
			home3_y.append(j[5][3][3])
			home4_id.append(j[5][4][1])
			home4_x.append(j[5][4][2])
			home4_y.append(j[5][4][3])
			home5_id.append(j[5][5][1])
			home5_x.append(j[5][5][2])
			home5_y.append(j[5][5][3])
			away1_id.append(j[5][6][1])
			away1_x.append(j[5][6][2])
			away1_y.append(j[5][6][3])
			away2_id.append(j[5][7][1])
			away2_x.append(j[5][7][2])
			away2_y.append(j[5][7][3])
			away3_id.append(j[5][8][1])
			away3_x.append(j[5][8][2])
			away3_y.append(j[5][8][3])
			away4_id.append(j[5][9][1])
			away4_x.append(j[5][9][2])
			away4_y.append(j[5][9][3])
			away5_id.append(j[5][10][1])
			away5_x.append(j[5][10][2])
			away5_y.append(j[5][10][3])

	event_dict[i] = [
		fga_dict[i][0],
		fga_dict[i][1],
		pd.DataFrame({
			"quarter": quarter,
			"game_clock": game_clock,
			"shot_clock": shot_clock,
			"ball_x": ball_x,
			"ball_y": ball_y,
			"ball_z": ball_z,
			"home1_id": home1_id,
			"home1_x": home1_x,
			"home1_y": home1_y,
			"home2_id": home2_id,
			"home2_x": home2_x,
			"home2_y": home2_y,
			"home3_id": home3_id,
			"home3_x": home3_x,
			"home3_y": home3_y,
			"home4_id": home4_id,
			"home4_x": home4_x,
			"home4_y": home4_y,
			"home5_id": home5_id,
			"home5_x": home5_x,
			"home5_y": home5_y,
			"away1_id": away1_id,
			"away1_x": away1_x,
			"away1_y": away1_y,
			"away2_id": away2_id,
			"away2_x": away2_x,
			"away2_y": away2_y,
			"away3_id": away3_id,
			"away3_x": away3_x,
			"away3_y": away3_y,
			"away4_id": away4_id,
			"away4_x": away4_x,
			"away4_y": away4_y,
			"away5_id": away5_id,
			"away5_x": away5_x,
			"away5_y": away5_y
			})
		]


with open("./data/events/events_dict_0021500013" + ".pkl", "wb") as f:
	pickle.dump(event_dict, f)
