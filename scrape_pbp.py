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

from nba_api.stats.endpoints import playbyplay as pbp

game_pbp_ep = pbp.PlayByPlay(game_id = "0021500502")

# game_pbp_dict = game_pbp_ep.get_dict()["resultSets"][0]
game_pbp_df = game_pbp_ep.get_data_frames()[0]

# print(game_pbp_df)
# print("MISS" in game_pbp_df.iloc[2, 9])
# print(game_pbp_df[game_pbp_df["VISITORDESCRIPTION"].str.contains("MISS", na = False)])

game_pbp_shots = game_pbp_df[
	(game_pbp_df["HOMEDESCRIPTION"].str.contains("MISS", na = False)) |
	(game_pbp_df["VISITORDESCRIPTION"].str.contains("MISS", na = False)) |
	(game_pbp_df["SCORE"].notnull())
	]

print(game_pbp_shots)
print(game_pbp_shots[game_pbp_shots["EVENTNUM"] == 101])
print(tracking_df[tracking_df["moment_or_event_idk"] == 101])
xyz
game_pbp_shots = game_pbp_shots[["PERIOD", "PCTIMESTRING", "HOMEDESCRIPTION", "VISITORDESCRIPTION", "SCORE"]]

# test = tracking_df[(tracking_df["quarter"] == 1) &
# 	(tracking_df["game_clock"] >= 420) & (tracking_df["game_clock"] <= 500)]

# print(test[["game_clock", "game_clock_timestring"]])

tracking_pbp = pd.merge(
	tracking_df,
	game_pbp_shots,
	how = "left",
	left_on = ["quarter", "game_clock_timestring"],
	right_on = ["PERIOD", "PCTIMESTRING"]
	)

tracking_pbp["HOMEDESCRIPTION"] = tracking_pbp["HOMEDESCRIPTION"].fillna(value = "No Description")
tracking_pbp["VISITORDESCRIPTION"] = tracking_pbp["VISITORDESCRIPTION"].fillna(value = "No Description")

tracking_pbp = tracking_pbp[~tracking_pbp["HOMEDESCRIPTION"].str.contains("Free Throw")]
tracking_pbp = tracking_pbp[~tracking_pbp["VISITORDESCRIPTION"].str.contains("Free Throw")]

tracking_pbp["fga_flag"] = np.where(
	(tracking_pbp["HOMEDESCRIPTION"] != "No Description")
	| (tracking_pbp["VISITORDESCRIPTION"] != "No Description"),
	1, 0
	)

tracking_pbp.drop_duplicates(
	subset = ["quarter", "game_clock"],
	keep = "first",
	inplace = True
	)

# tracking_shots = tracking_pbp[
# 	tracking_pbp["HOMEDESCRIPTION"].notnull() |
# 	tracking_pbp["HOMEDESCRIPTION"].notna() |
# 	tracking_pbp["VISITORDESCRIPTION"].notnull() |
# 	tracking_pbp["VISITORDESCRIPTION"].notna() |
# 	tracking_pbp["SCORE"].notnull() |
# 	tracking_pbp["SCORE"].notna()
# 	]

# with open("./data/tracking_pbp_0021500502" + ".pkl", "wb") as f:
# 	pickle.dump(tracking_pbp, f)

# import matplotlib.pyplot as plt
# import seaborn as sns
# # sns.kdeplot(x=tracking_shots.ball_x, y=tracking_shots.ball_y, cmap="Reds", fill=True)
# plt.scatter(x = tracking_shots["ball_x"], y = tracking_shots["ball_y"])
# plt.show()

