import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# this is good. go forward with this. possibly add variable, does it hit rim and/or backboard!!
# hitting backboard may affect fga_theta
with open("./data/events/events_dict_0021500013.pkl", "rb") as f:
	events_dict = pickle.load(f)

import random
random.seed(55)

basket1 = np.array([4.75, 25])
basket2 = np.array([94 - 4.75, 25])

def calculate_angle(p1, p2, p3):

	# p2 is basket
	optimal = p1 - p2
	fga = p1 - p3
	# fga = p3 - p2

	cosine_angle = np.dot(optimal, fga) / (np.linalg.norm(optimal) * np.linalg.norm(fga))
	angle_rad = np.arccos(cosine_angle)

	return np.degrees(angle_rad)

def apply_anglefunc_df(df):

	theta_deg = []

	basket1 = np.array([4.75, 25])
	basket2 = np.array([94 - 4.75, 25])

	for i in range(1, df.shape[0]):
		if df.iloc[i, 3] < 47:
			point2 = basket1
		else:
			point2 = basket2
		theta_deg_i = calculate_angle(
			p1 = np.array([df.iloc[(i - 1), 3], df.iloc[(i - 1), 4]]),
			p2 = point2,
			p3 = np.array([df.iloc[i, 3], df.iloc[i, 4]])
			)
		theta_deg.append(theta_deg_i)

	theta_deg.insert(0, 90)
	# tracking_fga_b1["shot_theta_rad"] = theta_rad
	df["fga_theta"] = theta_deg

	df["fga_theta_diff"] = abs(df["fga_theta"] - df["fga_theta"].shift(1))

	return df

for i in events_dict:

	events_dict[i][2] = apply_anglefunc_df(df = events_dict[i][2])

	np.seterr(divide='ignore', invalid='ignore')
	events_dict[i][2] = events_dict[i][2].assign(
		ball_distance = lambda x:((x["ball_x"] - x["ball_x"].shift(1)) ** 2 + (x["ball_y"] - x["ball_y"].shift(1)) ** 2 + (x["ball_z"] - x["ball_z"].shift(1)) ** 2) ** (1 / 2),
		ball_xy_distance = lambda x: ((x["ball_x"] - x["ball_x"].shift(1)) ** 2 + (x["ball_y"] - x["ball_y"].shift(1)) ** 2) ** (1 / 2),
		# abs(time) because some time stamps are repeated and/or not in order if clock was adjusted.
		delta_time = lambda x: abs(x["game_clock"].shift(1) - x["game_clock"]),
		ball_speed = lambda x: x["ball_distance"] / x["delta_time"],
		ball_xy_speed = lambda x: x["ball_xy_distance"] / x["delta_time"],
		ball_z_speed = lambda x: abs(x["ball_z"] - x["ball_z"].shift(1)) / x["delta_time"],
		event_id = i
		)

events_df = pd.concat([events_dict[i][2] for i in events_dict])

# events_df["ball_speed"].replace([np.inf], np.nan, inplace=True) 
# events_df.dropna(subset = ["ball_speed"], inplace=True) 
# plt.boxplot(events_df["ball_speed"])
# # plt.hist(events_df["ball_speed"], bins = 30, log = True)
# # plt.xlabel("Ball Speed [ft/s]")
# plt.ylabel("Ball Speed [ft/s]")
# plt.savefig("./plots/ballspeed_box.png")
# plt.show()

events_df = events_df[(events_df["ball_speed"]) <= 50]

# plt.boxplot(events_df["ball_speed"])
# plt.hist(events_df["ball_speed"])
# plt.xlabel("Ball Speed [ft/s]")
# plt.ylabel("Count")
# plt.savefig("./plots/ballspeed_hist1.png")
# plt.show()

events_df.dropna(subset = ["fga_theta"], inplace = True)
events_df.drop_duplicates(
	subset = ["quarter", "game_clock"],
	keep = "last",
	inplace = True
	)


# i = 1650
# i = 2125
# i = [2080, 2090]
# def draw_court(axis):
#     import matplotlib.image as mpimg
#     # https://github.com/gmf05/nba/blob/master/image/nba_court_T.png
#     img = mpimg.imread("./nba_court_T.png") 
#     plt.imshow(img, extent = axis, zorder = 0)

# events_df["game_clock_min"] = events_df["game_clock"].astype(int) % 3600 // 60
# events_df["game_clock_sec"] = events_df["game_clock"].astype(int) % 60
# timestring = "{:01d}:{:02d}"
# events_df["game_clock_timestring"] = events_df.apply(
# 	lambda r: timestring.format(r['game_clock_min'], r['game_clock_sec']),
# 	axis = 1)
# fig = plt.figure(figsize = (15, 7.5))
# ax = plt.gca()
# draw_court([0, 94, 50, 0])
# plt.axis("off")
# plt.scatter(
# 	x = events_df.iloc[i, 3],
# 	y = events_df.iloc[i, 4],
# 	c = "orangered",
# 	s = 100
# 	)
# point1 = [1, 2]
# point2 = [3, 4]
# r'$\lambda$'
# x_values = events_df.iloc[i, 3].tolist()
# y_values = events_df.iloc[i, 4].tolist()
# x_values.insert(0, 4.75)
# y_values.insert(0, 25)
# plt.plot(
# 	x_values,
# 	y_values,
# 	markersize = 10,
# 	marker = ".",
# 	color = "orangered",
# 	linestyle = "solid"
# 	)
# plt.text(25, 33, r'$\theta$')
# plt.show()
# plt.scatter(
# 	x = events_df.iloc[i, 7],
# 	y = events_df.iloc[i, 8],
# 	c = "#007AC1",
# 	s = 150
# 	)
# plt.scatter(
# 	x = events_df.iloc[i, 10],
# 	y = events_df.iloc[i, 11],
# 	c = "#007AC1",
# 	s = 150
# 	)
# plt.scatter(
# 	x = events_df.iloc[i, 13],
# 	y = events_df.iloc[i, 14],
# 	c = "#007AC1",
# 	s = 150
# 	)
# plt.scatter(
# 	x = events_df.iloc[i, 16],
# 	y = events_df.iloc[i, 17],
# 	c = "#007AC1",
# 	s = 150
# 	)
# plt.scatter(
# 	x = events_df.iloc[i, 19],
# 	y = events_df.iloc[i, 20],
# 	c = "#007AC1",
# 	s = 150
# 	)
# plt.scatter(
# 	x = events_df.iloc[i, 22],
# 	y = events_df.iloc[i, 23],
# 	c = "#000000",
# 	s = 150
# 	)
# plt.scatter(
# 	x = events_df.iloc[i, 25],
# 	y = events_df.iloc[i, 26],
# 	c = "#000000",
# 	s = 150
# 	)
# plt.scatter(
# 	x = events_df.iloc[i, 28],
# 	y = events_df.iloc[i, 29],
# 	c = "#000000",
# 	s = 150
# 	)
# plt.scatter(
# 	x = events_df.iloc[i, 31],
# 	y = events_df.iloc[i, 32],
# 	c = "#000000",
# 	s = 150
# 	)
# plt.scatter(
# 	x = events_df.iloc[i, 34],
# 	y = events_df.iloc[i, 35],
# 	c = "#000000",
# 	s = 150
# 	)
# plt.title("Q" + str(int(events_df.iloc[i, 0])) + " " + str(events_df.iloc[i, 47]))
# plt.show()

# test = events_df[["quarter", "game_clock", "ball_x", "ball_y",
# 	"ball_z", "ball_distance", "delta_time", "ball_speed"]]
# print(test.iloc[25100:25150, :])
# # print(test.tail(50))
# y

# clock_repeats = events_df.groupby(["quarter", "game_clock", "shot_clock"]).size().reset_index(name = "n")
# clock_repeats = clock_repeats[clock_repeats["n"] > 1]
# print(clock_repeats)

# def check_clock_repeats(df_repeats, df_check):

# 	n_x = []
# 	n_y = []
# 	n_z = []
# 	for i in range(0, len(df_repeats)):
# 		repeat_i = df_repeats.iloc[i, :]

# 		check_i = df_check[
# 			(df_check["quarter"] == repeat_i["quarter"])
# 			& (df_check["game_clock"] == repeat_i["game_clock"])
# 			& (df_check["shot_clock"] == repeat_i["shot_clock"])
# 			]

# 		if (len(check_i["ball_x"].unique())) | (len(check_i["ball_y"].unique())) | (len(check_i["ball_z"].unique())) != 1:
# 			print(check_i)

# 		n_x.append(len(check_i["ball_x"].unique()))
# 		n_y.append(len(check_i["ball_y"].unique()))
# 		n_z.append(len(check_i["ball_z"].unique()))

# 	return max(n_x), max(n_y), max(n_z)

# x, y, z = check_clock_repeats(
# 	df_repeats = clock_repeats,
# 	df_check = events_df
# 	)
# print(x, y, z)


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

events_m = events_df[["fga_theta", "ball_speed", "ball_z"]].to_numpy()

scaler = StandardScaler()
events_m = scaler.fit_transform(events_m)
# pca = PCA(n_components = 2)
# events_m_pc = pca.fit_transform(events_m)
# print(events_m_pc)
# print(pca.explained_variance_ratio_)
# print(sum(pca.explained_variance_ratio_))
# plt.scatter(
# 	x = events_m_pc[:, 0],
# 	y = events_m_pc[:, 1]
# 	)
# plt.show()

# sse = []
# for k in range(1, 15):
# 	m = KMeans(
# 		init = "random",
# 		n_clusters = k,
# 		n_init = 10,
# 		max_iter = 300,
# 		random_state = 55
# 		)
# 	m.fit(events_m)
# 	sse.append(m.inertia_)

# plt.plot(range(1, 15), sse)
# plt.xticks(range(1, 15))
# plt.xlabel("Clusters")
# plt.ylabel("SSE")
# plt.savefig("./plots/kmeans_n.png")
# plt.show()
# # from kneed import KneeLocator
# # kl = KneeLocator(range(1, 15), sse, curve="convex", direction="decreasing")
# # print(kl.elbow)

m = KMeans(
	init = "random",
	n_clusters = 4,
	n_init = 10,
	max_iter = 300,
	random_state = 55
	)

m.fit(events_m)

clusters = m.labels_.reshape(-1, 1)
events_m = np.concatenate((clusters, events_m), axis = 1)

events_df["cluster"] = events_m[:, 0]

def draw_court(axis):
    import matplotlib.image as mpimg
    # https://github.com/gmf05/nba/blob/master/image/nba_court_T.png
    img = mpimg.imread("./nba_court_T.png") 
    plt.imshow(img, extent = axis, zorder = 0)

def plot_ball_event(df):
	# from matplotlib.colors import ListedColormap
	# colors = ListedColormap(['red', 'blue', 'purple', "green"])

	fig = plt.figure(figsize = (15, 7.5))
	ax = plt.gca()
	# draw_court([0, 100, 0, 50])
	draw_court([0, 94, 50, 0])
	# xx = plt.scatter(
	# 	x = df["ball_x"],
	# 	y = df["ball_y"],
	# 	c = df["cluster"],
	# 	cmap = colors
	# 	)
	plt.scatter(
		x = df["ball_x"],
		y = df["ball_y"],
		c = df["cluster"]
		)
	# plt.axis([0, 100, 50, 0])
	# plt.legend(*xx.legend_elements())

	# plt.savefig("")

	# plt.show()




for i in events_dict:
	plot_ball_event(df = events_df[events_df["event_id"] == i])
	plt.savefig("./event_plots/" + "event_" + i + ".png")
	plt.close()


# from itertools import groupby
# from collections import Counter
# cluster_list = events_df["cluster"].tolist()
# streaks = [i for i, group in groupby(cluster_list) if len(list(group)) >= 15]
# numberOfStreaks = len(streaks)
# faceStreaks = dict(Counter(streaks))
# print(faceStreaks)

shot_c = 1
streak_min = 15

streak_idx = []
for i in range(streak_min, (events_df.shape[0] + 1)):
	events_sub = events_df.iloc[(i-streak_min):i, (events_df.shape[1] - 1)].tolist()
	if sum(1 for j in events_sub if j == shot_c) == streak_min:
		streak_idx.append(list(range(i-streak_min, i)))

streak_idx = [i for j in streak_idx for i in j]
streak_idx = sorted(set(streak_idx), key = streak_idx.index)
fga_cluster = events_df.iloc[streak_idx, :].copy()

fga_id = [0]
fga_id_i = 0
for i in range(1, len(streak_idx)):
	if streak_idx[i] == streak_idx[i-1] + 1:
		fga_id.append(fga_id_i)
	else:
		fga_id_i += 1
		fga_id.append(fga_id_i)

fga_cluster["fga_id"] = fga_id

for i in range(0, 4):
	df_i = events_df[events_df["cluster"] == i]
	plot_ball_event(df = df_i)
	plt.axis("off")
	plt.title("Cluster " + str(i))
	plt.savefig("./cluster_plots/cluster_" + str(i) + ".png")
	plt.close()

# plt.scatter(
# 	x = events_df[events_df["cluster"] == 0]["ball_x"],
# 	y = events_df[events_df["cluster"] == 0]["ball_y"]
# 	)
# plt.savefig("./cluster_plots/cluster_0.png")
# plt.show()
# plt.scatter(
# 	x = events_df[events_df["cluster"] == 1]["ball_x"],
# 	y = events_df[events_df["cluster"] == 1]["ball_y"]
# 	)
# plt.savefig("./cluster_plots/cluster_1.png")
# plt.show()
# plt.scatter(
# 	x = events_df[events_df["cluster"] == 2]["ball_x"],
# 	y = events_df[events_df["cluster"] == 2]["ball_y"]
# 	)
# plt.savefig("./cluster_plots/cluster_2.png")
# plt.show()
# plt.scatter(
# 	x = events_df[events_df["cluster"] == 3]["ball_x"],
# 	y = events_df[events_df["cluster"] == 3]["ball_y"]
# 	)
# plt.savefig("./cluster_plots/cluster_3.png")
# plt.show()

# plt.scatter(
# 	x = fga_cluster["ball_x"],
# 	y = fga_cluster["ball_y"]
# 	)
# plt.show()

for i in set(fga_id):
	plot_ball_event(df = fga_cluster[fga_cluster["fga_id"] == i])
	plt.savefig("./fga_plots/" + "fga_" + str(i) + ".png")
	plt.close()


# fga_dict = {i: None for i in set(fga_cluster["fga_id"])}
# print(fga_dict)

fga_stamps = fga_cluster.groupby(
	"fga_id").agg(
	{"game_clock": ["max", "min"]}
	)
print(fga_stamps)

z0 = events_df[events_df["cluster"] == shot_c]["ball_speed"]
z1 = events_df[events_df["cluster"] != shot_c]["ball_speed"]
z = [z0, z1]
plt.boxplot(z)
plt.show()