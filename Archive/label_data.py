import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("./data/events/events_dict_0021500013.pkl", "rb") as f:
	events_dict = pickle.load(f)

def draw_court(axis):
    import matplotlib.image as mpimg
    # https://github.com/gmf05/nba/blob/master/image/nba_court_T.png
    img = mpimg.imread("./nba_court_T.png") 
    plt.imshow(img, extent = axis, zorder = 0)

def plot_ball_event(events, event_id):

	fig = plt.figure(figsize = (15, 7.5))
	ax = plt.gca()
	draw_court([0, 100, 0, 50])
	plt.scatter(
		x = events[event_id][2]["ball_x"],
		y = events[event_id][2]["ball_y"],
		c = events[event_id][2].index
		)
	for i, txt in enumerate(events[event_id][2]["game_clock"]):
		ax.annotate(txt, (events[event_id][2]["ball_x"][i], events[event_id][2]["ball_y"][i]),
			fontsize = 6)


	plt.title(events[event_id][1])
	plt.show()

eid = "5"
# plot_ball_event(events = events_dict, event_id = eid)
scmin = 21
scmax = 24
subset = events_dict[eid][2][
	(events_dict[eid][2]["shot_clock"] > scmin)
	& (events_dict[eid][2]["shot_clock"] < scmax)
	& (events_dict[eid][2]["ball_z"] > 9)]
print(subset["ball_x"].iloc[0])
fig = plt.figure(figsize = (15, 7.5))
ax = plt.gca()
draw_court([0, 100, 0, 50])
plt.scatter(
	x = subset["ball_x"],
	y = subset["ball_y"],
	c = subset.index
	)

for i, txt in enumerate(subset["game_clock"]):
	
	ax.annotate(txt, (subset["ball_x"][i], subset["ball_y"][i]),
		fontsize = 6)

plt.show()
x
# test = events_dict["14"][2]["game_clock"].tolist()
# print(events_dict["14"][2].loc[550:690, ].tail(50))
# plt.plot(test)
# plt.show()

# print(events_dict["176"][2])
# x

ball_speed = []
for i in events_dict:
	events_dict[i][2] = events_dict[i][2].assign(
		ball_distance = lambda x:((x["ball_x"] - x["ball_x"].shift(1)) ** 2 + (x["ball_y"] - x["ball_y"].shift(1)) ** 2 + (x["ball_z"] - x["ball_z"].shift(1)) ** 2) ** (1 / 2),
		ball_xy_distance = lambda x: ((x["ball_x"] - x["ball_x"].shift(1)) ** 2 + (x["ball_y"] - x["ball_y"].shift(1)) ** 2) ** (1 / 2),
		delta_time = lambda x: abs(x["game_clock"].shift(1) - x["game_clock"]),
		ball_speed = lambda x: x["ball_distance"] / x["delta_time"],
		ball_xy_speed = lambda x: x["ball_xy_distance"] / x["delta_time"]
		)

	# ball_speed.append(events_dict[i][2]["ball_speed"].tolist())
	events_dict[i][2] = events_dict[i][2][(events_dict[i][2]["ball_speed"] <= 50)]
	# print(events_dict[i][2])
	# print(min(events_dict[i][2]["delta_time"]))

# ball_speed = [i for event in ball_speed for i in event if i <= 50]
# print(ball_speed)
# x

ball_z = [events_dict[i][2]["ball_z"].tolist() for i in events_dict]
ball_z = list(np.concatenate(ball_z))

ball_speed = [events_dict[i][2]["ball_speed"].tolist() for i in events_dict]
ball_speed = list(np.concatenate(ball_speed))

plt.scatter(
	x = ball_z,
	y = ball_speed
	)
plt.show()

import seaborn as sns
sns.set_style('whitegrid')
sns.kdeplot(np.array(ball_z))
plt.show()
x

def draw_court(axis):
    import matplotlib.image as mpimg
    # https://github.com/gmf05/nba/blob/master/image/nba_court_T.png
    img = mpimg.imread("./nba_court_T.png") 
    plt.imshow(img, extent = axis, zorder = 0)

def plot_ball_event(events, event_id):

	fig = plt.figure(figsize = (15, 7.5))
	ax = plt.gca()
	draw_court([0, 100, 0, 50])
	plt.scatter(
		x = events[event_id][2]["ball_x"],
		y = events[event_id][2]["ball_y"],
		c = events[event_id][2].index
		)
	for i, txt in enumerate(events[event_id][2]["game_clock"]):
		ax.annotate(txt, (events[event_id][2]["ball_x"][i], events[event_id][2]["ball_y"][i]),
			fontsize = 6)


	plt.title(events[event_id][1])
	plt.show()

# print(events_dict.keys())
# events = events_dict
# event_id = "17"

# plot_ball_event(
# 	events = events,
# 	event_id = event_id
# 	)

# events_dict[event_id][2]["fga_label"] = 0
# start_time = 694.34
# end_time = 694.27
# def labeller(s, start, end):
#     if (s <= start) & (s >= end):
#         return 1
#     else:
#         return 0
# events_dict[event_id][2]["fga_label"]=events_dict[event_id][2]["game_clock"].apply(
# 	lambda x: labeller(x, start = start_time, end = end_time))
# print(events_dict[event_id][2])