import numpy as np
import pickle
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.animation as animation

# https://danvatterott.com/blog/2016/06/16/creating-videos-of-nba-action-with-sportsvu-data/
def draw_court(axis):
    import matplotlib.image as mpimg
    # https://github.com/gmf05/nba/blob/master/image/nba_court_T.png
    img = mpimg.imread("./nba_court_T.png") 
    plt.imshow(img, extent = axis, zorder = 0)

with open("./data/events/events_dict_0021500013.pkl", "rb") as f:
	events_dict = pickle.load(f)

def plot_ball_event(events, event_id):

	fig = plt.figure(figsize = (15, 7.5))
	ax = plt.gca()
	draw_court([0, 100, 0, 50])
	plt.scatter(
		x = events[event_id][2]["ball_x"],
		y = events[event_id][2]["ball_y"],
		c = events[event_id][2]["ball_z"]
		# c = events[event_id][2].index
		)
	for i, txt in enumerate(events[event_id][2]["game_clock"]):
		ax.annotate(txt, (events[event_id][2]["ball_x"][i], events[event_id][2]["ball_y"][i]),
			fontsize = 6)


	plt.title(events[event_id][1])
	plt.show()

events = events_dict
event_id = "14"

plot_ball_event(
	events = events,
	event_id = event_id
	)

# x
fig = plt.figure(figsize=(15,7.5)) #create figure object
ax = plt.gca() #create axis object
draw_court([0,94,0,50])


ax.set_xlim(-5, 100)
ax.set_ylim(-5, 50)
hoop1_scat = ax.scatter(0, 0, color = "orange", marker = "o", s = 1)
hoop2_scat = ax.scatter(0, 0, color = "orange", marker = "o", s = 10)
hoop1_x = [4.75]*events_dict["562"][2].shape[0]
hoop1_y = [25]*events_dict["562"][2].shape[0]
hoop2_x = [94 - 4.75]*events_dict["562"][2].shape[0]
hoop2_y = [25]*events_dict["562"][2].shape[0]
ball_scat = ax.scatter(0, 0, color = "black")
ball_x = events[event_id][2]["ball_x"]
ball_y = events[event_id][2]["ball_y"]
p1_scat = ax.scatter(0, 0, color = "blue")
p1_x = events[event_id][2]["home1_x"]
p1_y = events[event_id][2]["home1_y"]
p2_scat = ax.scatter(0, 0, color = "blue")
p2_x = events[event_id][2]["home2_x"]
p2_y = events[event_id][2]["home2_y"]
p3_scat = ax.scatter(0, 0, color = "blue")
p3_x = events[event_id][2]["home3_x"]
p3_y = events[event_id][2]["home3_y"]
p4_scat = ax.scatter(0, 0, color = "blue")
p4_x = events[event_id][2]["home4_x"]
p4_y = events[event_id][2]["home4_y"]
p5_scat = ax.scatter(0, 0, color = "blue")
p5_x = events[event_id][2]["home5_x"]
p5_y = events[event_id][2]["home5_y"]
p6_scat = ax.scatter(0, 0, color = "green")
p6_x = events[event_id][2]["away1_x"]
p6_y = events[event_id][2]["away1_y"]
p7_scat = ax.scatter(0, 0, color = "green")
p7_x = events[event_id][2]["away2_x"]
p7_y = events[event_id][2]["away2_y"]
p8_scat = ax.scatter(0, 0, color = "green")
p8_x = events[event_id][2]["away3_x"]
p8_y = events[event_id][2]["away3_y"]
p9_scat = ax.scatter(0, 0, color = "green")
p9_x = events[event_id][2]["away4_x"]
p9_y = events[event_id][2]["away4_y"]
p10_scat = ax.scatter(0, 0, color = "green")
p10_x = events[event_id][2]["away5_x"]
p10_y = events[event_id][2]["away5_y"]


def animate(i):
	# hoop1_scat.set_offsets((hoop1_x[i], hoop1_y[i]))
	# hoop2_scat.set_offsets((hoop2_x[i], hoop2_y[i]))
	ball_scat.set_offsets((ball_x[i], ball_y[i]))
	p1_scat.set_offsets((p1_x[i], p1_y[i]))
	p2_scat.set_offsets((p2_x[i], p2_y[i]))
	p3_scat.set_offsets((p3_x[i], p3_y[i]))
	p4_scat.set_offsets((p4_x[i], p4_y[i]))
	p5_scat.set_offsets((p5_x[i], p5_y[i]))
	p6_scat.set_offsets((p6_x[i], p6_y[i]))
	p7_scat.set_offsets((p7_x[i], p7_y[i]))
	p8_scat.set_offsets((p8_x[i], p8_y[i]))
	p9_scat.set_offsets((p9_x[i], p9_y[i]))
	p10_scat.set_offsets((p10_x[i], p10_y[i]))
	return ball_scat, p1_scat, p2_scat, p3_scat, p4_scat, p5_scat, p6_scat, p7_scat, p8_scat, p9_scat, p10_scat,

ani = animation.FuncAnimation(fig, animate, repeat=False,
                                frames=len(events[event_id][2]) - 1, interval=50)
plt.show()
# # # To save the animation using Pillow as a gif
# # # writer = animation.PillowWriter(fps=15,
# # #                                 metadata=dict(artist='Me'),
# # #                                 bitrate=1800)
# # # ani.save('scatter.gif', writer=writer)

# # plt.show()

# # # fig,ax = plt.subplots()

# # # def animate(i):
# # #     ax.clear()
# # #     ax.set_xlim(-5, 100)
# # #     ax.set_ylim(-5, 50)

# # #     # hoop, = ax.plot(
# # #     # 	4.75,
# # #     # 	25,
# # #     # 	color = "red")
# # #     ball, = ax.plot(
# # #     	events_dict["5"]["ball_x"][i],
# # #     	events_dict["5"]["ball_y"][i],
# # #     	marker = '.',
# # #     	color = 'orange'
# # #     	)
# # #     p1, = ax.plot(
# # #     	events_dict["5"]["home1_x"][i],
# # #     	events_dict["5"]["home1_y"][i],
# # #     	marker = '.',
# # #     	color = 'blue'
# # #     	)

# # #     return ball, p1
        
# # # ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=False, frames=100)  
# # # ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=False)  
# # # ani.save("TLI.gif", dpi=300, writer=PillowWriter(fps=25))