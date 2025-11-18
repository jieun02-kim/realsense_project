

def Artificial_Potention_Field(start_x,start_y,goal_x,goal_y,obs):
	x,y = start_x,start_y

	trace_x = []
	trace_y = []

	trace_x.append(x)
	trace_y.append(y)

	while(1):
		att_x, att_y = calc_attractive_force(x,y,goal_x,goal_y)
		rep_x, rep_y = calc_repulsive_force(x,y,obs)

		pot_x = att_x+rep_x
		pot_y = att_y+rep_y

		x = x + pot_x
		y = y + pot_y

		trace_x.append(x)
		trace_y.append(y)

		error = np.linalg.norm([goal_x-x,goal_y-y])

#		if error < 1:
#			plt.plot(obs[:,0],obs[:,1],'bo')
#			plt.plot(trace_x,trace_y,'go',goal_x,goal_y,'ro')
#			plt.show()
#			break



def calc_attractive_force(x, y, goal_x=1.0, goal_y=1.0, k_att=1.0, stop_dist=1.0):
    dx = goal_x - x
    dy = goal_y - y
    dist = np.hypot(dx, dy)

    if dist <= stop_dist:
        return 0.0, 0.0   # 1 m 이내면 끌어당기지 않음
    else:
        fx = k_att * dx
        fy = k_att * dy
        return fx, fy
        
# def calc_attractive_force(x,y,gx,gy):
# 	e_x, e_y = gx-x, gy-y
# 	distance = np.linalg.norm([e_x,e_y])

# 	att_x = Kp_att * e_x/distance
# 	att_y = Kp_att * e_y/distance

# 	return att_x, att_y



def calc_repulsive_force(x,y,obs):

	rep_x,rep_y = 0,0

	for obs_xy in np.ndindex(obs.shape[0]):

		obs_dis_x, obs_dis_y = obs[obs_xy][0]-x, obs[obs_xy][1]-y 
		obs_dis = np.linalg.norm([obs_dis_x,obs_dis_y]) 

		if obs_dis < obstacle_bound:
			rep_x = rep_x - Kp_rel * (1/obs_dis - 1/obstacle_bound)*(1/(obs_dis*obs_dis))*obs_dis_x/obs_dis
			rep_y = rep_y - Kp_rel * (1/obs_dis - 1/obstacle_bound)*(1/(obs_dis*obs_dis))*obs_dis_y/obs_dis
		else:
			rep_x = rep_x
			rep_y = rep_y

	return rep_x, rep_y
