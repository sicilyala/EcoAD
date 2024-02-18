import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as scio  

from common.arguments import get_args

args = get_args()
log_dir = "../EcoHighway_DRL/" + args.dir_name + "/"    
drl_model = args.drl_model.lower()
model_dir = log_dir + drl_model + "-model-%s" % args.model_time
data_dir = model_dir + "-data"
print("-----process %s" % data_dir)
 
x_step = range(args.replay_steps)
action_0 = []
action_1 = []
action_2 = []
spd = []
position_x = []
position_y = []
crash = []
lane = []

for i in range(args.replay_steps):
    datai = scio.loadmat(data_dir+"/step%d"%i)
    
    actions = datai["action"][0]
    action_0.append(actions[0])
    action_1.append(actions[1])
    action_2.append(actions[2])    
    spd.append(datai["speed"][0][0])
    positions = datai["position"][0]
    position_x.append(positions[0])  
    position_y.append(positions[1])  
    crash.append(datai["crashed"][0][0])
    lane.append(datai["lane_index"][0][0])
    
plt.plot(x_step, lane)
plt.show()