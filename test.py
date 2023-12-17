import time
print(time.localtime())
s_time = time.time()
print(s_time)

print("start at %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# print("start at %s" % time.strftime("%Y-%m-%d %H:%M:%S", s_time))
print(time.localtime(s_time))
time.sleep(62)
n=time.time()-s_time
print("end at %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
