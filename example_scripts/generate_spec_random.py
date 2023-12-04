import random
import json

spec = { "max_x": 12, "max_y": 12, "rewards": [] }
for x in range(0, spec["max_x"]):
    for y in range(0, spec["max_y"]):
        r = random.uniform(0.0,1)
        spec["rewards"].append( [[x, y], r] )

for i in range(0, int(spec["max_x"]*spec["max_y"]*0.1) ):
    x = random.randint(0, spec["max_x"]-1)
    y = random.randint(0, spec["max_y"]-1) 
    r = random.uniform(10,20)

    spec["rewards"].append( [[x, y], r] )

print(json.dumps(spec))
