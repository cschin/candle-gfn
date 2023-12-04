import random
import json

spec = { "max_x": 64, "max_y": 64, "rewards": [] }

for x in range(0, spec["max_x"]):
    for y in range(0, spec["max_y"]):

        xp = abs(x/(spec["max_x"]-1) - 0.5)
        xr1 = 1 if xp <= 0.5 and xp >= 0.25 else 0
        xr2 = 1 if xp <= 0.4 and xp >= 0.3 else 0
        yp = abs(y/(spec["max_y"]-1) - 0.5)
        yr1 = 1 if yp <= 0.5 and yp >= 0.25 else 0
        yr2 = 1 if yp <= 0.4 and yp >= 0.3 else 0
        r = 0.1 + 0.5 * xr1 * yr1 + 2 * xr2 * yr2
        spec["rewards"].append( [[x, y], r] )

print(json.dumps(spec))
