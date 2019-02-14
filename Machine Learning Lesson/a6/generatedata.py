import random

for z in range(100):
  x = random.uniform(0.1, 0.5)
  x = round(x, 2)
  y = random.uniform(0.1, 0.5)
  y = round(y, 2)
  print `x` + ' ' +  `y` + ' ' + `round(x + y, 2)`
