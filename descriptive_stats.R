library(plyr)
library(dplyr)
v = read.csv("vehicles.csv")
m = read.csv("maintenance.csv")
# compute annual total maintenance cost
maint_cost = ddply(m, .(Year.WO.Completed), summarise, total_cost = sum(Actual.Labor.Cost + Commercial.Cost + Part.Cost))
colMeans(filter(maint_cost, Year.WO.Completed >= 2010, Year.WO.Completed < 2017))
# compute annual vehicle purchase costs
vehicle_cost = ddply(v, .(Year), summarise, total_cost = sum(Purchase.Cost))
colMeans(filter(vehicle_cost, Year >= 2010, Year < 2017))
# number of active vehicles
table(v$Status.Code)