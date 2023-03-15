# airliquide-sustainability-challenge

## About the data :

# 1- Government data

File in shp format, necessary to keep the four other files with same name in different formats to correctly load it.

Some data are from 2018 because last updates from the French data website have corrupted the latest data file.


RRN : files on the road shape with coordinates

airesPL : location of heavytrucks hubs in France

Regions : contour of french regions in shp file

traffic : data on the traffic per road section including heavytrucks

cities : data on the name and region of each city of France


# 2- Other CSV files 

Complementary data about the corresponding region to each department & the location of gas station already existing.

The gas station file has coordinates not in the same format than the government data so we implement a translation algorithm.


DepReg : file associating regions to departements

stations : location of gas station in France


## Remark : 
There are no national roads in Corsica so we might not want to include this region in our analysis.



## Analysis steps :

# 1- Number of stations per region :

Compute the number of stations per region for 2030 and 2040 given the need in H2.

Flexible model with a set of parameters that can be adapted to the scenario wanted (more explanations in the streamlit app).

# 2- Location of the stations :

Given the number of stations found in the previous part we try to find the best location for each station.

This relies on the capacity of the station, the traffic on the roads, the proximity to trucks activity areas. With these data we compute a fitness score to define best stations and then grid search the best permutations of these stations accross the possible points on the roads per region given the optimal number of stations from part 1.

# 3- Deployment plan :

After finding the best locations for the stations we want to deploy them in a strategic timeline and simulate that in different competition scenarios.


First Scenario : we are in monopoly. Then we just have to define a timeline and build the stations according to two concepts : the score of each station and the fact that we want to spread them so we cover a large part of the territory.


Second Scenario : we are in duopolyy. We modelise the competition in a game theory framework with a Multi Agent Reinforcement Learning model where eac player will build stations on the steps of the previously defined timeline
