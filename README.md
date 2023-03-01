# airliquide-sustainability-challenge

## About the data :

# 1- Government data

File in shp format, necessary to keep the four other files with same name in different formats to correctly load it.

Some data are from 2018 because last updates from the French data website have corrupted the latest data file.


RRN : files on the road shape with coordinates

airesPL : location of heavytrucks hubs in France

Regions : contour of french regions in shp file

traffic : data on the traffic per road section including heavytrucks


# 2- Other CSV files 

Complementary data about the corresponding region to each department & the location of gas station already existing.

The gas station file has coordinates not in the same format than the government data so we implement a translation algorithm.


DepReg : file associating regions to departements

stations : location of gas station in France


## Remark : 
There are no national roads in Corsica so we might not want to include this region in our analysis.
