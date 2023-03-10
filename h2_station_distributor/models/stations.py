import geopandas as gpd
import sys
import yaml

sys.path.insert(1, "../")
from utils.load_data import *
import math

with open("../config.yaml") as f:
    config = yaml.safe_load(f)
import scipy


class Stations:
    def __init__(self):
        self.df = None
        self.points = None
        self.shapefile = gpd.read_file("../" + config["tmja"])
        self.dfad = gpd.read_file("../" + config["airesPL"])

    def preprocess(self):
        df = self.shapefile
        df.loc[:, "ratio_PL"] = df["ratio_PL"].apply(lambda x: x / 100 if x > 40 else x)
        df = df[df["ratio_PL"] > 0.1]
        df["trucks"] = df["ratio_PL"] * df["TMJA"] / 100
        # trucks_PL * distance(km) * 8 KgH2/100km * % trucks H2
        df["H2"] = df["trucks"] * (df["longueur"] / 1000) * 8 / 100 * 1 / 60
        df["prD"] = df["prD"] + "_" + df["route"]
        df["prF"] = df["prF"] + "_" + df["route"]
        df = df[
            [
                "depPrD",
                "depPrF",
                "prD",
                "xD",
                "yD",
                "prF",
                "xF",
                "yF",
                "ratio_PL",
                "TMJA",
                "trucks",
            ]
        ]
        self.df = df
        self.dfad["center"] = self.dfad["geometry"].apply(
            lambda x: tuple(x.centroid.coords)[0]
        )
        return df

    def get_points(self, departement):
        self.preprocess()
        temp = self.df[self.df["depPrF"] == departement]

        points = []
        x = []
        y = []
        trucks = []
        coords = []

        for point in temp.prF.unique():
            points.append(point)
            x.append(temp.loc[temp.prF == point, "xF"].iloc[0])
            y.append(temp.loc[temp.prF == point, "yF"].iloc[0])
            trucks.append(round(temp[temp.prF == point].sum()["trucks"]))
            coords.append((x[-1], y[-1]))

        for point in temp.prD.unique():
            if point not in temp.prF.unique():
                points.append(point)
                x.append(temp.loc[temp.prD == point, "xD"].iloc[0])
                y.append(temp.loc[temp.prD == point, "yD"].iloc[0])
                trucks.append(round(temp[temp.prD == point].sum()["trucks"]))
                coords.append((x[-1], y[-1]))
        return pd.DataFrame({"point": points, "coords": coords, "trucks": trucks})

    def add_dist(self, points):  # sourcery skip: avoid-builtin-shadow
        mdist = []
        for i in range(points.shape[0]):
            min = 1000000
            for j in range(points.shape[0]):
                temp = math.dist(points.iloc[i, 1], self.dfad.iloc[j, -1])
                if temp < min:
                    min = temp
            mdist.append(min)
        return points.assign(mindist=mdist)

    def prepare_and_fitness_big(self, department):
        # sourcery skip: class-extract-method
        points = self.get_points(department)
        points_dist = self.add_dist(points)

        md = np.max(points_dist["mindist"])
        points_dist["trucks_original"] = points_dist["trucks"]
        points_dist["trucks"] = points_dist["trucks"] / np.max(points_dist["trucks"])
        points_dist["mindist"] = points_dist["mindist"] / md

        # Here distance to aires is not that important
        points_dist["fitness"] = (
            1.5 * points_dist["trucks"] + 0.1 * points_dist["mindist"]
        )
        points_dist["fitness"] = points_dist["fitness"].apply(lambda x: max(x, 0))
        return points_dist

    def prepare_and_fitness_medium(self, department, b):
        points = self.get_points(department)
        points_dist = self.add_dist(points)

        md = np.max(points_dist["mindist"])
        points_dist["trucks_original"] = points_dist["trucks"]
        points_dist["trucks"] = points_dist["trucks"] / np.max(points_dist["trucks"])
        points_dist["mindist"] = points_dist["mindist"] / md

        # Here trafic is less important, and distance more
        points_dist["fitness"] = points_dist["trucks"] + 0.3 * points_dist["mindist"]

        # Also need to make sure you are far from big stations
        mind_others = []
        for i in range(points_dist.shape[0]):
            mint = 100000
            for j in range(b.shape[0]):
                tpt = math.dist(points_dist.iloc[i, 1], b.iloc[j, 1])
                if tpt < mint:
                    mint = tpt
            mind_others.append(mint)
        points_dist["mind_others"] = mind_others

        points_dist["fitness"] = points_dist["fitness"] + 0.5 * points_dist[
            "mind_others"
        ] / max(mind_others)
        points_dist["fitness"] = points_dist["fitness"].apply(lambda x: max(x, 0))

        return points_dist.drop("mind_others", axis=1)

    def prepare_and_fitness_small(self, department, others):
        points = self.get_points(department)
        points_dist = self.add_dist(points)

        md = np.max(points_dist["mindist"])
        points_dist["trucks_original"] = points_dist["trucks"]
        points_dist["trucks"] = points_dist["trucks"] / np.max(points_dist["trucks"])
        points_dist["mindist"] = points_dist["mindist"] / md

        # Here trafic is less important, and distance more
        points_dist["fitness"] = (
            0.5 * points_dist["trucks"] + 0.6 * points_dist["mindist"]
        )

        # Also need to make sure you are far from big stations
        mind_others = []
        for i in range(points_dist.shape[0]):
            mint = 100000
            for j in range(others.shape[0]):
                tpt = math.dist(points_dist.iloc[i, 1], others.iloc[j, 1])
                if tpt < mint:
                    mint = tpt
            mind_others.append(tpt)
        points_dist["mind_others"] = mind_others

        points_dist["fitness"] = points_dist["fitness"] + 2 * points_dist[
            "mind_others"
        ] / max(mind_others)
        points_dist["fitness"] = points_dist["fitness"].apply(lambda x: max(x, 0))

        return points_dist.drop("mind_others", axis=1)

    def get_best_locations(self, df, nums, type):  # sourcery skip: avoid-builtin-shadow

        max = 0
        for i in range(df.shape[0]):
            for j in range(df.shape[0]):
                td = math.dist(df.iloc[i, 1], df.iloc[j, 1])
                if i != j and td > max:
                    max = td

        print("got max")

        best = []
        iters = int(2 * scipy.special.binom(df.shape[0], nums))
        for i in range(iters):
            temp = df.sample(nums)
            fit = temp["fitness"].sum()
            md_in = []
            ra = range(nums)
            t = 100000
            for j in ra:
                for k in ra:
                    if math.dist(temp.iloc[j, 1], temp.iloc[k, 1]) < t and k != j:
                        t = math.dist(temp.iloc[j, 1], temp.iloc[k, 1])
            fit -= t / max

            if len(best) == 0 or fit > best[0]:
                best = [fit, t, temp.reset_index().iloc[:, 1:]]
            step = round(iters / 20) + 1
            if i % step == 0:
                print(f"{int(i/step)*5} % done")
        if type == "big":
            best[2]["type"] = "big"
            best[2]["profitability"] = (
                best[2]["trucks_original"] * 0.5 * 40 - 0.6 * 4000
            )
        elif type == "medium":
            best[2]["type"] = "medium"
            best[2]["profitability"] = (
                best[2]["trucks_original"] * 0.5 * 40 - 0.8 * 3000
            )
        else:
            best[2]["type"] = "small"
            best[2]["profitability"] = (
                best[2]["trucks_original"] * 0.5 * 40 - 0.9 * 2000
            )

        best[2]["profitable"] = best[2]["profitability"].apply(lambda x: int(x > 0))
        return best

    def solution(self, department, nums):
        big = self.get_best_locations(
            self.prepare_and_fitness_big(department), nums, "big"
        )
        medium = self.get_best_locations(
            self.prepare_and_fitness_medium(department, big[2]), nums, "medium"
        )
        temp = pd.concat([big[2], medium[2]]).reset_index().iloc[:, 1:]
        small = self.get_best_locations(
            self.prepare_and_fitness_small(department, temp), nums, "small"
        )
        return pd.concat([big[2], medium[2], small[2]])
