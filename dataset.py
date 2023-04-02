import random
import sqlite3
import math
from datetime import datetime
from bisect import bisect_left, bisect_right

select_carbs_query = (
    "select carbs, timestamp from Treatments where carbs <> 0 order by timestamp;"
)
select_insulin_query = (
    "select insulin, timestamp from Treatments where insulin <> 0 order by timestamp;"
)
select_glucose_query = "select (calculated_value * 0.0555) as glucose, timestamp from BgReadings order by timestamp;"


class Datapoint:
    def __init__(
        self,
        current_bg: float,
        average_bg: float,
        next_bg: float,
        last_carbs: float,
        last_carbs_time: float,
        last_insulin: float,
        last_insulin_time: float,
    ):
        self.current_bg = current_bg
        self.average_bg = average_bg
        self.next_bg = next_bg
        self.last_carbs = last_carbs
        self.last_carbs_time = last_carbs_time
        self.last_insulin = last_insulin
        self.last_insulin_time = last_insulin_time

    @staticmethod
    def format_seconds(S: float) -> str:
        h = math.floor(S / 3600)
        s = math.floor(S % 60)
        m = math.floor(S / 60 % 60)
        if S >= 3600:
            return f"{h:02.0f}h{m:02.0f}m{s:02.0f}s"
        elif S >= 60:
            return f"{m:02.0f}m{s:02.0f}s"
        else:
            return f"{s:02.0f}s"

    def __str__(self) -> str:
        return f"Datapoint{{current={self.current_bg:.3f}; " \
               f"next={self.next_bg:.3f}; " \
               f"avg={self.average_bg:.3f}; " \
               f"carbs={self.last_carbs:.0f}/{Datapoint.format_seconds(self.last_carbs_time)}; " \
               f"insulin={self.last_insulin:.0f}/{Datapoint.format_seconds(self.last_insulin_time)}}}"


def get_data(
    name: str = "data/database.sqlite",
) -> tuple[
    list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]
]:
    """
    Fetches data from sql database
    timestamps are unix seconds
    :return: Returns a tuple containing (in order):
    [0] - list of tuples of glucose values tuple[value, unix_timestamp]
    [1] - list of tuples of carb values tuple[value, unix_timestamp] (mmol/L)
    [2] - list of tuples of insulin values tuple[value, unix_timestamp]
    """
    con = sqlite3.connect(name)
    cur = con.cursor()

    glucose: list[tuple[float, float]] = list(
        map(
            lambda x: (x[0], x[1] * 0.001), cur.execute(select_glucose_query).fetchall()
        )
    )
    carbs: list[tuple[float, float]] = list(
        map(lambda x: (x[0], x[1] * 0.001), cur.execute(select_carbs_query).fetchall())
    )
    insulin: list[tuple[float, float]] = list(
        map(
            lambda x: (x[0], x[1] * 0.001), cur.execute(select_insulin_query).fetchall()
        )
    )

    return glucose, carbs, insulin


def test_get_data() -> None:
    glucose, carbs, insulin = get_data("data/export20230402-102556.sqlite")

    print(f"Glucose: {len(glucose)}")
    print(f"Carbs:   {len(carbs)}")
    print(f"Insulin: {len(insulin)}")

    dataset = prep_data(glucose, carbs, insulin)
    print(f"Total data points: {len(dataset)}")

    for i in range(min(10, len(dataset))):
        print(dataset[i])


def prep_data(
    glucose: list[tuple[float, float]],
    carbs: list[tuple[float, float]],
    insulin: list[tuple[float, float]],
    average_interval: float = 3600,
    next_interval: float = 600,
    margin: float = 60,
) -> list[Datapoint]:
    res: list[Datapoint] = []

    for i in range(len(glucose)):
        current_bg = glucose[i][0]
        current_time = glucose[i][1]

        # +1/-1 to account for floating point comparison

        # find next bg
        # it must
        #  - exist
        #  - be about current-time+next_interval in timeline
        #  - be
        j = -1
        for ind in range(i+1, len(glucose)):
            if abs(glucose[ind][1] - (current_time+next_interval)) <= margin:
                j = ind
                break
            elif glucose[ind][1] > current_time+next_interval+margin:
                break
        if j == -1:
            # could not find appropriate next bg
            continue
        next_bg = glucose[j][0]

        # find last carbs value
        # it must
        # - exist
        # - TODO
        j = bisect_left(carbs, current_time - 1, key=lambda x: x[1]) - 1
        if j < 0 or j >= len(carbs) or carbs[j][1] > current_time:
            # could not find appropriate previous carbs
            continue
        last_carbs = carbs[j][0]
        last_carbs_time = current_time - carbs[j][1]

        # find last insulin value
        # reqs:
        # - TODO
        j = bisect_left(insulin, current_time - 1, key=lambda x: x[1]) - 1
        if j < 0 or j >= len(insulin) or insulin[j][1] > current_time:
            # could not find appropriate previous carbs
            continue
        last_insulin = insulin[j][0]
        last_insulin_time = current_time - insulin[j][1]

        # find the earliest glucose needed in average
        # consider only if there's 60 min of data available
        j = bisect_left(
            glucose, current_time - average_interval + 1, hi=i - 1, key=lambda x: x[1]
        )
        if j >= i or glucose[j][1] < current_time - average_interval - 60:
            # there is no data to calculate average
            continue

        # TODO: interpolate linearly between measurements to get better average :D
        # PS. hope that helps
        average_bg = 0.0
        # weighted by the number of min that each reading covers
        for k in range(j, i):
            average_bg += glucose[k][0] * (glucose[k + 1][1] - glucose[k][1])
        average_bg /= glucose[i][1] - glucose[j][1]

        res.append(
            Datapoint(
                current_bg,
                average_bg,
                next_bg,
                last_carbs,
                last_carbs_time,
                last_insulin,
                last_insulin_time,
            )
        )

    return res


def main(filename: str) -> list[Datapoint]:
    glucose, carbs, insulin = get_data(filename)
    return prep_data(glucose, carbs, insulin)




if __name__ == "__main__":
    test_get_data()