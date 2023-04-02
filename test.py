import sqlite3
from datetime import datetime
import time
from typing import Iterable


def get_data(db_name: str = "data/database.db") -> Iterable[tuple[float, float]]:
    con = sqlite3.connect(db_name)
    cur = con.cursor()
    res = cur.execute("select glucose, timestamp from measurements order by 2")
    return map(
        lambda x: (
            x[0],
            time.mktime(datetime.strptime(x[1], "%Y-%m-%d %H:%M:%S").timetuple()),
        ),
        res.fetchall(),
    )


def main() -> None:
    data = list(get_data())
    for i in range(min(100, len(data))):
        print(f"{data[i][0]:.1f} @ {datetime.utcfromtimestamp(data[i][1]).strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
