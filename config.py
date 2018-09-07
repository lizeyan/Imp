from sacred import Experiment, SETTINGS
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

VISDOM_SERVER = "http://lzy-pc"
VISDOM_PORT = 10201


def init_sacred(name: str):
    experiment = Experiment(name)
    experiment.observers.append(
        MongoObserver.create(
            "mongodb://lizytalk:I was a cat.@data1,data2,data3/admin?replicaSet=rs0",
            db_name="lizytalk",
        )
    )
    # capture all contents written into sys.stdout and sys.stderr
    SETTINGS["CAPTURE_MODE"] = "sys"
    experiment.captured_out_filter = apply_backspaces_and_linefeeds
    return experiment
