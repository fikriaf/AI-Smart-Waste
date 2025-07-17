from roboflow import Roboflow
rf = Roboflow(api_key="QO6LigDvwLNNNkRHdCn3")
project = rf.workspace("ellbendls-p1vkz").project("project1-adrzg")
version = project.version(3)
dataset = version.download("yolov8")