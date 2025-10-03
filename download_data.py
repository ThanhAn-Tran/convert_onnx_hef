
from roboflow import Roboflow
rf = Roboflow(api_key="tZvDiMmpYHE0lWMKx6D5")
project = rf.workspace("airz-x").project("paper_ball-m6mum")
version = project.version(4)
dataset = version.download("yolov11")
                