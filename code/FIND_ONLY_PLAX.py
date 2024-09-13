import os

path = r"H:\二尖瓣项目\dataset\3 class dataset\test\Barlow"
patients = os.listdir(path)
for p in patients:
    if len(p.split("."))!=2:
        patient_path = os.path.join(path, p)
        view_list = os.listdir(patient_path)
    if len(view_list) == 1 and view_list[0] == "PLAX":
        print(p)
