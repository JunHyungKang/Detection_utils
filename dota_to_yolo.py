# YOLO label format 만들기
import os
path = os.path.join('D://Dacon/Arirang_Dataset', 'train_tr_split', 'labelTxt')
path_yolo = os.path.join('D://Dacon/Arirang_Dataset', 'train_tr_split', 'labels')

filelist = os.listdir(path)
categories = {'small_ship': 0, 'large_ship': 1, 'civilian_aircraft': 2, 'military_aircraft': 3,
              'small_car': 4, 'bus': 5, 'truck': 6, 'train': 7, 'crane': 8, 'bridge': 9, 'oil_tank': 10,
              'dam': 11, 'athletic_field': 12, 'helipad': 13, 'roundabout': 14, 'etc': 15}
img_size = 640
for fullname in filelist:
    f = open(os.path.join(path, fullname), 'r')
    lines = f.readlines()
    if len(lines) > 0:
        ff = open(os.path.join(path_yolo, fullname), 'w')
        ff.close()
        for line in lines:
            line_temp = line.split(' ')[:-1]  # (0, 1), (2, 3), (4, 5), (6, 7)
            xmin = min(float(line_temp[0]), float(line_temp[2]), float(line_temp[4]), float(line_temp[6]))
            xmax = max(float(line_temp[0]), float(line_temp[2]), float(line_temp[4]), float(line_temp[6]))
            ymin = min(float(line_temp[1]), float(line_temp[3]), float(line_temp[5]), float(line_temp[7]))
            ymax = max(float(line_temp[1]), float(line_temp[3]), float(line_temp[5]), float(line_temp[7]))
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin
            ff = open(os.path.join(path_yolo, fullname), 'a')
            ff.write(str(categories[line_temp[-1]]) + ' ' + str(x/img_size) + ' ' + str(y/img_size) + ' ' + str(w/img_size) + ' ' + str(h/img_size) + '\n')
    f.close()
