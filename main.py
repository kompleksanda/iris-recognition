import os
import cv2
import pickle
import numpy as np
import pandas as pd
from skimage.morphology import square, disk, dilation, erosion
from skimage import draw
from math import sin, cos, radians
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from imutils import paths
from loc import IrisLoc, normalize, enhancement

name_label = ["Alex"]

WINDOW_SIZE = 450
HEIGHT = 0
WIDTH = 0
N_COMP0NENTS = 150

def show(image):
    return
    cv2.imshow("image", image)
    cv2.waitKey(0)

def get_radius(image_out):
    global HEIGHT, WIDTH
    #create a mask that picks brigtness between 3 and 50
    pupil_image = np.where((image_out < 50) & (image_out > 3), 1., 0.)
    #erode
    selem = disk(1)
    pupil_image = erosion(pupil_image, selem)
    #dilate
    selem = disk(1)
    pupil_image = dilation(pupil_image, selem)
    # create a mask to pick center of image
    height, width = pupil_image.shape
    circle_img = np.zeros((height,width), np.uint8)
    cv2.circle(circle_img,(int(width/2), int(height/2)), 95, 1, thickness=-1)
    pupil_image = cv2.bitwise_and(pupil_image, pupil_image, mask=circle_img)
    show(pupil_image)


    #Select the pupil in a rectangle
    rows = pupil_image.shape[0] #height
    cols = pupil_image.shape[1] #width
    if not(HEIGHT): HEIGHT = rows
    if not(WIDTH): WIDTH = cols
    
    for col in range(cols):
        #start from east(right) of image
        col = cols - 1 - col
        #pick the first col that's isn't black
        if sum(pupil_image[:,col]) > 0:
            east_mark = col
            break

    for col in range(east_mark):
        #start from west(left) of east_mark
        col = east_mark - 1 - col
        #pick the first col that's totally black
        if sum(pupil_image[:,col]) == 0:
            west_mark = col
            break

    for row in range(rows):
        #start from south(bottom) of image
        row = rows - 1 - row
        if sum(pupil_image[row,:]) > 0:
            south_mark = row
            break

    for row in range(south_mark):
        #start from north(up) of south_mark
        row = south_mark - 1 - row
        if sum(pupil_image[row,:]) == 0:
            north_mark = row
            break
    #get the axis centers of the rectangle 
    center_x = int((west_mark + east_mark) / 2)
    center_y = int((north_mark + south_mark) / 2)

    lines = np.zeros([rows,cols])
    rr, cc = draw.line(south_mark,east_mark,north_mark,east_mark)
    lines[rr,cc] = 1
    rr, cc = draw.line(south_mark,west_mark,north_mark,west_mark)
    lines[rr,cc] = 1
    rr, cc = draw.line(south_mark,west_mark,south_mark,east_mark)
    lines[rr,cc] = 1
    rr, cc = draw.line(north_mark,west_mark,north_mark,east_mark)
    lines[rr,cc] = 1
    rr, cc = draw.circle(center_y,center_x,3)
    lines[rr,cc] = 1

    #create a mask that picks brigtness between 50 and 120
    iris_image = np.where((image_out > 50) & (image_out < 120), 1., 0.)
    #erode
    selem = disk(3)
    iris_image = erosion(iris_image, selem)
    #dilate
    selem = disk(1)
    iris_image = dilation(iris_image, selem)
    show(iris_image)

    x = east_mark
    while(iris_image[center_y,x]) == 1: x += 1
    iris_east = x

    x = west_mark
    while(iris_image[center_y,x]) == 1: x -= 1
    iris_west = x

    rr, cc = draw.line(0,iris_east,rows-1,iris_east)
    lines[rr,cc] = 1
    rr, cc = draw.line(0,iris_west,rows-1,iris_west)
    lines[rr,cc] = 1



    # Displaying bounding boxes with lines
    full_color = np.zeros([rows,cols,3])
    for i in range(rows):
        for j in range(cols):
            full_color[i,j,0] = pupil_image[i,j]
            full_color[i,j,1] = lines[i,j]

    for i in range(rows):
        for j in range(cols):
            full_color[i,j,2] = iris_image[i,j]

    #print('Eastern distance: ' + str(iris_east - center_x))
    #print('Western distance: ' + str(center_x - iris_west))
    #disp(full_color)

    # Generating mask:
    radius = max([(iris_east - center_x),(center_x - iris_west)])
    mask = np.zeros([rows,cols])

    rr, cc = draw.circle(center_y, center_x,radius)
    for i in range(len(rr)):
        if rr[i] < 0: rr[i] = 0
        if rr[i] >= rows: rr[i] = rows - 1
    for i in range(len(cc)):
        if cc[i] < 0: cc[i] = 0
        if cc[i] >= cols: cc[i] = cols - 1
    mask[rr,cc] = 1

    rr, cc = draw.circle(center_y, center_x,(0.5*(east_mark-west_mark)))
    for i in range(len(rr)):
        if rr[i] < 0: rr[i] = 0
        if rr[i] >= rows: rr[i] = rows - 1
    for i in range(len(cc)):
        if cc[i] < 0: cc[i] = 0
        if cc[i] >= cols: cc[i] = cols - 1
    mask[rr,cc] = 0


    # img = bnw(fname)
    # img = fname
    pad = 0
    masked_eye = np.zeros([image_out.shape[0]-2*pad,image_out.shape[1]-2*pad])
    for i in range(rows):
            for j in range(cols):
                masked_eye[i,j] = min([mask[i,j],image_out[pad+i,pad+j]])

    check_mask = np.zeros([rows,cols,3])
    for i in range(rows):
        for j in range(cols):
            check_mask[i,j,0] = image_out[i,j] * 0.8
            check_mask[i,j,1] = image_out[i,j] * (0.8 + 0.2*mask[i-2*pad,j-2*pad])
            check_mask[i,j,2] = image_out[i,j] * (0.8 + 0.2*mask[i-2*pad,j-2*pad])
    
    show(check_mask)
    inner_radius = 0.5 * (east_mark - west_mark) #pupil radius
    outer_radius = 0.5 * (iris_east - iris_west) #iris radius
    center_r, center_c = center_y,center_x #center coordinates
    return [inner_radius, outer_radius, (center_r, center_c)]

def get_iris_portion(image_out, params):
    Ri = params[0] # Inner (pupil) radius
    Ro = params[1] # Outer (iris) radius
    y,x = params[2] # center coordinates
    x -= 10
    y -= 10

    H = int(Ro - Ri)
    W = 360
    newmap = np.zeros([H,W])
    for r in range(H):
        for c in range(W):
            mapped_point_col = int(x + (Ri + r) * cos(radians(c)))
            mapped_point_row = int(y + (Ri + r) * sin(radians(c)))
            dot = image_out[min([image_out.shape[0]-1,mapped_point_row]), min([image_out.shape[1]-1,mapped_point_col])]
            newmap[r,c] = dot
    # print("newmap[5:44,:]",len(newmap[5:44,:]))
    if len(newmap[5:44,:]) !=0:
        return newmap[5:44,:]  
    else:
        print("rectangle expression1")  
        return "invalid image"

def group_feat_sel(strip):
    if strip == "invalid image":
        # print("feature vector 1")
        return "invalid image"

    grid = np.zeros([13, 36])
    for i in range(13):
        for j in range(36):
            block = strip[3 * i:3 * i + 3, 10 * j:10 * j + 10]
            for row in block:
                grid[i, j] += sum(row)

    # Group encoding
    def encode(group):
        avg = sum(group) / 5
        group -= avg
        for i in range(1, 5):
            group[i] = sum(group[:i + 1])
        code = ''
        argmax = 0
        argmin = 0
        for i in range(5):
            if group[i] == max(group): argmax = i
            if group[i] == min(group): argmin = i
        for i in range(5):
            if i < argmax and i < argmin: code += '0'
            if i > argmax and i > argmin: code += '0'
            if i >= argmax and i <= argmin: code += '2'
            if i <= argmax and i >= argmin: code += '1'
        return code

    # Horizontal grouping
    horgroups = []
    # hor_ver_groups = []
    hor_ver_groups = ""
    for row in range(13):
        horgroups.append([])
        for col in range(32):
            group = np.zeros(5)
            for i in range(5): group[i] = grid[row, col + i]
            horgroups[row].append(encode(group))
            # hor_ver_groups.append(encode(group))
            # hor_ver_groups += encode(group)

    # Vertical grouping
    vergroups = []
    for col in range(36):
        vergroups.append([])
        for row in range(9):
            group = np.zeros(5)
            for i in range(5): group[i] = grid[row + i, col]
            vergroups[col].append(encode(group))
            # hor_ver_groups.append(encode(group))
            # hor_ver_groups += encode(group)


    return [horgroups, vergroups]

def serialize(encoding):
    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": encoding, "names": "default_name"}
    f = open("encoding.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()
    print ("OK")

def strip_image(path_to_image):
    image = cv2.imread(path_to_image)
    show(image)
    #Convert to grayscale
    image_out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show(image_out)
    return image_out

    params = get_radius(image_out)
    strip = get_iris_portion(image_out, params)

    return strip


#new = group_feat_sel(strip)
#ser = serialize(new)
#print('Strip: ', strip)

def to_svd(strip, name_label, n_component=4, random_state=2018):
    X = []
    labels = []
    scaler = StandardScaler()
    svd = TruncatedSVD(n_components=n_component, random_state=random_state)
    scaled = scaler.fit_transform(strip)
    X.append(svd.fit_transform(scaled))
    labels.append([name_label])
    X = np.vstack(X)
    return X

def iris_test_model(train_db_path):
    directory_list = list()
    for root, dirs, files in os.walk(
            train_db_path,
            topdown=False):
        for name in dirs:
            directory_list.append(os.path.join(root, name))

    print ("directory_list", directory_list)
    iris_names=[]
    iris_name_encodings=[]
    invalid_image=False
    for directory in directory_list:
        # grab the paths to the input images in our dataset
        paths_to_images = list(paths.list_files(os.path.join(directory)))
        # initialize the list of iris_name_encodings and iris_names
        iris_encodings = []
        name = directory.split(os.path.sep)[-1]
    
        print ("name",name)
        # Encode the images located in the folder to thier respective numpy arrays
        invalid_image=False
        for path_to_image in paths_to_images:
            print ("path_to_image",path_to_image)
            # image = scipy.misc.imread(path_to_image)
            iris_encodings_in_image = strip_image(path_to_image)
            locali = IrisLoc(iris_encodings_in_image, name)
            norma = normalize(locali, name)
            enh = enhancement(norma)
            svdd = to_svd(enh, name)  
            if iris_encodings_in_image == "invalid image":
                iris_encodings_in_image = iris_encodings_in_image
                invalid_image=True
            else:
                #iris_encodings_in_image = to_svd(iris_encodings_in_image, name)
                iris_names.append([name])
                iris_name_encodings.append(svdd)
            #if invalid_image == True :
            #    print("invalid_image",name)
            #    invalid_image=False
            #else:
            #   iris_names.append(name)     
            #    #iris_name_encodings.append(iris_encodings)
            #    iris_name_encodings.append(iris_encodings_in_image)

    print ("train_db_model_path",len(iris_names),len(iris_name_encodings))
    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    print(len(iris_name_encodings))
    #X = pd.DataFrame(iris_name_encodings)
    X = np.asarray(iris_name_encodings)
    
    print(X.shape)
    ix, iy, iz = X.shape
    X = X.reshape(ix, iy*iz)
    print(X.shape)
    df_label = pd.DataFrame(iris_names)
    print(df_label.shape)

    pickle.dump(X, open("data/X_iris_svd.p","wb"))
    pickle.dump(df_label, open("data/y_iris_svd.p","wb"))
    print("Done")
    return iris_names

iris_test_model("Input_database")