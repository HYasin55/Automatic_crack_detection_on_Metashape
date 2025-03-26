"""
Selecting the cameras respect to their position relative to object surface.
Projection of a crack edges on mesh and measuring the crack widths on 3D.
H. Yasin Ozturk - March 2025
"""
####################
import Metashape
import cv2, os, math
import numpy as np
import supervision as sv
import skimage.morphology
from skimage.morphology import skeletonize
from skimage import feature, draw
from roboflow import Roboflow

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from unet.unet_transfer import UNet16, input_size
from utils import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34

DEVICE = 'mps' # Tested on MacBook. Change this parameter depending on the system used.
if "__file__" in globals():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
####################
## FUNCTIONS
def crack_detection_alghorithm (image_directory, model, cut_off_threshold = 0.4, stride_ratio = 0.5, resize_ratio = 1, method = 'patched'):
    """
    Applies crack segmentation model on the selected image
    Args:
        image_directory (str): Directory of the image.
        model (UNetResNet): Crack detection model to be aplied.
        cut_off_threshold (float): Probability threshold for crack segmentation. Defined between 0.0-1.0.
        stride_ratio (float): Parameter for overlapping. 1.0 means no overlap while 0.5 means 50% overlap between each interface for sliding window.
        resize_ratio (float): Downsampling the image. 1 means using original, 0.5 corresponds to width and heigh is reduced to half (quarter in terms of area)
        method (str): Using the sliding window method or downsampled image. 'patched' or 'full' respectively.
    Returns:
        Binary segmentation (ndarray boolen).
    """
    image = cv2.imread(image_directory)
    if resize_ratio is not 1:
        image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
        # print(resize_ratio, image.shape)
    input_width, input_height = input_size[0], input_size[1]
    image_height, image_width, image_channels = image.shape
    stride = input_width * stride_ratio

    if method == 'patched':
        padded_image = pad_image(image, stride)
        prob_map_patch = evaluate_img_patch(model, padded_image, stride)
        prob_map_patch = crop_image(prob_map_patch , (image_height, image_width, 1))
        prob_map_patch = prob_map_patch/prob_map_patch.max()
        if resize_ratio is not 1:
            prob_map_patch = cv2.resize(prob_map_patch, None, fx=1/resize_ratio, fy=1/resize_ratio, interpolation=cv2.INTER_AREA)
        prob_map_patch_coor = prob_map_patch > cut_off_threshold
        return prob_map_patch_coor
    elif method == 'full':
        prob_map_full = evaluate_img(model, image)
        prob_map_full_coor = prob_map_full > cut_off_threshold
        return prob_map_full_coor
    else:
        print('Wrong method!')

def pad_image(image, pad_size, mode="reflect"):
    """
    Pads the RGB image using the specified padding mode.
    Args:
        image (numpy array): Input RGB image (H x W x C).
        pad_size (int): Number of pixels to pad on all sides.
        mode (str): Padding mode (default is 'reflect').
    Returns:
        Padded image.
    """
    pad_size = int(pad_size)  # Ensure pad_size is an integer
    return np.pad(image, pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode=mode)

def crop_image(padded_image, original_shape):
    """
    Crops the padded RGB image back to the original shape.
    Args:
        padded_image (numpy array): Padded RGB image (H x W x C).
        original_shape (tuple): Shape of the original image (H, W, C).
    Returns:
        Cropped image.
    """
    img_height, img_width, _ = original_shape
    padded_height, padded_width = padded_image.shape
    pad_h = (padded_height - img_height) // 2
    pad_w = (padded_width - img_width) // 2
    return padded_image[pad_h:padded_height-pad_h, pad_w:padded_width-pad_w]    
  
def add_black_border(binary_image, pixel_edge_thickness = 3):
    # Change the edge pixels to black to handle edge cases
    binary_image[:pixel_edge_thickness, :] = 0
    binary_image[-pixel_edge_thickness:, :] = 0
    binary_image[:, :pixel_edge_thickness] = 0
    binary_image[:, -pixel_edge_thickness:] = 0
    return binary_image

def simplify_contours(binary_image, epsilon=2):
    # Discrete Curve Evolution (DCE) to simplify geometry of crack. Change the epsilon for simplification level.
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified_image = np.zeros_like(binary_image)
    for contour in contours:
        # Approximate the contour with reduced points
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(simplified_image, [approx_contour], -1, 1, thickness=-1)
    return simplified_image

def get_skeleton(image, epsilon = 2, skeleton_method = 2):
    # Get the skeleton of the binary image with the required method
    dce_image = simplify_contours(image, epsilon = epsilon)
    if skeleton_method == 1:
        skeleton = skeletonize(dce_image, method = 'zhang')
    elif skeleton_method == 2:
        skeleton = skeletonize(dce_image, method = 'lee')
    else:
        raise SystemExit('Error: Use avaliable methods for skeletonize function')
    return skeleton

def get_directory_only(full_path, image_name):
    # Gets the direction of the image withouth the name of the image.
    directory = os.path.dirname(full_path)
    if image_name in os.path.basename(full_path):
        return directory
    return full_path

def save_and_get_image(camera, image):
    # Saves a image and returns on the directory
    image_path = camera.photo.path
    image_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)
    image_array = np.fromstring(image.tostring(), dtype=np.uint8)
    image_array_metashape = Metashape.Image.fromstring(image_array, image.width, image.height, 'RGB', datatype='U8')
    new_folder = image_dir + "/metashape_images"
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    new_directory = new_folder + '/metashape_' + image_name
    image_array_metashape.save(new_directory)
    return new_directory

def combine_mask_and_images (metashape_image, metashape_mask):
    """
    Combines an image with its corresponding mask by applying the mask to the image.
    Args:
        metashape_image (Metashape.Image): The image to which the mask will be applied.
        metashape_mask (Metashape.Mask): The mask to apply to the image.
    Returns:
        Metashape.Image: The resulting image after applying the mask.
    """
    image_array = np.fromstring(metashape_image.tostring(), dtype=np.uint8)
    mask_image = metashape_mask.image()
    mask_array = np.fromstring(mask_image.tostring(), dtype=np.uint8)
    mask_array_rgb = np.repeat(mask_array, 3)
    image_array[mask_array_rgb ==0] = 0

    masked_image = Metashape.Image.fromstring(image_array, metashape_image.width, metashape_image.height, 'RGB', datatype='U8')
    return masked_image

def save_and_get_new_path_masked_image(masked_image, original_name, original_path):
    directory = get_directory_only(original_path, original_name)
    new_folder = directory + "/images_combined_with_masks"
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    new_directory = new_folder + '/masked_' + original_name + '.jpg'
    masked_image.save(new_directory)
    return new_directory

def add_single_marker(point, marker_name, marker_group, camera, surface):
    """
    Projects a 2D image point onto a 3D surface and adds a marker at the intersection.
    Args:
        point2D_image (list/tuple): 2D coordinates [x, y] in the image, with origin at the upper-left corner.
        marker_name (str): Name/label for the new marker.
        marker_group_name (str): Name of the marker group to assign the marker to.
        camera (Metashape.Camera): Camera object associated with the image.
        surface (Metashape.Model): 3D model surface to project the point onto.
    Returns:
        Metashape.Vector or None: 3D coordinates of the added marker, or None if projection failed.
    """
    point2D = Metashape.Vector([point[1],point[0]])
    sensor = camera.sensor
    # calibration = sensor.calibration
    point3D = surface.pickPoint(camera.center, camera.transform.mulp(sensor.calibration.unproject(point2D)))
    if point3D is None: return None
    marker = chunk.addMarker(point3D)
    marker.label = marker_name
    marker.group = marker_group
    if True: return marker.position
    else: return None

def adding_markers (point_a, point_b, marker_group, camera_index, point_index, camera, surface, scale_bar_toggle, save_toggle = 1):
    """
    Adds paired edge markers representing crack edges to the Metashape project and optionally creates scale bars between them.
    Args:
        Metashape.Vector or None: 3D coordinates of the added marker, or None if projection failed.
        point_a (tuple): (row, col) coordinates of first edge point in image space
        point_b (tuple): (row, col) coordinates of second edge point in image space
        marker_group (Metashape.MarkerGroup): Group to organize related markers
        camera_index (int): Index of source camera in chunk.cameras list
        point_index (int): Unique identifier for this crack measurement
        camera (Metashape.Camera): Camera object containing image data
        surface (Metashape.Model): 3D model surface for projection
        scale_bar_toggle (bool): Enable/disable scale bar creation
        save_toggle (bool): Enable/disable data saving to file
    Outputs:
        - Creates two markers in Metashape project
        - Optional scale bar between markers
        - Appends measurement data to points_data.txt
    """
    name = 'P' + str(camera_index) + '_' + str(point_index)
    name_a = name + '_a'
    name_b = name + '_b'      
    marker_1 = add_single_marker(point_a, name_a, marker_group, camera, surface)
    marker_2 = add_single_marker(point_b, name_b, marker_group, camera, surface)
    if save_toggle:
        if (marker_1 != None) & (marker_2 != None): 
            marker_1 = chunk.crs.project(chunk.transform.matrix.mulp(marker_1))
            marker_2 = chunk.crs.project(chunk.transform.matrix.mulp(marker_2))
            save_midpoint_and_distance(marker_1, marker_2, name, camera_index = camera_index)

    if not scale_bar_toggle: return
    count = 0
    marker_a = None
    marker_b = None
    for marker in chunk.markers:
        if marker.label == name_a:
            marker_a = marker
            count = count + 1
        if marker.label == name_b:
            marker_b = marker
            count = count + 1
        if count == 2:
            break
    if not (marker_a == None or marker_b == None):
        scale_bar = chunk.addScalebar(marker_a, marker_b)
        scale_bar.label = name
    # scale_bar.group = marker_group

def save_midpoint_and_distance(p1, p2, name, camera_index, filename= SCRIPT_DIR+ "/points_data.txt"):
    """
    Computes the midpoint and Euclidean distance between two 3D points,
    then appends the result to a text file. If the file does not exist,
    it creates it and adds a header.

    Parameters:
    - p1: tuple or list of (x, y, z) coordinates for the first point
    - p2: tuple or list of (x, y, z) coordinates for the second point
    - name: identifier for the point pair (e.g., "P1", "P2")
    - filename: the name of the text file where data is stored (default: "points_data.txt")
    """
    # Check if the file exists
    file_exists = os.path.exists(filename)

    # Convert points to NumPy arrays
    p1, p2 = np.array(p1), np.array(p2)

    if p1.any() == None or p2.any() == None: return
    # Compute midpoint
    midpoint = (p1 + p2) / 2

    # Compute Euclidean distance
    distance = np.linalg.norm(p2 - p1)

    index_temp = active_camera.index(camera_index)
    camera_dist = active_distance[index_temp]
    camera_rot_x = active_rot_x[index_temp]
    camera_rot_y = active_rot_y[index_temp]

    # Open file in append mode
    with open(filename, 'a') as f:
        # If file does not exist, write the header first
        if not file_exists:
            f.write("name,middle_x,middle_y,middle_z,distance,camera_dist,camera_rot_x,camera_rot_y\n")

        # Append results with the name
        f.write(f"{name},{midpoint[0]},{midpoint[1]},{midpoint[2]},{distance},{camera_dist},{camera_rot_x},{camera_rot_y}\n")

def point_on_3d(point2D, camera):
    """
    Projects a 2D image point onto the 3D model surface.
    Args:
        point2D (tuple/list): (x, y) coordinates in image space (pixels)
        camera (Metashape.Camera): Source camera with calibration data
    Returns:
        Metashape.Vector: 3D coordinates of surface intersection point
        None: If projection fails (ray doesn't intersect surface)
    Note:
        - Image origin is assumed at top-left corner
    """
    sensor = camera.sensor
    point3D = surface.pickPoint(camera.center, camera.transform.mulp(sensor.calibration.unproject(point2D)))
    if point3D is None: return
    else: return point3D

def calculate_surface_angle(teta, distance_1, distance_2):
    """
    Calculates surface inclination angle using trigonometric relationships.
    Parameters:
        teta (float): Angular difference between measurements [radians]
        distance_1 (float): First measurement distance [meters]
        distance_2 (float): Second measurement distance [meters]
    Returns:
        float: Surface inclination angle in degrees
    Note:
        - Angles 90° indicate orthogonal camera angles
    """
    distance_on_surface = math.sqrt(distance_1**2 + distance_2**2 - 2*distance_1*distance_2*math.cos(teta))
    alpha = math.asin(math.sin(teta)*distance_1/distance_on_surface)
    radian = alpha + teta
    degrees = math.degrees(radian)
    return degrees

# Following functions are modified version of the "A-Hybrid-Method-for-Pavement-Crack-Width-Measurement" script.
# To check the original version please follow the link bellow:
# https://github.com/JeremyOng96/A-Hybrid-Method-for-Pavement-Crack-Width-Measurement

def get_pca_vector(skeleton_region, verbose=False):
    """
    This function uses principal component analysis to determine the orientation of the skeletion region. 
    Every point in the skeleton region is represented by (r,c) but can be represented by (0,r/c) instead by compressing the data.
    The orientation can be found by obtaining the largest eigenvalue corresponding to the eigenvector of the covariance matrix.
    Arguments:
    skeleton_region - a small region of the skeleton.
    
    Return:
    v - local orthogonal basis vector
    """
    
    row, column = np.nonzero(skeleton_region)
    row = sorted(row[np.newaxis,:])
    column = sorted(column[np.newaxis,:])
    data_pts = np.concatenate((row,column),axis = 0).T.astype(np.float64)
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts,mean)
    
    angle = math.atan2(eigenvectors[0,0], eigenvectors[0,1]) * 180 / math.pi # orientation in degrees 
    if verbose:
        print(f"The data points are {data_pts}")
        print(f"The eigenvalues are{eigenvalues}")
        print(f"The eigenvectors are {eigenvectors}")
        print(f"The principal vector angle is {angle}")
    try:
        v = np.asarray([eigenvectors[1,0],eigenvectors[1,1]])
    except IndexError:
        v = []
    return v

def get_constant_region_pca(coord, skeleton):
    """
    This function the orientation using a constant kernel (5x5)
    The output kernel has 2 assumptions:
    (1) Kernel is a square
    (2) Kernel is size is odd i.e. 5x5, 7x7 and so on
    Arguments:
    coord - coordinates of point of interest
    skeleton - the skeleton of the image, obtained by using medial axis transform / thinning / dse and etc
    diff - difference in error between iterations
    
    Return:
    gradient - the gradient at the point of interest
    """
    kernel_size = 5 # Initialize kernel size to be 7x7
    r,c = coord
    max_r ,max_c = skeleton.shape
    angle_1 = angle_2 = 0 # Initialize grad_1 to be gradient at i-1
    eps = 1e-8
    index = 0
    
    width = height = kernel_size // 2        
    skeleton_region = skeleton[r-height:r+height+1,c-width:c+width+1] # gets the skeleton region      
    v = get_pca_vector(skeleton_region)
    
    return v

def get_adaptive_region_pca(coord, skeleton, diff=0.01):
    """
    This function returns an appropriate sized kernel such that the gradient at the point of interest already converges. 
    The output kernel has 2 assumptions:
    (1) Kernel is a square
    (2) Kernel is size is odd i.e. 5x5, 7x7 and so on
    Arguments:
    coord - coordinates of point of interest
    skeleton - the skeleton of the image, obtained by using medial axis transform / thinning / dse and etc
    diff - difference in error between iterations
    
    Return:
    gradient - the gradient at the point of interest
    """
    kernel_size = 5 # Initialize kernel size to be 3x3
    r,c = coord
    max_r ,max_c = skeleton.shape
    angle_1 = angle_2 = 0 # Initialize grad_1 to be gradient at i-1
    eps = 1e-8
    index = 0

    while True:      
        width = height = kernel_size // 2        
        # Handle edge cases
        if r-width < 0 or r+width > max_r or c-height < 0 or c+height > max_c:
            return v
        
        skeleton_region = skeleton[r-height:r+height+1,c-width:c+width+1] # gets the skeleton region      
        v = get_pca_vector(skeleton_region)
        if len(v) > 0:
            angle_2 = math.atan2(v[0], v[1]) * 180 / math.pi # orientation in degrees 
        else:
            return []
            
        grad_rel_error = abs((angle_1-angle_2)/(angle_1+eps))

        if grad_rel_error < diff and index > 0:
            return v
        
        angle_1 = angle_2
        kernel_size += 2
        index += 1

def hybrid_method_to_find_edge_points(pois, distance_transform, binary_image, skeleton_image, boundary_image):
    """
    This function selects the points using the Euclidean distance between two points.
    To obtain the two points, we obtain the projection value based on the orthonormal vector, v.
    
    Return:
    pt1_arr - point on one edge
    pt2_arr - corresponding point on opposite edge
    """
    # Selected points close the edges are effected by image boundary. 
    # A small portion of area close to edge should not be used
    x_width, y_width = binary_image.shape 
    dead_thickness_ratio = 1/75
    lim_size = 1/dead_thickness_ratio
    x_lim = x_width//lim_size
    y_lim = y_width//lim_size

    img = np.reshape((skeleton_image+boundary_image),(*boundary_image.shape,1))
    img = np.ascontiguousarray(img, dtype=np.uint8)
    r_b, c_b = np.nonzero(boundary_image.astype(int))
    boundary_pts = list(zip(r_b,c_b))
    pt1_arr = []
    pt2_arr = []
    collision_pts = list(set(pois) & set(boundary_pts))  
    
    _, labels_bin_img = cv2.connectedComponents(binary_image)
    for poi in pois:
        collision = False
        r_p, c_p = poi
        radius = max(distance_transform[r_p,c_p]*2,6)
        onb = get_adaptive_region_pca(poi,skeleton_image)

        if len(onb) == 0:
            continue
        
        poi_vector = np.asarray(poi)
        label = labels_bin_img[r_p,c_p]
        sub_canny = feature.canny(labels_bin_img == label)
        r_b_sub, c_b_sub = np.nonzero(sub_canny)
        boundary_pts_sub = list(zip(r_b_sub,c_b_sub))
        B_matrix = np.asarray(boundary_pts_sub)
        if poi in boundary_pts_sub:
            collision = True
            index = np.where(np.all(B_matrix == poi_vector,axis = 1))
            B_matrix = np.delete(B_matrix,index,axis=0)
            
        B_prime = B_matrix - poi_vector
        B_prime_dist_mask = np.linalg.norm(B_prime,axis=1) < radius # Apply threshold to search only local vicinity
            
        # Apply double mask to B_matrix, just like covid-19. Double mask ensures safety. Double mask here ensures
        # that the B_matrix corresponds to the correct points
        B_matrix = B_matrix[B_prime_dist_mask]
        B_prime = B_prime[B_prime_dist_mask]
        B_prime_norm = B_prime / np.linalg.norm(B_prime,axis=1,keepdims=True)
        projection_coeff = np.matmul(B_prime_norm,onb.T)
        # if projection_coeff.all() == None: continue
        projection_coeff_mask = projection_coeff != 0
        projection_coeff = projection_coeff[projection_coeff_mask]
        B_matrix = B_matrix[projection_coeff_mask]

        if projection_coeff.size > 0:
            max_val = max(projection_coeff) 
            min_val = min(projection_coeff) 
        else:
            continue
        constant = 0.999
        
        while True:
            num_pts_1 = B_matrix[(projection_coeff / max_val) > constant]
            num_pts_2 = B_matrix[(projection_coeff / min_val) > constant]

            pt1 = pt2 = None  
            num_pts_1_b = num_pts_1[:,None,:]
            num_pts_2_b = num_pts_2[None,:,:]
            dist = np.linalg.norm((num_pts_1_b-num_pts_2_b),axis=-1)
            location = np.argmin(dist)
            location_2 = np.argmin(dist,-1)
            loc1 = location // dist.shape[-1]
            loc2 = location_2[loc1]
            pt1 = num_pts_1[loc1]
            pt2 = num_pts_2[loc2]
            r0, c0 = pt1
            r1, c1 = pt2
            rr, cc = draw.line(r0,c0,r1,c1)
            line_pts = list(zip(rr,cc))
            num_intersections = list(set(line_pts) & set(boundary_pts_sub))
            if constant < 0.99 or len(num_intersections) <= 2:
                break
            elif len(num_intersections) > 2:
                constant -= 0.01
            if collision:
                # Check for possibility of unbounded skeleton points 
                max_point = B_matrix[np.argmax(projection_coeff)]
                min_point = B_matrix[np.argmin(projection_coeff)]
                two_pts = [max_point,min_point]
                pts = np.asarray([max_point,min_point])
                dist = np.linalg.norm(pts-poi_vector,axis=1)
                min_dist_index = np.argmin(dist)
                pt1 = poi_vector
                pt2 = two_pts[min_dist_index]
        
        if (pt1[1]< y_lim or pt1[0]< x_lim or pt1[1] > (y_width - y_lim) or pt1[0] >(x_width - x_lim) 
            or pt2[1]< y_lim or pt2[0]< x_lim or pt2[1] > (y_width - y_lim) or pt2[0] >(x_width - x_lim)):
            continue
        else:
            pt1_arr.append(pt1)
            pt2_arr.append(pt2) 
    return pt1_arr, pt2_arr

# Following two functions are modified version of inference part of the "crack-detection" script.
# To check the original version please follow the link bellow:
# https://github.com/khanhha/crack_segmentation

def evaluate_img(model, img):
    input_width, input_height = input_size[0], input_size[1]
    img_height, img_width, _ = img.shape
    img_1 = cv2.resize(img, (input_width, input_height), cv2.INTER_AREA)
    X = train_tfms(img_1)
    X = Variable(X.unsqueeze(0)).to(DEVICE)  # [N, 1, H, W]
    mask = model(X)
    mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
    mask = cv2.resize(mask, (img_width, img_height), cv2.INTER_AREA)
    return mask

def evaluate_img_patch(model, img, stride):
    input_width, input_height = input_size[0], input_size[1]
    img_height, img_width, _ = img.shape
    if img_width < input_width or img_height < input_height:
        return evaluate_img(model, img)
    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)
    patches = []
    patch_locs = []
    for y in range(0, int(img_height - input_height + 1), int(stride)):
        for x in range(0, int(img_width - input_width + 1), int(stride)):
            segment = img[y:y + input_height, x:x + input_width]
            normalization_map[y:y + input_height, x:x + input_width] += 1
            patches.append(segment)
            patch_locs.append((x, y))
    patches = np.array(patches)

    if len(patch_locs) <= 0:
        return None
    preds = []

    for i, patch in enumerate(patches):
        patch_n = (train_tfms(patch))
        X = Variable(patch_n.unsqueeze(0)).to(DEVICE)  # [N, 1, H, W]
        masks_pred = model(X)
        mask = F.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
        preds.append(mask)

    probability_map = np.zeros((img_height, img_width), dtype=float)
    for i, response in enumerate(preds):
        coords = patch_locs[i]
        probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response
    
    return probability_map

#######################

# PARAMETERS
cut_off_threshold = 0.4 
pixel_step_size = 10 # change the density of selected points. Default = 50
gap_percantage = 1 # Radius of the circle to determine the surface orientation
resize_ratio = 0.5
max_dis = 0.5 # [m]
min_angle = 70 # [deg]
mask_toggle = 1 # 1 No mask, 2 Combine images with mask
seed = 42  # Replace with your desired seed value
torch.manual_seed(seed)

# Ensures deterministic behavior (may impact performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initilizing model
channel_means = [0.485, 0.456, 0.406]
channel_stds  = [0.229, 0.224, 0.225]
train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

model_path = SCRIPT_DIR + '/out_model/model_best_v2.pt'
model = load_unet_resnet_101(model_path = model_path)
model.eval()

# Initilizing the Metashape
doc = Metashape.app.document
chunk = doc.chunk
surface = chunk.model

camera_indexes = []
camera_labels = []
camera_path = []
selected_camera_indexes = []

# Select enabled cameras
for index, camera in enumerate(chunk.cameras):
    camera_labels.append(camera.label)
    camera_path.append(camera.photo.path)
    if camera.enabled:
        camera_indexes.append(index)

active_camera = []
active_distance = []
active_rot_x = []
active_rot_y = []

# Filtering the camares and selecting images for crack segmentation
for camera_index in camera_indexes:
    camera = chunk.cameras[camera_index]
    sensor = camera.sensor
    
    center_x = sensor.calibration.width / 2
    center_y = sensor.calibration.height / 2
    gap = (center_x + center_y)*gap_percantage//100
    focal_len = sensor.calibration.f
    teta = 2*math.atan(gap/focal_len) 

    center_point2D = Metashape.Vector([center_x, center_y])
    center_point3D = point_on_3d(center_point2D, camera)
    center_point3D_transformed = chunk.crs.project(chunk.transform.matrix.mulp(center_point3D))
    camera_center_transformed = chunk.crs.project(chunk.transform.matrix.mulp(camera.center))
    if center_point3D is not None:
        center_dist = (camera_center_transformed - center_point3D_transformed).norm()

    else:
        center_dist = 1000

    point2D = []
    point2D.append(Metashape.Vector([center_x + gap, center_y]))
    point2D.append(Metashape.Vector([center_x - gap, center_y]))
    point2D.append(Metashape.Vector([center_x, center_y + gap]))
    point2D.append(Metashape.Vector([center_x, center_y - gap]))

    point3D = []
    distance = []
    for i in range(0,4):
        point3D.append(point_on_3d(point2D[i], camera))
        if point3D[i] is None: 
            distance.append(None)
            continue
        # print(point3D[i])
        dis = (camera.center - point3D[i]).norm()
        distance.append(dis)
        # print("camera: " ,camera_index," point: ", i, " dist: ", distance)
    if distance[0] and distance[1] is not None:
        rotation_x = calculate_surface_angle(teta, distance[0], distance[1])
    else:
        rotation_x = -1

    if distance[2] and distance[3] is not None:
        rotation_y = calculate_surface_angle(teta, distance[2], distance[3])
    else:
        rotation_y = -1
    active_camera.append(camera_index)
    active_distance.append(center_dist)
    active_rot_x.append(rotation_x)
    active_rot_y.append(rotation_y)
    # print("camera: ", camera_index, "x: ", rotation_x, "y: ", rotation_y, "dist", center_dist)

    if center_dist <= max_dis and rotation_x >= min_angle and rotation_y >= min_angle:
        selected_camera_indexes.append(camera_index)

print("Selected cameras are: ", selected_camera_indexes)

for camera_index in selected_camera_indexes:
# for camera_index in range(1,2):
    camera = chunk.cameras[camera_index]
    image = camera.image()
    mask = camera.mask
   
    if mask_toggle == 1:
        image_directory = save_and_get_image(camera, image)
    elif mask_toggle == 2:
        masked_image = combine_mask_and_images(image, mask)
        image_directory = save_and_get_new_path_masked_image(masked_image, camera_labels[camera_index], camera_path[camera_index])
    
    with torch.no_grad():
        binary_crack_bool = crack_detection_alghorithm(image_directory = image_directory, model = model, cut_off_threshold = cut_off_threshold, stride_ratio = 1, resize_ratio = resize_ratio)
        binary_crack = binary_crack_bool.astype(np.uint8)
        if not binary_crack.mean():
            continue

    binary_height, binary_width = binary_crack.shape

    # Blob removal
    area_threshold = binary_width*binary_height//10000        
    binary_crack = skimage.morphology.area_opening(binary_crack, area_threshold=area_threshold, connectivity=2)

    binary_crack = add_black_border(binary_crack)
    skeleton = get_skeleton(binary_crack)
    _, distance_transform = skimage.morphology.medial_axis(binary_crack,return_distance=True)

    r,c = np.nonzero(skeleton)
    skeleton_points = list(zip(r,c))
    skeleton_points.sort(key=lambda x: x[1])

    selected_skeleton_points = [skeleton_points[i] for i in range(0,len(skeleton_points), pixel_step_size)]

    boundary = feature.canny(binary_crack > 0)
    pt1_arr, pt2_arr = hybrid_method_to_find_edge_points(selected_skeleton_points, distance_transform, binary_crack, skeleton, boundary)
    scale_bar_toggle = 1
    marker_group_name = 'Group_' + str(camera_index)
    marker_group = chunk.addMarkerGroup()
   
    marker_group.label = marker_group_name

    for i in range(len(pt1_arr)):
        adding_markers(pt1_arr[i], pt2_arr[i], marker_group, camera_index = camera_index, point_index = i, camera = camera, surface = surface, scale_bar_toggle=scale_bar_toggle)

print("Finished")

