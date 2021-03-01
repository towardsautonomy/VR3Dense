import math
import numpy as np

def affineTransform(coordinate, Rx, Ry, Rz, Tx, Ty, Tz):
    '''
    This function performs affine transformation on a world coordinate.
    World coordinate systems are as following:

    Parameters:
        coordinate : World coordinate as list [x, y, z]'
        Rx              : Rotation about x-axis in degree
        Ry              : Rotation about y-axis in degree
        Rz              : Rotation about z-axis in degree
        Tx              : Translation in the x-axis in meters
        Ty              : Translation in the y-axis in meters
        Tz              : Translation in the z-axis in meters

    x -> ahead of the car/object
    y -> pointing left of the car/object
    z -> pointing up
    '''
    orig_shape = np.asarray(coordinate).shape
    # convert coordinate array to a numpy array
    coordinate = np.asarray(coordinate).reshape(3, 1)
    
    # convert degree to radians
    Rx = math.radians(Rx)
    Ry = math.radians(Ry)
    Rz = math.radians(Rz)

    ## rotation matrix
    # rotation about x-axis
    Rx_mat = [[1,   0           ,     0           ],
              [0,   math.cos(Rx),    -math.sin(Rx)],
              [0,   math.sin(Rx),     math.cos(Rx)]]
    # rotation about y-axis
    Ry_mat = [[ math.cos(Ry),    0,    math.sin(Ry)],
              [0            ,    1,    0           ],
              [-math.sin(Ry),    0,    math.cos(Ry)]]
    # rotation about z-axis
    Rz_mat = [[math.cos(Rz),    -math.sin(Rz),  0],
              [math.sin(Rz),     math.cos(Rz),  0],
              [0           ,     0           ,  1]]
    # combined rotation matrix
    R = np.matmul(Rx_mat, np.matmul(Ry_mat, Rz_mat))

    # apply rotation
    coordinate = np.matmul(R, coordinate)

    # apply translation
    coordinate = np.add(coordinate, np.array([[Tx],[Ty],[Tz]]))
    coordinate = np.asarray(coordinate).reshape(orig_shape)

    return coordinate