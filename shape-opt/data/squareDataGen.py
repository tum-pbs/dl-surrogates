import numpy as np
import os
from utils import random_signal

NUM_OF_SAMPLES    = 1000
SUPER_SAMPLE      = True
SUPER_SAMPLE_RATE = 100

ORIGIN_X          = 0.5
ORIGIN_Y          = 0.0

NUM_RECTANGLE     = 32

MAX_HEIGHT        = 0.80
MAX_WIDTH         = 0.80
MIN_WIDTH         = 0.10

OUT_DIR           = "./database/square_airfoil_database/"

class InvalidOperationOnPointType(Exception):
    pass

class Point:
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y
    
    def __repr__(self):
        return str(self.x) + ", " + str(self.y)
    
    def __ne__(self, rhs):
        return ( abs(self.x - rhs.x) > 1e-3 or abs(self.y - rhs.y) > 1e-3)

    def __sub__(self, rhs):
        return Point(self.x - rhs.x, self.y - rhs.y)

    def __add__(self, rhs):
        return Point(self.x + rhs.x, self.y + rhs.y)

    def __truediv__(self, rhs):
        if type(rhs) is float or type(rhs) is int:
            return Point(self.x / rhs, self.y / rhs)
        else:
            raise InvalidOperationOnPointType

    def __div__(self, rhs):
        if type(rhs) is float or type(rhs) is int:
            return Point(self.x / rhs, self.y / rhs)
        else:
            raise InvalidOperationOnPointType

    def __mul__(self, rhs):
        if type(rhs) is float or type(rhs) is int:
            return Point(self.x * rhs, self.y * rhs)
        else:
            raise InvalidOperationOnPointType

class Sample:
    def __init__(self, start_point, end_point, num_of_sample):
        self.start   = start_point
        self.end     = end_point
        self.nos     = num_of_sample
        self.samples = []
    
    def _clc_samples(self):
        for k in range(1, self.nos + 1):
            self.samples.append(self.start + (self.end - self.start)/(self.nos + 1)*(k))
    
    def get_samples(self):
        self._clc_samples()
        return self.samples

class Rectangle:
    def __init__(self, center, width, height):
        self.vertices = []
        self.center   = center
        self.width    = width
        self.height   = height
        self._fillVertices()

    def _fillVertices(self):
        self.vertices.append(Point(self.center.x - self.width/2, self.center.y - self.height/2))
        self.vertices.append(Point(self.center.x - self.width/2, self.center.y + self.height/2))
        self.vertices.append(Point(self.center.x + self.width/2, self.center.y + self.height/2))
        self.vertices.append(Point(self.center.x + self.width/2, self.center.y - self.height/2))
    
    def getVertex(self, index):
        return self.vertices[index]

    def __repr__(self):
        string =  "Center: " + str(self.center) + "\n"
        string += "Width: " + str(self.width) + " Height: " + str(self.height) + "\n"
        string += "Vertices: " + str(self.vertices[0]) + "; " + str(self.vertices[1]) + "; " + str(self.vertices[2]) + "; "+ str(self.vertices[3]) + "\n"
        return string

class ClosedLoop:
    def __init__(self, num_of_rectangles, height, width, **kwargs):
        assert type(width) == type([])

        self.height = height if type(height) == type([]) else [height for _ in range(num_of_rectangles)]
        self.width  = width
        self.num_of_rectangles = num_of_rectangles

        assert len(self.width)   == self.num_of_rectangles
        assert len(self.height)  == self.num_of_rectangles

        self.origin_x = kwargs['ORIGIN_X'] if 'ORIGIN_X' in kwargs else 0.5
        self.origin_y = kwargs['ORIGIN_Y'] if 'ORIGIN_Y' in kwargs else 0.0
        self.super_sampling_rate = kwargs['SUPER_SAMPLE_RATE'] if 'SUPER_SAMPLE_RATE' in kwargs else 10

        self.points = []
        self.rectangle_array = []

    def _get_rectangles(self):
        for k in range(self.num_of_rectangles):
            center = Point( self.origin_x, (self.origin_y - ( self.num_of_rectangles - 1.0 ) / 2.0 * self.height[k]) + k * self.height[k] )
            self.rectangle_array.append(Rectangle(center, self.width[k], self.height[k]))

    def _get_rectangle_corners(self):
        for i in range(0, self.num_of_rectangles):
            self.points.append(Point(self.rectangle_array[i].getVertex(0).x, self.rectangle_array[i].getVertex(0).y))
            self.points.append(Point(self.rectangle_array[i].getVertex(1).x, self.rectangle_array[i].getVertex(1).y))
    
        for i in range(self.num_of_rectangles, 0, -1):
            self.points.append(Point(self.rectangle_array[i-1].getVertex(2).x, self.rectangle_array[i-1].getVertex(2).y))
            self.points.append(Point(self.rectangle_array[i-1].getVertex(3).x, self.rectangle_array[i-1].getVertex(3).y))

        self.points.append(self.points[0])
    
    def _clean_point_list(self):
        tempPoints = []
        tempPoints.append( self.points[0] )

        for i in range(1, len(self.points)):
            if tempPoints[-1] != self.points[i]:
                tempPoints.append( self.points[i] )

        self.points = tempPoints

    def _super_sample(self):
        super_sampled_points = []
        for k in range(len(self.points) - 1):
            super_sampled_points +=     Sample(self.points[k],
                                               self.points[k+1], 
                                               self.super_sampling_rate).get_samples()
        return super_sampled_points

    def _clc_points(self):
        self._get_rectangles()
        self._get_rectangle_corners()
        self._clean_point_list()

    def get_points(self):
        self._clc_points()
        return self.points

    def get_super_sampled_points(self):
        self._clc_points()
        return self._super_sample()

class PointsToTupple:
    def __init__(self, point_list):
        self.plist = point_list
        self.tupple_list = []

    def _convert_point_list(self):
        for point in self.plist:
            self.tupple_list.append((point.x, point.y))
    
    def get_tupple_list(self):
        self._convert_point_list()
        return self.tupple_list

def writePointsIntoFile(dataName, array, fileName):
    with open(fileName, "wt") as outFile:
        outFile.write(dataName + "\n")
        for i in range(array.shape[0]):
            outFile.write( 4*" " + np.array2string( array[i], precision=10, separator=' ' )[1:-1].lstrip(' ') + "\n" )

def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    height    = MAX_HEIGHT / NUM_RECTANGLE
    
    for filesuffix in range(NUM_OF_SAMPLES):
        rand_width = (MAX_WIDTH - MIN_WIDTH) * random_signal(sampling=NUM_RECTANGLE) + MIN_WIDTH

        loop = ClosedLoop(NUM_RECTANGLE, height, rand_width.tolist(), SUPER_SAMPLE_RATE=SUPER_SAMPLE_RATE)
        if SUPER_SAMPLE:
            points = loop.get_super_sampled_points() 
        else:
            points = loop.get_points()
        
        SQPoints = PointsToTupple(points).get_tupple_list()

        dataNeme = "SQ" + str(filesuffix)
        outFile  = os.path.join(OUT_DIR, dataNeme + ".dat") 
        writePointsIntoFile(dataNeme, np.array(SQPoints), outFile)
        filesuffix += 1


if __name__ == '__main__':
    main()
