#!/usr/bin/python
#
# Various helper methods and includes for Cityscapes
#

# Python imports
from __future__ import print_function, absolute_import, division
import os, sys, getopt
import glob
import math
import json
from collections import namedtuple
import logging 
import traceback

# Image processing
# Check if PIL is actually Pillow as expected
#try:
#    from PIL import PILLOW_VERSION
#except:
#    print("Please install the module 'Pillow' for image processing, e.g.")
#    print("pip install pillow")
#    sys.exit(-1)

try:
    import PIL.Image     as Image
    import PIL.ImageDraw as ImageDraw
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)

# Numpy for datastructures
try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

# Cityscapes modules
try:
    #from kitti360scripts.helpers.annotation import Annotation
    from kitti360scripts.helpers.labels import labels, name2label, id2label, trainId2label, category2labels
except ImportError as err:
    print("Failed to import all Cityscapes modules: %s" % err)
    sys.exit(-1)
except Exception as e:
    logging.error(traceback.format_exc())
    sys.exit(-1)
except:
    print("Unexpected error in loading Cityscapes modules")
    print(sys.exc_info()[0])
    sys.exit(-1)

# Print an error message and quit
def printError(message):
    print('ERROR: ' + str(message))
    sys.exit(-1)

# Class for colors
class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

# Colored value output if colorized flag is activated.
def getColorEntry(val, args):
    if not args.colorized:
        return ""
    if not isinstance(val, float) or math.isnan(val):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN

# Cityscapes files have a typical filename structure
# <city>_<sequenceNb>_<frameNb>_<type>[_<type2>].<ext>
# This class contains the individual elements as members
# For the sequence and frame number, the strings are returned, including leading zeros
CsFile = namedtuple( 'csFile' , [ 'sequenceNb' , 'frameNb' , 'ext' ] )
CsWindow = namedtuple( 'csFile' , [ 'sequenceNb' , 'firstFrameNb' , 'lastFrameNb' ] )

# Returns a CsFile object filled from the info in the given filename
def getFileInfo(fileName):
    baseName = os.path.basename(fileName)
    sequenceName = [x for x in fileName.split('/') if x.startswith('2013_05_28')]
    if not len(sequenceName)==1:
        printError( 'Cannot parse given filename ({}). Does not seem to be a valid KITTI-360 file.'.format(fileName) )
    sequenceNb = int(sequenceName[0].split('_')[4])
    frameNb = int(os.path.splitext(baseName)[0])
    csFile = CsFile(sequenceNb, frameNb, os.path.splitext(baseName)[1])
    return csFile

# Returns a CsFile object filled from the info in the given filename
def getWindowInfo(fileName):
    baseName = os.path.basename(fileName)
    sequenceName = [x for x in fileName.split('/') if x.startswith('2013_05_28')]
    if not len(sequenceName)==1:
        printError( 'Cannot parse given filename ({}). Does not seem to be a valid KITTI-360 file.'.format(fileName) )
    sequenceNb = int(sequenceName[0].split('_')[4])
    windowName = os.path.splitext(baseName)[0]
    firstFrameNb = int(windowName.split('_')[0])
    lastFrameNb = int(windowName.split('_')[1])
    csWindow = CsWindow(sequenceNb, firstFrameNb, lastFrameNb)
    return csWindow

# Returns a CsFile object filled from the info in the given filename
def getCsFileInfo(fileName):
    baseName = os.path.basename(fileName)
    parts = baseName.split('_')
    parts = parts[:-1] + parts[-1].split('.')
    if not parts:
        printError( 'Cannot parse given filename ({}). Does not seem to be a valid Cityscapes file.'.format(fileName) )
    if len(parts) == 5:
        csFile = CsFile( *parts[:-1] , type2="" , ext=parts[-1] )
    elif len(parts) == 6:
        csFile = CsFile( *parts )
    else:
        printError( 'Found {} part(s) in given filename ({}). Expected 5 or 6.'.format(len(parts) , fileName) )

    return csFile

# Returns the part of Cityscapes filenames that is common to all data types
# e.g. for city_123456_123456_gtFine_polygons.json returns city_123456_123456
def getCoreImageFileName(filename):
    csFile = getCsFileInfo(filename)
    return "{}_{}_{}".format( csFile.city , csFile.sequenceNb , csFile.frameNb )

# Returns the directory name for the given filename, e.g.
# fileName = "/foo/bar/foobar.txt"
# return value is "bar"
# Not much error checking though
def getDirectory(fileName):
    dirName = os.path.dirname(fileName)
    return os.path.basename(dirName)

# Make sure that the given path exists
def ensurePath(path):
    if not path:
        return
    if not os.path.isdir(path):
        os.makedirs(path)

# Write a dictionary as json file
def writeDict2JSON(dictName, fileName):
    with open(fileName, 'w') as f:
        f.write(json.dumps(dictName, default=lambda o: o.__dict__, sort_keys=True, indent=4))

# Convert rotation matrix to axis angle
def Rodrigues(matrix):
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """

    # Axes.
    axis = np.zeros(3, np.float64)
    axis[0] = matrix[2,1] - matrix[1,2]
    axis[1] = matrix[0,2] - matrix[2,0]
    axis[2] = matrix[1,0] - matrix[0,1]

    # Angle.
    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
    t = matrix[0,0] + matrix[1,1] + matrix[2,2]
    theta = np.arctan2(r, t-1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return axis * theta

# dummy main
if __name__ == "__main__":
    printError("Only for include, not executable on its own.")
