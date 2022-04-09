import numpy as np
import cv2


def cannyImageProcess(grayImage):
    edgeImage = cv2.Canny(grayImage, 80, 150, apertureSize=3)
    kernel = np.ones((3,3), np.uint8)
    thickLineImage = cv2.dilate(edgeImage, kernel, iterations = 1)
    return lineDirections(thickLineImage)


def lineDirections(image):
    inputs = [0, 0, 1, 0, 0]
    carPosRow = 62
    carPosCol = 48
    curPos = [carPosRow, carPosCol]

    # check left
    for i in range(0, 40):
        curPos[1] -= 1
        if image[carPosRow][curPos[1]] == 255:
            inputs[0] = 1 - (abs(carPosCol - curPos[1]) / 40)
            break

    curPos = [carPosRow, carPosCol]
    
    # check up-left
    for i in range(0, 40):
        curPos[0] -= 1
        curPos[1] -= 1
        if image[curPos[0]][curPos[1]] == 255:
            inputs[1] = 1 - (abs(carPosCol - curPos[1]) / 40)
            break

    curPos = [carPosRow, carPosCol]

    # check up
    for i in range(0, 40):
        curPos[0] -= 1
        if image[curPos[0]][carPosCol] == 255:
            inputs[2] = (abs(curPos[0] - carPosRow) / 40)
            break

    curPos = [carPosRow, carPosCol]

    # check up-right
    for i in range(0, 40):
        curPos[0] -= 1
        curPos[1] += 1
        if image[curPos[0]][curPos[1]] == 255:
            inputs[3] = 1 - (abs(curPos[1] - carPosCol) / 40)
            break
        
    curPos = [carPosRow, carPosCol]
    
    # check right
    for i in range(0, 40):
        curPos[1] += 1
        if image[carPosRow][curPos[1]] == 255:
            inputs[4] = 1 - (abs(curPos[1] - carPosCol) / 40)
            break

    return inputs
