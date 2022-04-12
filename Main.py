import cv2
import numpy as np
import os
import csv
import sys
import time

import DetectChars
import DetectPlates
import PossiblePlate

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

    filename = "Number_Plates_records.csv"

    if blnKNNTrainingSuccessful == False:
        print("\nerror: KNN training was not successful\n")
        return
    
    if(len(sys.argv) == 0):
        imgOriginalScene = cv2.imread("source.jpg")    
    else:
        imgOriginalScene = cv2.imread(sys.argv[1])

    if imgOriginalScene is None:
        print("\nerror: image not read from file \n\n")
        os.system("pause")
        return

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

    cv2.imshow("imgOriginalScene", imgOriginalScene)

    if len(listOfPossiblePlates) == 0:
        print("\nno license plates were detected\n")
    else:



        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)


        licPlate = listOfPossiblePlates[0]

        cv2.imshow("imgPlate", licPlate.imgPlate)
        cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:
            print("\nno characters were detected\n\n")
            return


        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)

        print("\nlicense plate read from image = " + licPlate.strChars + "\n")
        print("----------------------------------------")
        
        if(len(sys.argv) == 0):
            record_date = time.ctime(os.path.getctime("source.jpg"))   
        else:
            record_date = time.ctime(os.path.getctime(sys.argv[1]))
            

        row = [licPlate.strChars, record_date]

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(row)


        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)

        cv2.imshow("imgOriginalScene", imgOriginalScene)

        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)



    cv2.waitKey(0)

    return



def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)

    p0 = (int(p2fRectPoints[0][0]),int(p2fRectPoints[0][1]))
    p1 = (int(p2fRectPoints[1][0]),int(p2fRectPoints[1][1]))
    p2 = (int(p2fRectPoints[2][0]),int(p2fRectPoints[2][1]))
    p3 = (int(p2fRectPoints[3][0]),int(p2fRectPoints[3][1]))

    cv2.line(imgOriginalScene,p0,p1,SCALAR_RED,2)
    cv2.line(imgOriginalScene,p1,p2,SCALAR_RED,2)
    cv2.line(imgOriginalScene,p2,p3,SCALAR_RED,2)
    cv2.line(imgOriginalScene,p3,p0,SCALAR_RED,2)
    

def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = float(plateHeight) / 30.0
    intFontThickness = int(round(fltFontScale * 1.5))

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)


    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)

    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
    else:
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))
    # end if

    textSizeWidth, textSizeHeight = textSize

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))


    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)



if __name__ == "__main__":
    main()
