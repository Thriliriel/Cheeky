import cv2
import os
import numpy as np
import argparse

#get params
parser = argparse.ArgumentParser(description='Cheeky!')

parser.add_argument('--path', type=str, default='aaa', help='Path to the image. If a directory is sent, it checks all jpgs and pngs from there')
parser.add_argument('--savePath', type=str, default='', help='Path where to save the cheeky image. If empty, it just dont save it')
parser.add_argument('--fileName', type=str, default='results.txt', help='Name of the file to save the results. If empty, default is results.txt')

args = parser.parse_args()

print(args.path)

# load haar classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# process each cheek
def process_cheek(cheek_img, label):
    # mean RGB and standard dev
    mean_rgb = cv2.mean(cheek_img)
    std_rgb = cv2.meanStdDev(cheek_img)
        
    print(f"--- {label} ---")
    print(f"Mean RGB: (R: {mean_rgb[2]:.2f}, G: {mean_rgb[1]:.2f}, B: {mean_rgb[0]:.2f})")
    print(f"Standard Deviation: (R: {std_rgb[1][2][0]:.2f}, G: {std_rgb[1][1][0]:.2f}, B: {std_rgb[1][0][0]:.2f})")

    return cheek_img, mean_rgb, std_rgb

# find multiple images in dir
def find_multiple_images(imagePath, savePath, face_cascade, results):    
    #keep the last found subject, so we can use as default if no face is found
    lastFace = None

    #qnt images found with face
    qntImages = 0

    #totals mean and svg, to calculate the whole later
    totalMeanRGB = [0,0,0]
    totalStdRGB = [0,0,0]

    #for each image in the directory
    for imgs in os.listdir(imagePath):
        #if it is jpg or png, we try... otherwise, we skip it
        if imgs.endswith(".png") or imgs.endswith(".jpg"):
            imgPath = imagePath+imgs
            try:
                # read the image
                img = cv2.imread(imgPath)
                if img is None:
                    print("Error in reading the image. Is the path correct?")
                    results.write(imgs+"\n")
                    results.write("Error in reading the image. Is the path correct?\n")
                    return

                # Convert to grayscale (for haar)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # face detection
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                #if did not find any face in the image
                if len(faces) == 0:
                    #if we have last face, use it
                    if lastFace != None:
                        #use the last one found
                        faces = lastFace
                    #otherwise, shame...
                    else:
                        print("No face detected!")
                        results.write(imgs+"\n")
                        results.write("No face detected!\n")
                        return                    

                #if we have more than 1 face, we use only the biggest one
                if len(faces) > 1:
                    lower = 0
                    ind = 0
                    keep = 0
                    for (x, y, w, h) in faces:
                        if w > lower:
                            lower = w
                            keep = ind

                        ind += 1                
                    faces = [faces[keep]]

                #if found a face, put it in the last face, so we can use it for reference if it cant find the next
                if len(faces) > 0:
                    lastFace = [faces[0]]
            
                for (x, y, w, h) in faces:
                    # Define cheeks region
                    # left: nearby face's left side
                    # right: nearby face's right side
                    square_size = int(w * 0.15)  # size of 15% from face width
                    offset_y = int(h * 0.5)     # more or less the inferior part of the face

                    # Coords of left cheek square
                    left_cheek_x = x + int(w * 0.18)
                    left_cheek_y = y + offset_y

                    # Coords of right cheek square
                    right_cheek_x = x + w - int(w * 0.18) - square_size
                    right_cheek_y = y + offset_y

                    #make sure it is inside of the image
                    left_square = img[left_cheek_y:left_cheek_y+square_size, left_cheek_x:left_cheek_x+square_size]
                    right_square = img[right_cheek_y:right_cheek_y+square_size, right_cheek_x:right_cheek_x+square_size]

                    # Process cheeks
                    left_result, meanLeft, stdLeft = process_cheek(left_square, "Left Cheek")
                    right_result, meanRight, stdRight = process_cheek(right_square, "Right Cheek")

                    #mean and average between left and right
                    meanCheek = [(meanLeft[0]+meanRight[0])/2.0, (meanLeft[1]+meanRight[1])/2.0,(meanLeft[2]+meanRight[2])/2.0]
                    stdCheek = [(stdLeft[1][0][0]+stdRight[1][0][0])/2.0, (stdLeft[1][1][0]+stdRight[1][1][0])/2.0,(stdLeft[1][2][0]+stdRight[1][2][0])/2.0]

                    #update qntImages
                    qntImages += 1

                    #update the total, to calculate for all later
                    totalMeanRGB = [totalMeanRGB[0]+meanCheek[0],totalMeanRGB[1]+meanCheek[1],totalMeanRGB[2]+meanCheek[2]]
                    totalStdRGB = [totalStdRGB[0]+stdCheek[0],totalStdRGB[1]+stdCheek[1],totalStdRGB[2]+stdCheek[2]]
                    
                    #write on file
                    results.write(imgs+"\n")
                    results.write("Left Cheek: MEAN RGB: R: "+str(meanLeft[2])+", G: "+str(meanLeft[1])+", B: "+str(meanLeft[0])+"\n")
                    results.write("Left Cheek: STDDEV: R: "+str(stdLeft[1][2][0])+", G: "+str(stdLeft[1][1][0])+", B: "+str(stdLeft[1][0][0])+"\n")
                    results.write("Right Cheek: MEAN RGB: R: "+str(meanRight[2])+", G: "+str(meanRight[1])+", B: "+str(meanRight[0])+"\n")
                    results.write("Right Cheek: STDDEV: R: "+str(stdRight[1][2][0])+", G: "+str(stdRight[1][1][0])+", B: "+str(stdRight[1][0][0])+"\n")
                    results.write("Total: MEAN RGB: R: "+str(meanCheek[2])+", G: "+str(meanCheek[1])+", B: "+str(meanCheek[0])+"\n")
                    results.write("Total: STDDEV: R: "+str(stdCheek[2])+", G: "+str(stdCheek[1])+", B: "+str(stdCheek[0])+"\n\n")

                    # draw cheek squares for visualization
                    cv2.rectangle(img, (left_cheek_x, left_cheek_y), 
                                  (left_cheek_x + square_size, left_cheek_y + square_size), (255, 0, 255), 2)
                    cv2.rectangle(img, (right_cheek_x, right_cheek_y), 
                                  (right_cheek_x + square_size, right_cheek_y + square_size), (255, 0, 255), 2)

                    # if a path was provided to save the cheek image, save it
                    if savePath != "":
                        cv2.imwrite(savePath+imgs, img)
            
            except Exception as error:
                print("A problem occurred: "+ str(error)+ "\n")
                results.write(imgs+"\n")
                results.write("A problem occurred: "+ str(error)+ "\n")
        else:
            print("Image in the wrong format!")
            results.write(imgs+"\n")
            results.write("Image in the wrong format!\n")

    #calculate the mean and avg for all faces found
    totalMeanRGB = [x / float(qntImages) for x in totalMeanRGB]
    totalStdRGB = [x / float(qntImages) for x in totalStdRGB]

    #write on file
    results.write("\nMean RGB for all: R: "+str(totalMeanRGB[2])+", G: "+str(totalMeanRGB[1])+", B: "+str(totalMeanRGB[0])+"\n")
    results.write("STDDEV  RGB for all: R: "+str(totalStdRGB[2])+", G: "+str(totalStdRGB[1])+", B: "+str(totalStdRGB[0]))


# find for one image only
def find_one_image(imagePath, savePath, face_cascade, results):
    try:
        # read the image
        img = cv2.imread(imagePath)
        if img is None:
            print("Error in reading the image. Is the path correct?")
            results.write(imagePath+"\n")
            results.write("Error in reading the image. Is the path correct?\n")
            return

        # Convert to grayscale (for haar)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # no face detected
        if len(faces) == 0:
            print("No face detected!")
            results.write(imagePath+"\n")
            results.write("No face detected!\n")
            return

        #if we have more than 1 face, we use only the biggest one
        if len(faces) > 1:
            lower = 0
            ind = 0
            keep = 0
            for (x, y, w, h) in faces:
                if w > lower:
                    lower = w
                    keep = ind

                ind += 1
            faces = [faces[keep]]

        # for each face (1, presumably...)
        for (x, y, w, h) in faces:
            # Define cheeks region
            # left: nearby face's left side
            # right: nearby face's right side
            square_size = int(w * 0.15)  # size of 15% from face width
            offset_y = int(h * 0.5)     # more or less the inferior part of the face

            # Coords of left cheek square
            left_cheek_x = x + int(w * 0.18)
            left_cheek_y = y + offset_y

            # Coords of right cheek square
            right_cheek_x = x + w - int(w * 0.18) - square_size
            right_cheek_y = y + offset_y

            #make sure it is inside of the image
            left_square = img[left_cheek_y:left_cheek_y+square_size, left_cheek_x:left_cheek_x+square_size]
            right_square = img[right_cheek_y:right_cheek_y+square_size, right_cheek_x:right_cheek_x+square_size]

            # Process cheeks
            left_result, meanLeft, stdLeft = process_cheek(left_square, "Left Cheek")
            right_result, meanRight, stdRight = process_cheek(right_square, "Right Cheek")

            #mean and average between left and right
            meanCheek = [(meanLeft[0]+meanRight[0])/2, (meanLeft[1]+meanRight[1])/2,(meanLeft[2]+meanRight[2])/2]
            stdCheek = [(stdLeft[1][0][0]+stdRight[1][0][0])/2, (stdLeft[1][1][0]+stdRight[1][1][0])/2,(stdLeft[1][2][0]+stdRight[1][2][0])/2]

            #write on file
            results.write(imagePath+"\n")
            results.write("Left Cheek: MEAN RGB: R: "+str(meanLeft[2])+", G: "+str(meanLeft[1])+", B: "+str(meanLeft[0])+"\n")
            results.write("Left Cheek: STDDEV: R: "+str(stdLeft[1][2][0])+", G: "+str(stdLeft[1][1][0])+", B: "+str(stdLeft[1][0][0])+"\n")
            results.write("Right Cheek: MEAN RGB: R: "+str(meanRight[2])+", G: "+str(meanRight[1])+", B: "+str(meanRight[0])+"\n")
            results.write("Right Cheek: STDDEV: R: "+str(stdRight[1][2][0])+", G: "+str(stdRight[1][1][0])+", B: "+str(stdRight[1][0][0])+"\n")
            results.write("Total: MEAN RGB: R: "+str(meanCheek[2])+", G: "+str(meanCheek[1])+", B: "+str(meanCheek[0])+"\n")
            results.write("Total: STDDEV: R: "+str(stdCheek[2])+", G: "+str(stdCheek[1])+", B: "+str(stdCheek[0])+"\n\n")
                
            # draw cheek squares for visualization
            cv2.rectangle(img, (left_cheek_x, left_cheek_y), 
                            (left_cheek_x + square_size, left_cheek_y + square_size), (255, 0, 255), 2)
            cv2.rectangle(img, (right_cheek_x, right_cheek_y), 
                            (right_cheek_x + square_size, right_cheek_y + square_size), (255, 0, 255), 2)

            # if a path was provided to save the cheek image, save it
            if savePath != "":
                cv2.imwrite(savePath+imagePath, img)

            # show image with cheek squares
            cv2.imshow('Cheeky!', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as error:
        print("A problem occurred: "+ str(error)+ "\n")
        results.write(imagePath+"\n")
        results.write("A problem occurred: "+ str(error)+ "\n")

# open file to write
results = open(args.fileName, 'w')

#checks if the path is an image or a dir
if os.path.isdir(args.path):
    #call the multiple images code
    find_multiple_images(args.path, args.savePath, face_cascade, results)
else:
    #call the single image code
    find_one_image(args.path, args.savePath, face_cascade, results)

results.close()