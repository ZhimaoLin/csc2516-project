from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import os
from PIL import Image


def convert_to_jpg(path):
    if path.endswith(".png"):
        image = Image.open("./" + path)
        image.save('./target' + path[: -4] + ".jpg")
    else:
        for f in os.listdir("./" + path):
            if not f.endswith('.DS_Store'):
                if not f.endswith(".png"):
                    os.mkdir("./target" + path + "/" + f)
                convert_to_jpg(path + "/" + f)


def crop_image(path):
    if path.endswith(".jpg"):
        image = extract_face("./" + path)
        image.save('./crop' + path)
    else:
        for f in os.listdir("./" + path):
            if not f.endswith('.DS_Store'):
                if not f.endswith(".jpg"):
                    os.mkdir("./crop" + path + "/" + f)
                crop_image(path + "/" + f)


""" call this function with string parameter "filename" (name of the root folder containing all the images) and it will 
    create a "targetfilename" folder that will contain the jpg images and a "croptargetfilename"
    folder that will contain the cropped jpg images. It will do so by calling the other two functions.
    note that both these new folders will preserve the structure of the original "filename" folder.
    Note** this function assumes it is being called from the same directory as the parameter "filename" folder."""


def combine(filename):
    os.mkdir('./target' + filename)
    os.mkdir('./croptarget' + filename)

    convert_to_jpg(filename)
    crop_image('target' + filename)


def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    return image


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # TODO: replace "filename" with the string name of the images folder and make sure to save this python file in the
    #  same directory as the "filename" folder
    combine("filename")
