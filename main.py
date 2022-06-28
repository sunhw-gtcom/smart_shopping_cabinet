from fastapi import FastAPI, UploadFile, File
import matplotlib.pyplot as plt
import matplotlib.image as img
import fastapi.responses as response
import shutil
import torch
import cv2
import delete_files

app = FastAPI()

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


@app.post("/img/")
async def root(file: UploadFile = File(...)):
    with open(file.filename, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    buffer.close()

    img = 'C:/Users/BHASKAR BOSE/{}'.format(file.filename)
    # Inference
    results = model(img)
    results.save()

    im = cv2.imread('C:/Users/BHASKAR BOSE/runs/detect/exp/{}'.format(file.filename))
    im = cv2.putText(im, "persons : {}".format(
        len(list(a for a in results.pandas().xyxy[0]['name'] == 'person' if a != False))), (150, 200),
                     cv2.FONT_HERSHEY_SIMPLEX, 9, (255, 0, 0), 7)
    cv2.imwrite('C:/Users/BHASKAR BOSE/{}'.format("final.jpg"), im)
    delete_files.delete_image_files()

    return response.FileResponse('C:/Users/BHASKAR BOSE/{}'.format("final.jpg"))


@app.post("/video/")
async def vid(file: UploadFile = File(...)):
    with open(file.filename, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    buffer.close()

    vidcap = cv2.VideoCapture(file.filename)
    success, image = vidcap.read()
    count = 0

    while success:
        cv2.imwrite("frame_split/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1

    for i in range(count):
        img = "C:/Users/BHASKAR BOSE/frame_split/frame{}.jpg".format(i)
        results = model(img)
        results.save('C:/Users/BHASKAR BOSE/runs/detect/exp')
        im = cv2.imread('C:/Users/BHASKAR BOSE/runs/detect/exp/frame{}.jpg'.format(i))
        im = cv2.putText(im, "persons : {}".format(
            len(list(a for a in results.pandas().xyxy[0]['name'] == 'person' if a != False))), (100, 100),
                         cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 4)
        cv2.imwrite('C:/Users/BHASKAR BOSE/runs/detect/frame{}.jpg'.format(i), im)
    img_array = []

    for i in range(count):
        filename = 'C:/Users/BHASKAR BOSE/runs/detect/frame{}.jpg'.format(i)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    delete_files.delete_video_files()

    return response.FileResponse('C:/Users/BHASKAR BOSE/project.avi')