import cv2
import argparse
import os
import numpy as np
import math
import torch
import torchvision
from torchvision import datasets, transforms
from torchsummary import summary
import pytesseract
from pytesseract import Output
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

parser = argparse.ArgumentParser(description='Process image')
parser.add_argument('--input', metavar='FILE', type=str, nargs=1, required=True, help='input image')
parser.add_argument('--model_type', type=str, nargs=1, required=True, help='model type for text extraction')

args = parser.parse_args()

# extract the axis labels as text from the chart
def extract_text():
    img = cv2.imread(args.input[0])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img_gray, 0, 100)
    # print(edges[0:50, 0:50])
    _, thresholded = cv2.threshold(img_gray, 127, 255, cv2.THRESH_OTSU)
    enlarged = cv2.resize(thresholded, None, fx=3.0, fy=3.0)
    _, enlarged = cv2.threshold(enlarged, 127, 255, cv2.THRESH_OTSU)

    # text = pytesseract.image_to_data(img_gray[0:50, 0:40], output_type=Output.DICT)
    # text = pytesseract.image_to_string(thresholded[50:100, 0:40], config='tessedit_char_whitelist=0123456789')
    text = pytesseract.image_to_string(enlarged[650:750, 0:100])
    print(text)
    print(thresholded[200:250, 0:40])
    # print(enlarged[650:750, 0:100])

    # cv2.imshow('edges', edges)
    # cv2.imshow('thresholded', thresholded[50:100, 0:40])
    cv2.imshow('enlarged', enlarged[650:750, 0:100])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# img: 50x50 image containing text to recognize
# if cnn model, reshape image to (1, 1, 50, 50)
# if simple model, reshape image to (1, 2500)
def extract_text_cnn(img):
    train_data = datasets.MNIST(root='data', 
                                download=True,
                                train=True, 
                                transform=transforms.ToTensor())
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    if args.model_type[0] == 'simple_model':
        model = torch.nn.Sequential(
            torch.nn.Linear(2500, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            torch.nn.LogSoftmax(dim=1)
        )

    if args.model_type[0] == 'cnn':
        model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding=1),
            torch.nn.MaxPool2d(kernel_size=(2,2)),
            torch.nn.Dropout(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1),
            torch.nn.MaxPool2d(kernel_size=(2,2)),
            torch.nn.Dropout(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
            # torch.nn.MaxPool2d(kernel_size=(2,2)),
            # torch.nn.Dropout(),
            # torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(9216, 64),
            torch.nn.Dropout(),
            torch.nn.Linear(64, 10),
            torch.nn.LogSoftmax(dim=1)
        )
        print(summary(model, (1, 50, 50), batch_dim=0))

    if os.path.exists(args.model_type[0]):
        model.load_state_dict(torch.load(args.model_type[0]))
        return model(img)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.NLLLoss()

    epochs = 50
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            # print(images.shape)
            # want to process 50x50 size image - add zeros to the end
            zeros = torch.zeros((images.shape[0], 2500-784))
            # print(images.shape, zeros.shape)
            images = torch.cat((images, zeros), 1)
            images = images.view(images.shape[0], 1, 50, 50)
            # print(images.shape)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            # print("Loss: ", loss.item())
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
        else:
          print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))


    torch.save(model.state_dict(), args.model_type[0])
    return model(img)

'''
img = cv2.imread(args.input[0])
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresholded = cv2.threshold(img_gray, 127, 255, cv2.THRESH_OTSU)
enlarged = cv2.resize(thresholded, None, fx=3.0, fy=3.0)
_, enlarged = cv2.threshold(enlarged, 127, 255, cv2.THRESH_OTSU)

# cv2.imshow('enlarged', enlarged[870:920, 270:320])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print(enlarged[870:920, 570:620])

enlarged = enlarged[870:920, 570:620]

invert = 255*np.ones(enlarged.shape)
enlarged = invert - enlarged
# print(enlarged)

# thresholded = thresholded[50:100, 0:50].reshape((1, 1, 50, 50))
if args.model_type[0] == 'cnn':
    enlarged = enlarged.reshape((1, 1, 50, 50))

if args.model_type[0] == 'simple_model':   
    enlarged = enlarged.reshape((1, 2500))

log_prob = extract_text_cnn(torch.from_numpy((enlarged.astype(np.float32))))
prob = list(torch.exp(log_prob).detach().numpy()[0])
prediction = prob.index(max(prob))
print("Prediction: ", prediction)
'''