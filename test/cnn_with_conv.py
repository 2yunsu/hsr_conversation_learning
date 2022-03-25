from camera_ready import camera_ready
import cv2
import pandas as pd
# from cnn import start_cnn
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import cv2
from cnn import CustomImageDataset, Net

labels = pd.read_csv('labels.csv', names=['Name', 'Labels'])




if __name__ == '__main__':
    transform = tr.Compose([tr.Resize([64, 64]), tr.ToTensor(), tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    customset = CustomImageDataset(annotations_file='labels.csv',
                                   img_dir='images',
                                   transform=transform)
    trainset, testset = train_test_split(customset, test_size=0.2)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    net = Net()
    print(net)

    # optimizer 사용 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    while True:
        name = input("file name:")
        camera_ready(name)
        image_name = name+'.png'
        input_label = pd.DataFrame({'Name':[image_name], 'Labels': [input("Label: ")]})
        if image_name in labels[['Name']].values:
            print("file name is already exists: ")
            continue
        labels=labels.append(input_label, ignore_index=True)
        labels.to_csv("labels.csv", mode='w', index=False, header=False)
        start_learning = input("start learning?(Y/N): ")

        if start_learning=="Y" or start_learning=="y":
            # start train
            for epoch in tqdm(range(5)):
                running_loss = 0.0
                # traindata 불러오기(배치 형태로 들어옴)
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    # optimizer 초기화
                    optimizer.zero_grad()
                    # net에 input 이미지 넣어서 output 나오기
                    outputs = net(inputs)
                    # output로 loss값 계산
                    loss = criterion(outputs, labels)
                    # loss를 기준으로 미분자동계산
                    loss.backward()
                    # optimizer 계산
                    optimizer.step()
                    # loss값 누적
                    running_loss += loss.item()
                    if i % 2000 == 1999:
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 2000))
                        running_loss = 0.0

        finish_answer = input('finish the training?(Y/N): ')
        if finish_answer=="Y" or finish_answer=="y":
            break

    # 학습한 모델 저장
    PATH = "hsr_net.pth"
    torch.save(net.state_dict(), PATH)