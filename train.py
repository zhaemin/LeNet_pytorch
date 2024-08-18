import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dataloader
import model


def train():
    net = model.Net()
    trainloader,testloader,classes = dataloader.load_cifar10()
    
    criterion = nn.CrossEntropyLoss()
    # SGD -> BGD보다 계산이 빠르고, global minima에 수렴할 가능성이 크다
    # momentum => 관성 : 이전에 내려왔던 방향을 고려하는 변수 ->(속도 velocity) 특정 방향으로 계속 이동시 가속이 붙는다고 생각
    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
    epochs = 2
    
    for epoch in range(epochs):
        
        running_loss = 0.0
        # 배치 단위로 이미지와 정답값을 방음
        # batch size가 4이므로 4장의 이미지마다 학습 -> mini batch로 봐야하나?
        for i,data in enumerate(trainloader, 0):
            
            inputs, labels = data
            
            # 초기 기울기 파라미터 = 0
            # 기본적으로 pytorch는 gradient를 누적시킨다 (batch별 gradient를 더하게 됨)
            # weight는 이미 수정되었으므로 gradient만 초기화!
            optimizer.zero_grad()
            
            #forward -> Input을 모델에 넣음
            outputs = net(inputs)
            # 결과값과 실제값의 loss를 계산하고 해당 loss로 backward + optimize 싱행
            # loss 계산
            loss = criterion(outputs, labels)
            # gradient 값이 계산 , 이때 weight값은 아직 변하지 않음
            loss.backward()
            # model weight를 loss값이 작게 업데이트한다. (미분값의 반대방향으로 움직임)
            optimizer.step()
            
            #epoch 및 loss 출력
            running_loss += loss.item()
            if i % 2000 == 1999: #2000 Mini batch마다 값 출력         
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    
    # 모델 저장하기
    # state dict는 각 layer를 
    torch.save(net.state_dict(), './model_state_dict.pt')
    
    
if __name__ == "__main__":
    train()
                