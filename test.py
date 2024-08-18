import torch
import dataloader
import model


def test():
    trainloader,testloader,classes = dataloader.load_cifar10()

    # 학습된 모델 불러오기 -> state dict를 활용했으므로
    net = model.Net()
    net.load_state_dict(torch.load('./model_state_dict.pt'))

    correct  = 0
    total = 0

    # gradient 계산을 일시적으로 끄고 정확도를 계산 (즉, 정확도에서 사용된 계산은 기억될 필요 x)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            ouputs = net(images)
            #torch.max(data, dimension) 해당 dimension 기준 최대값과 그 인덱스 반환
            _, predicted = torch.max(ouputs.data,1)
            # 현재 batch의 이미지 개수
            total += labels.size(0)
            # predicted == labels인 것의 개수를 다 더함 .sum() -> 텐서로 되어 있으므로 정수로 변환 .item()
            correct += (predicted == labels).sum().item()
            
    print('정확도: %d %%'%(100*correct/total))
    
    
    
if __name__ == "__main__":
    test()
                