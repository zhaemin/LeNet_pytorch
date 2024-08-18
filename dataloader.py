import torch
import torchvision
import torchvision.transforms as transforms


def load_cifar10():
    transform = transforms.Compose(
        # ToTensor()는 이미지의 픽셀 값 범위를 0~1로 조정
        # normalize 하는 이유 -> 오차역전파시 gradient계산 수행 -> 데이터가 유사한 범위를 가지도록 하기 위함
        # transofrms.Normalize((R채널 평균, G채널 평균, B채널 평균), (R채널 표준편차, G채널 표준편차, B채널 표준편차))
        # 각 채널 별 평균을 뺀 후 표준편차로 나누어 계산
        # 아래 예시에서는 -1 ~ 1로 변환
        [transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5, 0.5, 0.5))])

    # 데이터셋 instance 생성
    # 데이터를 저장하려는 파일시스템 경로, 학습용 여부, 다운로우 여부, transform 객체
    trainset = torchvision.datasets.CIFAR10(root = './data', train=True, download=True, transform=transform)
    # 무작위 추출한 4개의 batch image를 trainset에서 추출
    # num workers => 복수 개의 프로세스
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root = './data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car','bird','cat','deer','dog','frog','horse','ship','truck')
    
    
    return trainloader,testloader,classes