"""
Example code of a simple RNN, GRU, LSTM on the MNIST dataset.

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-05-09 Initial coding
*    2022-12-16 Updated with more detailed comments, docstrings to functions, and checked code still functions as intended.

"""
"""
위에 사람이 만든 MNIST용 RNN계열 모델중에 GRU만 똑 떼와서 MIT-BIH 용도로 재작업
"""

# Imports
import os
import torch
import numpy as np # .npy 읽기용
import pandas as pd # csv 읽기용
import torch.nn.functional as F  # 일부 활성화 함수 등 파라미터 없는 함수에 사용
import torchvision.datasets as datasets  # 일반적인 데이터셋; 이거 아마 MIT-BIH로 바꿔야 할 듯?
import torchvision.transforms as transforms  # 데이터 증강을 위한 일종의 변형작업이라 함
from torch import optim  # SGD, Adam 등의 옵티마이저
from torch import nn  # 모든 DNN 모델들
from torch.utils.data import (
    DataLoader, Dataset,
)  # 미니배치 등의 데이터셋 관리를 도와주는 녀석
from tqdm import tqdm  # 진행도 표시용

# Cuda 써야겠지?
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # GPU 번호별로 0번부터 나열
os.environ["CUDA_VISIBLE_DEVICES"]= "2"  # 일단 원석이가 0, 1번 쓰고 있다 하니 2번으로 지정
device = "cuda" if torch.cuda.is_available() else "cpu" # 연산에 GPU 쓰도록 지정
print("Device :" + device) # 확인용
# input() # 일시정지용

# 하이퍼파라미터와 사전 설정값들(당장은 여기서 조정해가면서 시도해볼 것)
input_size = 10 # 입력사이즈; MNIST라서 가로세로 찢어서 넣는거같은데 내 경우는 일단 10개로 해보자.
hidden_size = 256 # 히든레이어 크기; 이정도면 적절히 충분하겠지?
num_layers = 2 # 레이어 크기; 히든과 출력 이렇게 2개 말하는듯
num_classes = 2 # 클래스 갯수; 난 일단 정상/비정상만 볼 것이니 2개로 지정
sequence_length = 187 # 시퀀스 길이; MIT-BIH 길이에 맞춰야 함, 총 188개 열에 마지막 값은 라벨이므로 187개의 길이가 됨
learning_rate = 0.005 # 러닝레이트
batch_size = 64 # 배치크기(웬만해선 줄일수록 좋다지만 일단 이대로 놓고 천천히 줄여보기)
num_epochs = 3 # 에포크(이거 나중에 early stop 걸어야 함)
num_workers = 8 # 데이터 불러올 때 병렬화 갯수
train_path = "/data/common/MIT-BIH/mitbih_train.csv" # 훈련데이터 위치
test_path = "/data/common/MIT-BIH/mitbih_test.csv" # 테스트데이터 위치
train_encoded_path = "/data/leehyunwon/MIT-BIH_TP_encoding/mitbih_train_encoded.npy" # 인코딩된 훈련데이터 위치
test_encoded_path = "/data/leehyunwon/MIT-BIH_TP_encoding/mitbih_test_encoded.npy" # 인코딩된 테스트데이터 위치


# RNN 기반 GRU 모델 (many-to-one이니 내 작업에 그대로 쓸 수 있음)
class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes) # 마지막에 나오는 풀커넥트로 classification

    def forward(self, x):
        # 히든상태와 셀상태 초기화
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # 순전파
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)

        # 풀커넥트로 classification
        out = self.fc(out)
        return out
    




# 데이터 가져오기(아마 여길 가장 많이 바꿔야 할 듯,,,)
# MIT-BIH를 인코딩해야 하므로, 공간 많이 잡아먹지 싶다. 이건 /data/leehyunwon/ 이쪽에 변환 후 넣고 나서 여기서 불러오는 식으로 해야 할 듯

# 커스텀 데이터셋 관리 클래스
class MITLoader(Dataset):

    def __init__(self, original_csv, encoded_npy, transforms: None) -> None:
        super().__init__()
        self.annotations = pd.read_csv(original_csv).values # MIT 라벨 읽기용
        self.encoded = np.load(encoded_npy) # MIT 인코딩된 데이터 로딩용
        self.transforms = transforms
        
        # 근데 이제 RNN은 입력 텐서의 차원을 (시퀀스, 배치, 입력크기) 로 기대하므로, 원본 인코딩 데이터의 (데이터, 입력크기(뉴런갯수), 시퀀스) 를 변형해야 한다.
        # 참고로 배치는 나중에 추가되는거니까 크게 신경 안써도 되고, 데이터도 어차피 인덱스별로 날아가므로 시퀀스와 입력크기 순서를 바꾸도록 한다.
        self.encoded = np.transpose(self.encoded, (0, 2, 1))

    def __len__(self):
        return self.annotations.shape[0] # shape은 차원당 요소 갯수를 튜플로 반환하므로 행에 해당하는 0번 값 반환 : 이것도 혹시 모르니 변환된 데이터에 대한 걸로 바꿀까? 걍 냅둘까?

    def __getitem__(self, item):
        signal = self.encoded[item, :-1] # 마지막꺼 빼고 집어넣기
    
        # numpy 배열을 텐서로 변경
        signal = torch.from_numpy(signal).float()
        
        # transform이 있는 경우 적용
        if self.transforms:
            signal = self.transforms(signal)
            
        # 라벨 변경 : 이진 분류를 할 것이므로 0인 경우 0, 아니면 1로 바꿔야 함
        label = int(self.annotations[item, -1])
        if label > 0:
            label = 1  # 1 이상인 모든 값은 1로 변환(난 이진값 처리하니깐)
            
        label = torch.tensor(label, dtype=torch.long) # 라벨은 마지막꺼만 집어넣고 텐서로 변환 : 이건 그대로 둬도 될듯?

        return signal, label



# 일단 raw 데이터셋 가져오기
train_dataset = MITLoader(original_csv=train_path, encoded_npy=train_encoded_path)
test_dataset = MITLoader(original_csv=test_path, encoded_npy=test_encoded_path)

# 랜덤노이즈, 랜덤쉬프트는 일단 여기에 적어두기만 하고 구현은 미뤄두자.


# 레거시 : MNIST 넣을땐 일단 이렇게 했음.. 근데 이제 외부에서 끌고 오는 거라서 이걸 이제 수작업으로 바꿔야 한다는 것
# train_dataset = datasets.MNIST(
#     root="dataset/", train=True, transform=transforms.ToTensor(), download=True
# )
# test_dataset = datasets.MNIST(
#     root="dataset/", train=False, transform=transforms.ToTensor(), download=True
# )

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # 물론 이건 그대로 써도 될 듯?
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)






# 네트워크 초기화
model = RNN_GRU(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss와 optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습 시작(학습추이 확인해야 하니 훈련, 평가 모두 Acc, F1, AUROC, AUPRC 넣을 것!)
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # 데이터 cuda에 갖다박기
        data = data.to(device=device).squeeze(1) # 일차원이 있으면 제거, 따라서 batch는 절대 1로 두면 안될듯
        targets = targets.to(device=device)

        # 순전파
        scores = model(data)
        loss = criterion(scores, targets)

        # 역전파
        optimizer.zero_grad()
        loss.backward()

        # 아담 옵티머스 프라임 출격
        optimizer.step()

# test 데이터로 정확도 측정
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # 모델 평가용으로 전환
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # 모델 다시 훈련으로 전환
    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")