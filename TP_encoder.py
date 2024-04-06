import sys
import os
import json
import numpy as np

import torch
from spikingjelly.activation_based import neuron

# 인코딩용 뉴런 정의
class TP_neuron(neuron.BaseNode) : 
    
    # 생성자
    def __init__(self, tau, g, threshold = 1.0, reset = False, reset_value = 0.0, leaky = False) : 
        super().__init__()
        self.v_threshold = threshold
        self.v_reset = None if reset is False else reset_value
        self.leaky = leaky
        self.tau = tau
        self.g = g
    
    # 인코딩 스텝
    def neuronal_charge(self, x: torch.Tensor) : 
        if self.leaky : 
            # LIF 뉴런 체계인 경우 leak 값인 tau를 이용하여 계산
            self.v = np.exp(-1 / self.tau) * self.v + (x * self.g)
        else : 
            self.v = self.v + (x * self.g)
    


# json 읽어다가 반환(파일경로 없으면 에러띄우기)
def loadJson() : 
    if (len(sys.argv) != 2) : 
        print("config.json 파일 경로가 없거나 그 이상의 인자가 들어갔습니다!", len(sys.argv))
        exit()
    else : 
        with open(sys.argv[1], 'r') as f:
            print("config.json파일 읽기 성공!")
            return json.load(f)
        

# 인코딩하고 결과 저장까지 진행
def encode(json_data) : 
    # 파일명 분리
    fileName = os.path.basename(json_data["inputPath"]).split('.')[0:-1] # 파일명에 .이 들어간 경우를 위한 처리
    fileName = ".".join(fileName)
    # 데이터 파일 읽기 시도
    inputData = np.loadtxt(json_data["inputPath"], delimiter=',')
    
    # 파일 형변환
    inputData = torch.tensor(inputData)

    # print(inputData)
    # print(fileName)
    
    # 임시 : 뉴런 생성(뉴런 여러개 만들기)
    # 임시 : 필요한 경우 내부 막전위값 변화를 timestep별로 보도록 할 수도 있겠지만.. 일단은 패스
    neuron_list = []
    encoded_list = []
    for i in range(json_data["dim"]) : 
        # Leaky인 경우 tau값은 0.5~1.5 사이에서 랜덤 지정
        if json_data["leaky"] : 
            this_tau = np.random.rand() + 0.5
        else : 
            this_tau = 1
        neuron_list.append(TP_neuron(tau = this_tau, g = ((float(i) + 1) / float(json_data["dim"])) + 0.5))
        neuron_list[i].step_mode = 'm'
        encoded_list.append(neuron_list[i](inputData))
        print(i,"째 뉴런 인코딩 결과 : ", encoded_list[i])
        neuron_list[i].reset()
    
    
    # 결과값(텐서) 의 첫 열과 둘째 열에 각각 tau, g 값 추가
    for i in range(len(encoded_list)) : 
        encoded_list[i] = encoded_list[i].tolist()
        encoded_list[i].insert(0, neuron_list[i].g)
        encoded_list[i].insert(0, neuron_list[i].tau)
        
    print("인코딩 완료")
    
    # 임시 : csv로 저장(각 뉴런들의 결과 값 리스트 합치고 저장)
    np.savetxt('./data/output/' + fileName + '_encoded.csv', encoded_list, fmt="%f", delimiter=',')



    print("저장 완료")



if __name__ == "__main__" : 
    json_data = loadJson()
    # config 파일 출력
    print(json_data)

    encode(json_data) # 인코딩 및 저장 함수