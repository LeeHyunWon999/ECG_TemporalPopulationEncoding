# 인자명에 해당하는 데이터를 input, output에서 읽어다가 플롯찍어 저장

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

        

# 파일명으로 데이터 찾아서 시각화
def visualize(data) : 
    # 입력, 출력용으로 서로 다른 경로 생성
    inputPath = "./data/input/" + data + ".csv"
    outputPath = "./data/output/" + data + "_encoded.csv"
    
    # 파일명 분리
    # fileName = os.path.basename(json_data["inputPath"]).split('.')[0:-1] # 파일명에 .이 들어간 경우를 위한 처리
    # fileName = ".".join(fileName)
    
    # 데이터 파일 읽기 시도
    inputData = np.loadtxt(inputPath, delimiter=',')
    print(inputData)
    outputData = np.loadtxt(outputPath, delimiter=',')
    print(outputData)
    
    # 입력데이터 플롯찍기
    # plt.plot(inputData)
    # plt.savefig('./data/visual/' + data + '_visual.png')
    # plt.clf()
    
    plt.subplot(1,2,1)
    plt.plot(inputData)
    
    # 출력데이터 플롯찍기 : 뉴런갯수따라 2차원이므로 서브플롯 형태가 좋을듯(0,1째의 타우, g값은 제거필요)
    plot_row = len(outputData)
    # 뉴런 갯수에 해당하는 서브플롯 생성
    for i in range(plot_row) : 
        ith_np = outputData[i]
        # 타우와 g값 제거
        ith_np_deleted = np.delete(ith_np, [0, 1], axis = None)
        plt.subplot(plot_row, 2, (i + 1) * 2)
        plt.bar(np.arange(len(ith_np_deleted)), ith_np_deleted)
    
    # 출력데이터 시각화 저장
    plt.savefig('./data/visual/' + data + '_encoded_visual.png')



if __name__ == "__main__" : 
    if len(sys.argv) != 2 : 
        print("인자가 맞지 않습니다 :", len(sys.argv))
    else : 
        print("대상 파일명 :", sys.argv[1])
        visualize(sys.argv[1])