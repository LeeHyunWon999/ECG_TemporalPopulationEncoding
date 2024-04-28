# 인자명에 해당하는 데이터를 input, output에서 읽어다가 플롯찍어 저장

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

        

# 파일명으로 데이터 찾아서 시각화
def visualize(data) : 
    # 입력, 출력용으로 서로 다른 경로 생성
    inputPath = "./data/input/" + data + ".csv"
    outputPath = "./data/output/" + data + "_encoded.npy"
    
    # MIT용 데이터는 다른 곳에 있으므로 수동으로 입력해야 할 듯...
    print("MIT-BIH의 데이터를 시각화합니까? (Y/N) : ", end=" ")
    userInput = input()
    if userInput == "Y" : 
        inputPath = "/data/common/MIT-BIH/mitbih_test.csv"
        outputPath = "/data/leehyunwon/MIT-BIH_TP_encoding/mitbih_test_encoded.npy"
    
    
    
    # 파일명 분리
    # fileName = os.path.basename(json_data["inputPath"]).split('.')[0:-1] # 파일명에 .이 들어간 경우를 위한 처리
    # fileName = ".".join(fileName)
    
    # 데이터 파일 읽기 시도 및 확인
    inputData = np.loadtxt(inputPath, delimiter=',')
    outputData = np.load(outputPath)
    print(inputData)
    print(outputData)
    
    # 입력데이터 플롯찍기
    # plt.plot(inputData)
    # plt.savefig('./data/visual/' + data + '_visual.png')
    # plt.clf()
    
    # plt.subplot(1,2,1) # 1행 2열로 나눈 구간의 첫번째 구역에 해당하는 크기만큼의 플롯구역 지정
    # plt.plot(inputData) 여기선 이렇게 하면 안되겠지요?
    
    
    # 색 개별 지정하려면 전체 figure 생성해야 돼서..
    fig = plt.figure()
    
    inputDataRow = inputData.shape[0] if inputData.shape[0] <= 3 else 3
    
    for i in range(inputDataRow):  # inputData의 행의 개수만큼 반복
        ax = fig.add_subplot(inputDataRow, 2, i * 2 + 1)
        ax.plot(inputData[i])  # 각 열을 개별적으로 플롯
    
    
    
    
    
    # # 출력데이터 플롯찍기 : 뉴런갯수따라 2차원이므로 서브플롯 형태가 좋을듯(0,1째의 타우, g값은 제거필요)
    # plot_row = len(outputData)
    # # 뉴런 갯수에 해당하는 서브플롯 생성
    # for i in range(plot_row) : 
    #     ith_np = outputData[i]
    #     # 타우와 g값 제거
    #     ith_np_deleted = np.delete(ith_np, [0, 1], axis = None)
    #     plt.subplot(plot_row, 2, (i + 1) * 2)
    #     plt.bar(np.arange(len(ith_np_deleted)), ith_np_deleted)
    
    # 출력데이터 플롯찍기 : 데이터 갯수만큼 빅로우, 뉴런갯수만큼 스몰로우.
    plot_bigRow = outputData.shape[0] if outputData.shape[0] <= 3 else 3
    plot_smallRow = outputData.shape[1] if outputData.shape[1] <= 8 else 8
    
    # 색 바꾸려면 좀 복잡하게 들어가야 하는듯
    
    
    
    # 빅로우만큼 반복
    for i in range(plot_bigRow) : 
        # 스몰로우만큼 반복
        for j in range(plot_smallRow) : 
            # 이제 해당하는 만큼 가져와서 찍기
            ax = fig.add_subplot(plot_bigRow * plot_smallRow, 2, (i * plot_smallRow + j + 1) * 2)
            ax.bar(np.arange(outputData.shape[2]), outputData[i][j])
            if j == 0 : 
                ax.set_facecolor('lightblue') # 구분용 색지정
    
    # 출력데이터 시각화 저장
    plt.savefig('./data/visual/' + data + '_encoded_visual.png')



if __name__ == "__main__" : 
    if len(sys.argv) != 2 : 
        print("인자가 맞지 않습니다 :", len(sys.argv))
    else : 
        print("대상 파일명 :", sys.argv[1])
        visualize(sys.argv[1])