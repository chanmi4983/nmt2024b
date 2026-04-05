import torch

# 체크포인트 파일 경로
checkpoint_path = './results/en2krContextTest_case3_1.gpu5.demo.pth'

# 체크포인트 불러오기
checkpoint = torch.load(checkpoint_path)

# 저장된 정보 출력
print("Checkpoint Keys:", checkpoint.keys())  # 체크포인트에 저장된 키 확인
print("Iteration:", checkpoint['iloop'])  # 마지막 iteration 확인
print("Model State Dict:", checkpoint['state_dict'].keys())  # 모델의 state_dict 확인
