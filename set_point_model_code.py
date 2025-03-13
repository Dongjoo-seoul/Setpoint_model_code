import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import shap

class WeightDataset(Dataset):
    def __init__(self, time_series_data, static_data, targets):
        self.time_series_data = torch.FloatTensor(time_series_data)
        self.static_data = torch.FloatTensor(static_data)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.time_series_data)
    
    def __getitem__(self, idx):
        return {
            'time_series': self.time_series_data[idx],
            'static': self.static_data[idx],
            'target': self.targets[idx]
        }

def preprocess_data(time_series_df, static_df):
    """
    데이터 전처리를 수행하는 함수
    
    Args:
        time_series_df: 시계열 데이터 (체중, 근육량, 체지방 등)
        static_df: 정적 데이터 (성별, 생활습관 등)
    
    Returns:
        처리된 데이터셋
    """
    # 결측치 처리
    time_series_df = time_series_df.interpolate(method='linear')
    static_df = static_df.fillna(0)  # 이진 데이터의 경우
    
    # 정규화
    time_scaler = StandardScaler()
    static_scaler = StandardScaler()
    
    time_series_normalized = time_scaler.fit_transform(time_series_df)
    
    # 범주형 데이터 원-핫 인코딩
    categorical_cols = ['gender', 'smoking_status', 'alcohol_status']  # 예시
    onehot = OneHotEncoder(sparse=False)
    categorical_encoded = onehot.fit_transform(static_df[categorical_cols])
    
    # 수치형 데이터 정규화
    numerical_cols = [col for col in static_df.columns if col not in categorical_cols]
    numerical_normalized = static_scaler.fit_transform(static_df[numerical_cols])
    
    # 정적 데이터 결합
    static_processed = np.hstack([categorical_encoded, numerical_normalized])
    
    return time_series_normalized, static_processed

def create_sequences(data, seq_length):
    """
    시계열 데이터를 시퀀스로 변환하는 함수
    
    Args:
        data: 원본 시계열 데이터
        seq_length: 시퀀스 길이
    
    Returns:
        시퀀스 형태로 변환된 데이터
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
        
    return np.array(sequences), np.array(targets)

class SetPointModel(nn.Module):
    def __init__(self, time_series_dim, static_dim, hidden_dim, num_layers):
        """
        세트 포인트 이론을 검증하기 위한 듀얼 브랜치 모델
        
        Args:
            time_series_dim: 시계열 데이터의 차원
            static_dim: 정적 데이터의 차원
            hidden_dim: LSTM과 MLP의 은닉층 차원
            num_layers: LSTM 레이어 수
        """
        super(SetPointModel, self).__init__()
        
        # 시계열 브랜치 (LSTM)
        self.lstm = nn.LSTM(
            input_size=time_series_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 정적 데이터 브랜치 (MLP)
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 통합 레이어
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 멀티태스크 출력 레이어
        self.weight_pred_layer = nn.Linear(hidden_dim, 1)  # W_pred
        self.set_point_layer = nn.Linear(hidden_dim, 1)    # SP
        self.k_pred_layer = nn.Linear(hidden_dim, 1)       # k_pred
        
    def forward(self, time_series, static_data):
        # LSTM 브랜치 처리
        lstm_out, (h_n, c_n) = self.lstm(time_series)
        lstm_features = lstm_out[:, -1, :]  # 마지막 타임스텝의 출력
        
        # 정적 데이터 브랜치 처리
        static_features = self.static_net(static_data)
        
        # 특성 통합
        combined_features = torch.cat([lstm_features, static_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # 멀티태스크 예측
        w_pred = self.weight_pred_layer(fused_features)
        sp = self.set_point_layer(fused_features)
        k_pred = self.k_pred_layer(fused_features)
        
        return w_pred, sp, k_pred

class SetPointLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        세트 포인트 모델의 손실 함수
        
        Args:
            alpha: 주 손실과 보조 손실 간의 가중치 균형을 조정하는 파라미터
        """
        super(SetPointLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, w_pred, sp, k_pred, w_t, w_next, delta_t):
        # 주 손실: 다음 체중 예측의 MSE
        primary_loss = self.mse(w_pred, w_next)
        
        # 보조 손실: 세트 포인트 방정식 기반
        model_change = k_pred * delta_t * (sp - w_t)
        actual_change = w_next - w_t
        auxiliary_loss = torch.mean(torch.abs(actual_change - model_change))
        
        # 전체 손실
        total_loss = primary_loss + self.alpha * auxiliary_loss
        
        return total_loss, primary_loss, auxiliary_loss

def train_model(model, train_loader, val_loader, num_epochs, device, learning_rate=0.001):
    """
    모델 학습을 수행하는 함수
    
    Args:
        model: SetPointModel 인스턴스
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        num_epochs: 학습 에포크 수
        device: 학습에 사용할 디바이스 (CPU/GPU)
        learning_rate: 학습률
    """
    model = model.to(device)
    criterion = SetPointLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 학습 단계
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            time_series = batch['time_series'].to(device)
            static = batch['static'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            w_pred, sp, k_pred = model(time_series, static)
            
            # 현재 체중과 시간 간격 추출 (데이터셋에서 적절히 수정 필요)
            w_t = time_series[:, -1, 0]  # 마지막 타임스텝의 체중
            delta_t = torch.ones_like(w_t)  # 실제 구현시 적절한 시간 간격으로 수정 필요
            
            loss, _, _ = criterion(w_pred, sp, k_pred, w_t, targets, delta_t)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                time_series = batch['time_series'].to(device)
                static = batch['static'].to(device)
                targets = batch['target'].to(device)
                
                w_pred, sp, k_pred = model(time_series, static)
                w_t = time_series[:, -1, 0]
                delta_t = torch.ones_like(w_t)
                
                loss, _, _ = criterion(w_pred, sp, k_pred, w_t, targets, delta_t)
                val_loss += loss.item()
        
        # 학습률 조정
        scheduler.step(val_loss)
        
        # 체크포인트 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model_checkpoint.pth')
        
        print(f'에포크 {epoch+1}/{num_epochs}')
        print(f'학습 손실: {train_loss/len(train_loader):.4f}')
        print(f'검증 손실: {val_loss/len(val_loader):.4f}') 

def analyze_feature_importance(model, data_loader, device):
    """
    SHAP를 사용하여 특성 중요도를 분석하는 함수
    
    Args:
        model: 학습된 SetPointModel 인스턴스
        data_loader: 분석에 사용할 데이터 로더
        device: 모델이 로드된 디바이스
    """
    model.eval()
    
    # 배경 데이터 수집
    background = []
    for batch in data_loader:
        time_series = batch['time_series'].to(device)
        static = batch['static'].to(device)
        background.append((time_series, static))
        if len(background) >= 100:  # 배경 데이터 제한
            break
    
    # SHAP 설명자 생성
    def model_wrapper(time_series, static):
        return model(time_series, static)[0].cpu().detach().numpy()
    
    explainer = shap.DeepExplainer(model_wrapper, background)
    
    # SHAP 값 계산
    shap_values = explainer.shap_values(next(iter(data_loader)))
    
    # 시각화
    shap.summary_plot(shap_values, feature_names=['시계열 특성', '정적 특성'])

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    TIME_SERIES_DIM = 7  # 체중, 근육량, 체지방 등
    STATIC_DIM = 20      # 성별, 생활습관 등
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # CUDA 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'사용 디바이스: {device}')
    
    # 데이터 로드 및 전처리
    # 실제 구현시 데이터 로드 코드 필요
    time_series_data = None  # 시계열 데이터 로드
    static_data = None       # 정적 데이터 로드
    
    # 데이터 전처리
    time_series_processed, static_processed = preprocess_data(time_series_data, static_data)
    
    # 시퀀스 생성
    sequences, targets = create_sequences(time_series_processed, seq_length=10)
    
    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, targets, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=42
    )
    
    # 데이터 로더 생성
    train_dataset = WeightDataset(X_train, static_processed[:len(X_train)], y_train)
    val_dataset = WeightDataset(X_val, static_processed[len(X_train):len(X_train)+len(X_val)], y_val)
    test_dataset = WeightDataset(X_test, static_processed[-len(X_test):], y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 모델 초기화
    model = SetPointModel(
        time_series_dim=TIME_SERIES_DIM,
        static_dim=STATIC_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    )
    
    # 모델 학습
    train_model(model, train_loader, val_loader, NUM_EPOCHS, device, LEARNING_RATE)
    
    # 특성 중요도 분석
    analyze_feature_importance(model, test_loader, device)
    
    print('모델 학습 및 분석이 완료되었습니다.') 