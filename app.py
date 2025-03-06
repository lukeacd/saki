import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
from sklearn.linear_model import LinearRegression
import random

inputs = st.text_input('Enter new sequence (or press Enter to use default): ')


alpha1="alpha_helix_sequences.csv"
feature1="feature_matrix.csv"


def check_files(*file_paths):
    """필요한 파일들이 존재하는지 확인합니다."""
    """
    missing_files = [fp for fp in file_paths if not os.path.exists(fp)]
    if missing_files:
        for fp in missing_files:
            st.write(f"Error: {fp} not found. Please ensure the file exists or adjust the path.")
        sys.exit(1)"""
    pass


def load_feature_matrix(file_path):
    """feature_matrix CSV 파일을 로드하고 필수 컬럼이 있는지 검증합니다."""
    df = pd.read_csv(file_path)
    required_columns = {"Min_Levenshtein", "Avg_BLOSUM62"}
    if not required_columns.issubset(df.columns):
        st.write(f"Error: {file_path} must contain the columns: {required_columns}.")
        sys.exit(1)
    return df


def group_and_filter_data(df):
    """
    'Min_Levenshtein'별로 그룹화한 후,
    각 그룹에서 최소 'Avg_BLOSUM62'를 선택하고,
    'Min_Levenshtein' 값이 4 이상인 행만 반환합니다.
    """
    grouped = df.groupby("Min_Levenshtein", as_index=False)["Avg_BLOSUM62"].min()
    grouped = grouped[grouped["Min_Levenshtein"] >= 4]
    return grouped


def perform_linear_regression(X, y):
    """선형 회귀 모델을 학습하고 결과를 반환합니다."""
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)
    pred = model.predict(X)
    return model, slope, intercept, r2, pred


def load_dataset_sequences(file_path):
    """dataset_sequences CSV 파일에서 시퀀스 목록을 로드합니다."""
    df_sequences = pd.read_csv(file_path)
    if "sequence" not in df_sequences.columns:
        st.write(f"Error: {file_path} must contain a 'sequence' column.")
        sys.exit(1)
    return df_sequences["sequence"].tolist()


def load_new_sequence(default_seq="ACDEFGHIKLMNPQRSTVWY"):
    global inputs
    """
    새 시퀀스를 사용자 입력으로 받습니다.
    사용자가 아무 입력도 하지 않으면 기본 시퀀스를 반환합니다.
    """
    #new_seq = input("Enter new sequence (or press Enter to use default): ").strip()
    new_seq=inputs
    if not new_seq:
        new_seq = default_seq
    return new_seq


def levenshtein_distance(seq1, seq2):
    """두 시퀀스 간의 Levenshtein 거리를 계산합니다."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def blosum62_similarity(seq1, seq2, aligner):
    """두 시퀀스 간의 BLOSUM62 similarity (global alignment) 점수를 반환합니다."""
    return aligner.score(seq1, seq2)


def compute_features_for_new_sequence(new_seq, dataset_sequences, aligner):
    """
    새 시퀀스에 대해 두 가지 특징을 계산합니다.
    - Feature 1: dataset_sequences 내 시퀀스들과의 최소 Levenshtein 거리.
    - Feature 2: dataset_sequences 내 시퀀스들과의 평균 BLOSUM62 similarity.
    """
    lev_dists = [levenshtein_distance(new_seq, ds) for ds in dataset_sequences]
    feat1 = min(lev_dists)
    blo_scores = [blosum62_similarity(new_seq, ds, aligner) for ds in dataset_sequences]
    feat2 = np.mean(blo_scores)
    return np.array([feat1, feat2])


def configure_aligner():
    """BLOSUM62를 사용하도록 PairwiseAligner를 구성합니다."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    # 필요한 경우 gap penalty를 설정할 수 있습니다.
    # aligner.open_gap_score = -10
    # aligner.extend_gap_score = -0.5
    return aligner


def plot_data_and_fit(df, X, pred, new_feature):
    """데이터와 회귀 결과, 그리고 새 시퀀스 포인트를 시각화합니다."""
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Min_Levenshtein"], df["Avg_BLOSUM62"], color="blue", label="Data points",s=10)
    plt.plot(X, pred, color="red", label="Linear fit")
    plt.scatter(new_feature[0], new_feature[1], color="red", s=150, marker="*", label="New Sequence")
    plt.xlabel("Min_Levenshtein")
    plt.ylabel("Minimum Avg_BLOSUM62 Score")
    plt.title("Linear Regression: Min_Levenshtein vs. Avg_BLOSUM62")
    plt.legend()
    st.pyplot(plt.gcf())


def sequence_module():
    # 파일 경로 설정
    FEATURE_MATRIX_FILE = feature1
    ALPHA_SEQ_FILE = alpha1

    # 1. 파일 존재 여부 확인
    check_files(FEATURE_MATRIX_FILE, ALPHA_SEQ_FILE)

    # 2. feature_matrix 로드 및 검증
    df = load_feature_matrix(FEATURE_MATRIX_FILE)

    # 3. 데이터 그룹화 및 필터링
    grouped = group_and_filter_data(df)

    # 4. 선형 회귀 수행
    X = grouped[["Min_Levenshtein"]].values
    y = grouped["Avg_BLOSUM62"].values
    model, slope, intercept, r2, pred = perform_linear_regression(X, y)

    st.write("\n=== Linear Regression Results ===")
    st.write(f"Slope:      {slope:.4f}")
    st.write(f"Intercept:  {intercept:.4f}")
    st.write(f"R^2 score:  {r2:.4f}")

    # 5. 새 시퀀스 로드 (사용자 입력 또는 기본값 사용)
    new_seq = load_new_sequence()
    st.write("\nNew sequence:", new_seq)

    # 6. PairwiseAligner 구성
    aligner = configure_aligner()

    # 7. dataset_sequences 로드
    dataset_sequences = load_dataset_sequences(ALPHA_SEQ_FILE)

    # 8. 새 시퀀스에 대한 특징 계산
    new_feature = compute_features_for_new_sequence(new_seq, dataset_sequences, aligner)
    st.write("New sequence feature vector:", new_feature)

    # 9. 예측 수행 (여기서는 Min_Levenshtein 값만 사용)
    predicted_value = model.predict(np.array([[new_feature[0]]]))[0]
    st.write("Predicted regression value:", predicted_value)

    # 10. 분류 결정 (예시: predicted_value < 평균 BLOSUM62 점수이면 "Alpha")
    decision = "Alpha" if predicted_value < new_feature[1] else "Not Alpha"
    st.write("Decision for Output sequence:", decision)

    # 11. 결과 시각화
    plot_data_and_fit(df, X, pred, new_feature)
    


# alpha_helix_sequences.csv 파일에서 무작위 서열 선택
alpha = pd.read_csv(alpha1)
n = random.randint(1, len(alpha))
# 'ID' 컬럼이 n인 행의 'sequence' 값을 가져와서, 개별 아미노산 문자 리스트로 변환
selected_seq = alpha[alpha['ID'] == n]['sequence'].values[0]
AMINO_ACIDS = list(selected_seq)
st.write(f"Selected sequence for AMINO_ACIDS (ID={n}): {selected_seq}")


def modify_sequence(seq, method="random_substitution", mutation_rate=0.1):
    """
    기존 서열을 변형하는 함수.

    매개변수:
    - seq: 변형할 원본 서열
    - method: 변이 방법 선택
        "random_substitution": 각 위치마다 일정 확률로 무작위 치환 (기본)
        "single_substitution": 단일 위치에서 한 번만 치환
        "insertion": 무작위 위치에 한 개의 아미노산 삽입
        "deletion": 무작위 위치의 아미노산 삭제
    - mutation_rate: 치환 확률 (method가 "random_substitution"인 경우)
    """
    if method == "random_substitution":
        new_seq = ""
        for char in seq:
            if np.random.rand() < mutation_rate:
                new_seq += random.choice(AMINO_ACIDS)
            else:
                new_seq += char
        return new_seq
    elif method == "single_substitution":
        if len(seq) == 0:
            return seq
        index = random.randint(0, len(seq) - 1)
        # 원래 아미노산과 다른 아미노산으로 치환
        possible = [aa for aa in AMINO_ACIDS if aa != seq[index]]
        new_amino = random.choice(possible)
        new_seq = seq[:index] + new_amino + seq[index + 1:]
        return new_seq
    elif method == "insertion":
        index = random.randint(0, len(seq))
        new_amino = random.choice(AMINO_ACIDS)
        new_seq = seq[:index] + new_amino + seq[index:]
        return new_seq
    elif method == "deletion":
        if len(seq) == 0:
            return seq
        index = random.randint(0, len(seq) - 1)
        new_seq = seq[:index] + seq[index + 1:]
        return new_seq
    else:
        raise ValueError("Unsupported mutation method: " + method)


def generate_new_alpha_sequence(seed, dataset_sequences, aligner, regression_model,
                                mutation_method="single_substitution", mutation_rate=0.1, max_iter=1000):
    """
    모듈의 판단 로직을 활용하여 'Alpha'로 분류되는 새로운 서열을 생성합니다.

    매개변수:
    - seed: 초기 서열
    - dataset_sequences: 기준 데이터셋 서열 목록
    - aligner: PairwiseAligner 객체
    - regression_model: 학습된 선형 회귀 모델
    - mutation_method: 사용할 변이 방법 (예: "single_substitution")
    - mutation_rate: 변이 확률 (치환 방식에 적용)
    - max_iter: 최대 반복 횟수

    반환:
    - 조건을 만족하는 새로운 'Alpha' 서열과 그 특징 벡터.
    - 제한 내에 생성하지 못하면 (None, None) 반환.
    """
    current_seq = seed
    for i in range(max_iter):
        candidate = modify_sequence(current_seq, method=mutation_method, mutation_rate=mutation_rate)
        features = compute_features_for_new_sequence(candidate, dataset_sequences, aligner)
        predicted_value = regression_model.predict(np.array([[features[0]]]))[0]
        decision = "Alpha" if predicted_value < features[1] else "Not Alpha"
        st.write(f"Iteration {i + 1}: Candidate: {candidate}, Features: {features}, Decision: {decision}")
        if decision == "Alpha":
            return candidate, features
        # 다음 변이를 위해 현재 서열 업데이트
        current_seq = candidate
    return None, None


def generator():
    # 파일 경로 설정
    FEATURE_MATRIX_FILE = feature1
    ALPHA_SEQ_FILE = alpha1
    BASE_DIR = r"C:/Users/조용민/Desktop/학교 파일/saki"
    NEW_SEQUENCE_TXT = os.path.join(BASE_DIR, "new_sequence.txt")
    NEW_SEQUENCE_CSV = os.path.join(BASE_DIR, "new_sequence.csv")

    # 1. feature_matrix 로드 및 검증
    df = load_feature_matrix(FEATURE_MATRIX_FILE)

    # 2. 데이터 그룹화 및 필터링
    grouped = group_and_filter_data(df)

    # 3. 선형 회귀 수행
    X = grouped[["Min_Levenshtein"]].values
    y = grouped["Avg_BLOSUM62"].values
    model, slope, intercept, r2, pred = perform_linear_regression(X, y)
    st.write(f"Slope: {slope:.4f}, Intercept: {intercept:.4f}, R^2: {r2:.4f}")

    # 4. 초기 시퀀스(Seed) 로드 (기존 파일 또는 기본값)
    seed_seq = selected_seq
    st.write("Seed sequence:", seed_seq)

    # 5. PairwiseAligner 구성 및 데이터셋 서열 로드
    aligner = configure_aligner()
    dataset_sequences = load_dataset_sequences(ALPHA_SEQ_FILE)

    # 6. 변이 방법 선택: 여기서는 "single_substitution" 방식을 선택
    mutation_method = "single_substitution"

    # 7. 모듈 판단 로직을 활용하여 새로운 "Alpha" 서열 생성
    new_alpha_seq, features = generate_new_alpha_sequence(
        seed_seq, dataset_sequences, aligner, model,
        mutation_method=mutation_method, mutation_rate=0.1, max_iter=1000
    )
    if new_alpha_seq:
        st.write("\nGenerated Output 'Alpha' sequence:", new_alpha_seq)
        st.write("Features of generated sequence:", features)
    else:
        st.write("\nFailed to generate a Output 'Alpha' sequence within the iteration limit.")

    # 8. 결과 시각화 (원하는 경우)
    plot_data_and_fit(df, X, pred, features)



button =st.button("sequence_module")
button2 =st.button("generator")
if button:
    st.write("access")
    sequence_module()
if button2:
    st.write("access2")
    generator()
    
#           streamlit run app.py
