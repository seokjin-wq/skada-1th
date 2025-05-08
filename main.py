import streamlit as st
import glob
import pandas as pd
import src
import matplotlib.pyplot as plt
import torch



st.title('철근 단가 예측')

# files = glob.glob('./*.xlsx')

# option = st.selectbox(
#     '데이터 파일을 선택해주세요.',
#     files
# )
# st.write('선택된 데이터 파일:', option)

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

with st.sidebar:
    uploaded_file = st.file_uploader('파일 업로드')

    download_area = st.empty()
    
if uploaded_file:
    # 업로드 완료시
    df = pd.read_excel(uploaded_file)

    tab1, tab2 = st.tabs(["시각화", "머신러닝 예측"])

    with tab1:
        # 그래프를 그릴 컬럼 선택(다중)
        options = st.multiselect(
            '그래프에 출력할 컬럼 선택(다중선택)',
            df.columns,
            [])

        # 차트 생성
        st.line_chart(x='Date', y=options, data=df)

    with tab2:
        # 컬럼 선택
        option = st.selectbox('예측할 가격 컬럼을 선택해주세요.', df.columns)

        days = st.number_input('예측할 기간을 입력해 주세요(기본값: 45일)',
                                min_value=10,
                                max_value=100,
                                value=45,
                                )
        days = int(days)

        btn = st.button('선택 완료')

        if btn:
            # Datasets 생성
            dataset = src.DataSets(df, option, split_days=days, batch_size=32, window_size=6)

            # train, test 데이터 로더
            train_loader = dataset.get_train_loader()
            test_loader = dataset.get_test_loader()

            # device 설정
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            with st.spinner('모델 학습중...'):
                # 모델 훈련
                losses, model = src.train(train_loader, num_epochs=600, device=device)

                # test_loader 에 대한 추론
                y_trues, preds = src.evaluate(test_loader, model, dataset.get_scaler())

            # 예측, 실제 값 비교 시각화
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(8, 4)
            ax.plot(preds, color='tomato', linewidth=1.0, label='Prediction', linestyle='-.')
            ax.plot(y_trues, color='skyblue', linewidth=1.0, label='Actual')
            ax.set_title("Prediction vs Actual on {days} days")
            ax.legend()
            fig

            date_index = dataset.series.index[-days:]

            df_result = pd.DataFrame({
                'Date': date_index.strftime("%Y-%m-%d"),
                'Ground Truth': y_trues,
                'Predictions': preds
            })

            st.dataframe(df_result)

            # 다운로드 버튼
            csv = convert_df(df_result)

            download_area.download_button(
                label="예측 결과 다운로드",
                data=csv,
                file_name="예측결과.csv",
                mime='text/csv'
            )
