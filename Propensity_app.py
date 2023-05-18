import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def preprocess_data(df, dummy_columns):
    df = df.fillna(0)
    # ダミー変数化
    df_dummy = pd.get_dummies(df, columns=dummy_columns)

    return df_dummy

def calculate_propensity_score(df_dummy, assignment_columns, covariate_columns):
    propensity_scores = []
    inverse_probability_weights = []

    for col in assignment_columns:
        # 割り当て変数と共変量のデータを抽出
        assignment_data = df_dummy[col].values.reshape(-1, 1)
        covariate_data = df_dummy[covariate_columns]

        # 共変量の標準化
        scaler = StandardScaler()
        covariate_data = scaler.fit_transform(covariate_data)

        # ロジスティック回帰モデルを構築し、傾向スコアを予測
        model = LogisticRegression()
        model.fit(covariate_data, assignment_data)
        propensity_score = model.predict_proba(covariate_data)[:, 1]  # 1のクラスの予測確率を取得

        # 逆確率重み付け変数の計算
        inverse_probability_weight = 1 / propensity_score

        propensity_scores.append(propensity_score)
        inverse_probability_weights.append(inverse_probability_weight)

    # 傾向スコアと逆確率重み付け変数の結合
    propensity_scores = np.column_stack(propensity_scores)
    inverse_probability_weights = np.column_stack(inverse_probability_weights)

    return propensity_scores, inverse_probability_weights

def main():
    # ファイルのアップロード
    st.title("傾向スコア算出アプリ")
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

    if uploaded_file is not None:
        # CSVデータの読み込み
        df = pd.read_csv(uploaded_file)

        # ユーザーによるダミー変数化する変数の選択
        dummy_columns = st.multiselect("ダミー変数化する変数を選択してください", df.columns.tolist())

        # データの前処理
        df_dummy = preprocess_data(df, dummy_columns)

        # ユーザーによる割り当て変数と共変量の選択
        assignment_columns = st.multiselect("割り当て変数を選択してください", df_dummy.columns.tolist())
        covariate_columns = st.multiselect("共変量を選択してください", df_dummy.columns.tolist())

        # 傾向スコアの計算
        propensity_scores, inverse_probability_weights = calculate_propensity_score(df_dummy, assignment_columns, covariate_columns)

        # 傾向スコアと逆確率重み付け変数をデータフレームに追加
        for i, col in enumerate(assignment_columns):
            df[col + "_PropensityScore"] = propensity_scores[:, i]
            df[col + "_InverseProbabilityWeight"] = inverse_probability_weights[:, i]

        for col in assignment_columns:
            mask = df_dummy[col] == 1
            df.loc[mask, 'IPW'] = df.loc[mask, col + "_InverseProbabilityWeight"]

        # 結果の表示
        st.write(df)

        # グループ化する変数の選択
        group_column = st.selectbox("グループ化する変数を選択してください", df.columns)

        # グループごとのIPWの合計値とサンプルサイズを計算
        grouped_sum = df.groupby(group_column)['IPW'].sum()
        grouped_size = df.groupby(group_column).size()

        # Assignment weightの計算
        assignment_weight = grouped_size / grouped_sum

        # データフレームにAssignment weight列を追加
        df['Assignment weight'] = df[group_column].map(assignment_weight)

        # Weightbackscoreの計算
        df['Weightbackscore'] = df['IPW'] * df['Assignment weight']

        # 結果の表示
        st.write(df)

        # CSVファイルの出力
        st.write("CSVファイルの出力")
        csv_file = df.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv_file, file_name="result.csv")

if __name__ == '__main__':
    main()

