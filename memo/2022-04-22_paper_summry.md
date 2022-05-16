## 論文探す

キーワード
- remote communication
- virtual meetings

- Improved Gazing Transition Patterns for Predicting Turn-Taking in Multiparty Conversation
  - 手法のキモ（キーコンセプト、コアバリュー）
    - 最初の2-gramモデル, 4人の視線モデル
  - 実験方法
    - 4人の対面対話
  - 評価指標
    - ishii, Liらと比較、
  - 結果 (=contribution)
    - ishii, Liよりも精度高かった
    - Turn-taking: 0.58

- Situation exchange system using nonverbal information for remote communication
  - pdfなし
  - 遠隔コミュニケーションのために，利用者の状況情報を伝達し，非言語情報を含むメッセージを交換するシステム

- On the Sound of Successful Meetings: How Speech Prosody Predicts Meeting Performance
  - 概要
    - 会議の成功が、全体的な音響-プロソディーの測定によってどの程度予測できるかを調査
  - 手法のキモ（キーコンセプト、コアバリュー）
    - 会議の成功
  - 実験方法
    - 対面
    - 3~6人の70のグループが大学の交通状況について議論
  - 評価指標
    - 提言の数, 実現可能性、質、およびグループメンバー間の平均有効性と満足度評価
  - 結果 (=contribution)
    - 個々の会議の全体的な「音」とかなり相関しており、ピッチ特徴が最も多様で強力な予測因子
    - 韻律的な特徴パターンから、効果的な会議は短く、淡々としているのに対し、生産的な会議は長く、生き生きとした発話旋律を持つ

- Online Error Detection of Barge-In Utterances by Using Individual Users’ Utterance Histories in Spoken Dialogue System
  -  ASR信頼度指標と発話履歴
  - 結果
    - 発話履歴を用いた場合、誤認識率は15%減少するこ

- Can Prediction of Turn-management Willingness Improve Turn-changing Modeling?
  - 概要
    - 会話からの音響的、言語的、視覚的を用いてターンの管理（話す，聞く）意欲を予測する手法の提案
  - 手法のキモ
    - 話者／リスナーのターン操作の意志のスコアを注釈したダイアド会話コーパスを導入??
    - マルチタスク学習アプローチの活用
  - Research Question
    - Q1. 話し手と聞き手の言語行動と非言語行動は、ターンマネジメントの意欲を予測するのに有用か？
    - Q2. 意思を明示的にモデル化することは、ターンチェンジ予測に有用か？
  - 前処理
    - 会話コーパスは初対面における対面対話，24名
    - 話し手と聞き手の4種類のターンマネジメントの意欲をスコア化
        - ターンキープの意志（別名：話し手の発言意志。話し手に順番を守る（話し続ける）意志があるか？
        - ターンを譲る意志（話し手の聞く意志。話し手に順番を譲る（聞き手が話すのを聞く）意志があるか？
        - ターンを奪う意欲（別名：聞き手の発言意欲）。聞き手に順番を奪う（話し始める）意志があるか？
        - 聴く意欲（聴き手の聞く意欲）。聞き手は、話し手の話を聞き続けようとする意志を持っているか？
    - アノテータ10名の級内相関係数(ICC)の指標で発話意欲の一致度を計る
  - 結果
  - メモ
    - 論文より）その結果、VGGish [14], BERT [7], ResNet-50 [13] などの最新の高次抽象化特徴が、MFCC [9], LIWC [29], アクションユニット [1] などの解釈可能特徴よりも、ダイアド間の相互作用における自己開示発話の推定に有用であることが実証された。
    - [49]の特徴量から使える
    - 視覚情報はResNet-50（ニューラルネットワーク）

- 改善版
  - バックチャネル（uum, yuhなど）の追加

やりたいのは，個人の内部状態（predicting the internal state of an individual）
発話の欲求に着目

- Parasocial Consensus Sampling: Combining Multiple Perspectives to Learn Virtual Human Behavior
  - 概要
    - 対面状況で，個人の反応を観察することで，
    - 複数の個人が同じ状況を代替的に体験することで、社会的相互作用における人間の反応の典型（＝コンセンサスビュー）についての洞察を得る
    - パラソシャルコンセンサスサンプリング（PCS）の紹介
  - アプローチ利点
    (1) 同じ話者と対話する複数の独立した聞き手のデータを統合することができる。
    (2) 時間の経過とともにフィードバックがなされる確率を関連付けることができる。
    (3) 対面インタラクションデータを分析・理解するための事前準備として利用できる。
    (4）より迅速で安価なデータ収集が可能になる。本論文では、我々のPCSアプローチを適用し、聞き手のバックチャネルフィードバックの予測モデルを学習する。
  - メモ
    - コンセンサス（一致）

- How People Distinguish Individuals from their Movements: Toward the Realization of Personalized Agents
  - 概要
    - 人間が短い動作からどの程度個人を識別できるのか、また、動作のどのような要素が個性の認識に寄与しているのかを明らかにするために、個人識別の確信度とジェスチャ動作の静的特性との関係を検討
  - 問題提起
    - 特定の個人に属すると識別可能な個性の側面を作り出す方法は、これまで研究されていない。
  - 結果(=contribution)
    - 同じ動作をしている個人を見分けられる
    - 手首を早く動かしているか，どれだけ速く動かすかなどの空間的・時間的広がりを手がかりに