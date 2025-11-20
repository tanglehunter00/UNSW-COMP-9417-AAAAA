# UNSW-COMP-9417-AAAAA
UNSW COMP9417 GROUP AAAAA FINAL PROJECT

GROUP MEMBER: 
Qianru Lin(eva)
Winston Chen
Tian Huang
Jifei Wang
Linyang Yu

PROJECT SPEC:

COMP9417 Project: Forecasting Air Pollution with Machine Learning 
November 1, 2025 

Project Description 
    Air pollution forecasting is a major environmental and public health challenge. In this project, you will design, implement, and evaluate machine learning models that predict air quality indicators using time-series sensor data. The goal is to model how pollutant concentrations evolve and forecast future pollution levels using historical observations and environmental factors. 
    You will work with theAir Qualitydataset available from the UCI Machine Learning Repository1. This dataset contains9,358 hourly recordscollected by an air quality monitoring device located at road level in a heavily polluted urban area of Italy. The device includes an array offive metal oxide chemical sensorsmeasuring gaseous pollutants, along with meteorological variables such as temperature and humidity. The data spans fromMarch 2004 to February 2005and represents one of the longest publicly available on-field sensor recordings for air quality monitoring. 
    Your task is to build machine learning models that learn temporal dependencies and environmental patterns from this dataset. You will explore both regression and classification approaches to forecasting pollutant levels, assess different feature representations, and evaluate model robustness under realistic deployment scenarios. 

Description of the Data 
    Table 1 summarises the main variables included in the dataset. The features consist of bothsensor responses (PT08.S1--S5) andreference pollutant concentrationsfor carbon monoxide (CO), non-methane hydrocarbons (NMHC), benzene (C 6H6), nitrogen oxides (NO x), and nitrogen dioxide (NO 2). In addition, the dataset includes meteorological attributessuch as temperature (T), relative humidity (RH), and absolute humidity (AH), which capture environmental factors influencing pollutant behaviour. Several variables contain missing values represented by the sentinel value-200, which should be treated as missing data and handled appropriately during preprocessing. Together, these variables form a multivariate time series suitable for time-based predictive modelling. 
        Table 1: Variables in the Air Quality dataset. 
        Variable        Description 
        CO(GT)          True hourly averaged concentration of carbon monoxide 
        PT08.S1(CO)     Hourly averaged response of the CO sensor 
        NMHC(GT)        True hourly averaged concentration of non-methane hydrocarbons 
        C6H6(GT)        True hourly averaged concentration of benzene 
        PT08.S2(NMHC)   Hourly averaged response of the NMHC sensor 
        NOx(GT)         True hourly averaged concentration of nitrogen oxides 
        PT08.S3(NOx)    Hourly averaged response of the NOx sensor 
        NO2(GT)         True hourly averaged concentration of nitrogen dioxide 
        PT08.S4(NO2)    Hourly averaged response of the NO2 sensor 
        PT08.S5(O3)     Hourly averaged response of the O3 sensor 
        T               Ambient temperature (°C) 
        RH              Relative humidity (%) 
        AH              Absolute humidity 
#1 https://archive.ics.uci.edu/dataset/360/air+quality

Main Objectives 
The project focuses on developing a data-driven framework for forecasting pollutant concentrations and analysing temporal dynamics in environmental data. Your work should address the following components: 
    •Exploratory Data Analysis (EDA):Examine time patterns, seasonal effects, correlations among pollutants, and relationships between meteorological and chemical variables. Identify missing values and data quality issues. 
    •Data Preprocessing:Handle missing values (e.g.,-200), mergeDateandTimefields into a unified timestamp, convert decimal separators, and normalise continuous features. Create derived features such as hour, weekday, and month. 
    •Anomaly and Event Detection:Identify and analyse unexpected pollution spikes or sensor faults by ap- plying residual-based anomaly detection and/or unsupervised methods. Compare detected anomalies with me- teorological or calendar features (e.g., temperature extremes, weekends) to interpret likely causes. Evaluate precision–recall trade-offs and assess how anomaly detection can enhance model robustness and data reliability. 
    •Feature Engineering:Design temporal features such as lagged variables and moving averages to capture short-term and long-term dependencies. Investigate how temporal granularity (hourly vs. daily averages) affects prediction accuracy. 
    •Temporal Data Splitting:Since the data are time-oriented, evaluate models using temporal splits. Specifi- cally, implement achronological split(train on 2004 data, test on 2005 data) to simulate model deployment and post-deployment evaluation. If a validation set is needed, use the latest portion of the training data. 
    •Model Assessment:
        1. Use RMSE for regression tasks, accuracy for classification. Visualise residuals and the trends between predicted and observed values over time.
        2. Predict pollutant concentrations for the following horizons: 1 hour, 6 hours, 12 hours, and 24 hours ahead.
        3. Compare results against ana¨ ıve baselinethat uses the concentration at timetas the prediction fort+ 1, t+ 6,t+ 12, andt+ 24. 
    •Regression Model Development:Choose two or more regression algorithms (e.g., Linear Regression, Re- gression Trees, Random Forest, Gradient Boosting, Neural Networks, or Support Vector Regression) to predict concentrations of the five pollutants: CO, NMHC, C6H6, NO x, and NO 2. 
    •Classification Model Development:
        1. For classification tasks, use CO (GT) as the target variable. Discretise it into the following categories: low(<1.5mg/m3), mid (1.5≤CO<2.5), and high (>2.5).
        2. Choose two or more classification algorithms (e.g., Logistic Regression, Decision Trees, Random Forest,Gradient Boosting, Neural Networks, or Support Vector Machines).
        3. Compare results against ana¨ ıve baselinethat uses the discretised concentration at timetas the prediction for t+ 1,t+ 6,t+ 12, andt+ 24. 
    •Discussion:Analyse which features and modelling strategies perform best. Discuss limitations such as sensor drift, data imbalance, and temporal concept drift that may affect real-world deployment. 
    Submission Guidelines 
        •The deadline to submit the report and code is 23 November 2025 (Sunday) at 6 pm. 
        •Work in groups of 4–5 students and register your group on Moodle. 
        •Submit your report (.pdf) and code (.zip) via Moodle — one submission per group. 
        •Late submissions incur a 5% penalty per day. Submissions more than 5 days late will receive a mark of zero.

Report Structure 
Your final report (maximum 6 pages, 12 pt font, 1.5 spacing) should follow this structure:
    1.Introduction:Problem motivation, dataset description, and project goals.
    2.Data Analysis:Summary statistics, data cleaning, and exploration of patterns.
    3.Methodology:Preprocessing, feature engineering, and model selection rationale.
    4.Results:Quantitative evaluation and visualisation of model performance.
    5.Discussion:Interpretation of results, limitations, and improvement strategies.
    6.Conclusion:Summary of findings and implications.
    7.References:Include all sources used for models, algorithms, or background theory.

Peer Review 
All group members must complete the peer review survey by 24 November 2025 (Monday) by 6 pm. Failure to do so will result in a 10% penalty to that student’s mark. Peer evaluations will be used to adjust individual grades based on contribution. 

Project Support 
You may use any open-source libraries such asscikit-learn,pandas,numpy, orxgboost. Consult official documen- tation and lecture materials for guidance. General questions can be posted in the COMP9417 project discussion forum on Moodle.

PROJECT SPEC:中文版

COMP9417 项目：利用机器学习预测空气污染
2025 年 11 月 1 日

项目描述
    空气污染预测是重要的环境与公共健康挑战。本项目需设计、实现并评估能够利用时间序列传感器数据预测空气质量指标的机器学习模型。目标是建模污染物浓度的演化过程，并利用历史观测与环境因素来推断未来污染水平。
    你将使用来自 UCI 机器学习资料库的 Air Quality 数据集。该数据集包含 9,358 条逐小时记录，采集自意大利某污染严重城市道路附近的空气质量监测装置。该装置由五个金属氧化物化学传感器和温度、湿度等气象变量组成。数据跨度为 2004 年 3 月至 2005 年 2 月，是公开可用的最长现场空气质量监测序列之一。
    任务是构建能够学习时间依赖和环境模式的模型，探索回归与分类两种预测方式，比较不同特征表示，并在逼真的部署场景下评估模型稳健性。

数据说明
    表 1 总结了数据集的主要变量。特征由传感器响应（PT08.S1--S5）及多个污染物的参考浓度构成，包括一氧化碳 (CO)、非甲烷碳氢化合物 (NMHC)、苯 (C6H6)、氮氧化物 (NOx) 与二氧化氮 (NO2)。此外还包含温度 (T)、相对湿度 (RH) 与绝对湿度 (AH) 等气象属性，用于反映影响污染行为的环境因素。部分变量以 -200 表示缺失值，需要在预处理中正确处理。这些变量组成可用于时间序列预测的多变量数据集。
        表 1：Air Quality 数据集中的变量
        变量              描述
        CO(GT)            一氧化碳真实小时平均浓度
        PT08.S1(CO)       CO 传感器小时平均响应
        NMHC(GT)          非甲烷碳氢化合物真实小时平均浓度
        C6H6(GT)          苯的真实小时平均浓度
        PT08.S2(NMHC)     NMHC 传感器小时平均响应
        NOx(GT)           氮氧化物真实小时平均浓度
        PT08.S3(NOx)      NOx 传感器小时平均响应
        NO2(GT)           二氧化氮真实小时平均浓度
        PT08.S4(NO2)      NO2 传感器小时平均响应
        PT08.S5(O3)       O3 传感器小时平均响应
        T                 环境温度 (°C)
        RH                相对湿度 (%)
        AH                绝对湿度
#1 https://archive.ics.uci.edu/dataset/360/air+quality

主要目标
项目聚焦于构建基于数据的污染物浓度预测框架，并分析环境数据的时间动态。工作应涵盖以下内容：
    • 探索性数据分析 (EDA)：检视时间模式、季节效应、污染物间相关性，以及气象变量与化学变量之间的关系；识别缺失值与质量问题。
    • 数据预处理：处理缺失值（例如 -200）、合并 Date 与 Time 字段生成统一时间戳、转换小数分隔符、归一化连续特征，构造小时、星期、月份等派生特征。
    • 异常与事件检测：利用残差型或无监督方法识别并分析异常污染峰值或传感器故障；将检测到的异常与气象或日历特征（如极端温度、周末）对应，解释潜在原因；评估查准率-召回率权衡，并论证异常检测如何增强模型稳健性与数据可靠性。
    • 特征工程：设计滞后变量、移动平均等时间特征，捕获短期与长期依赖；研究不同时间粒度（逐小时 vs. 日均值）对预测精度的影响。
    • 时间序列划分：采用时间顺序进行评估，建议以 2004 年数据训练，2005 年数据测试，模拟部署与后评估；若需要验证集，可使用训练数据的最新部分。
    • 模型评估：
        1. 回归任务使用 RMSE，分类任务使用准确率，并可视化残差及预测与观测的时间趋势。
        2. 预测 1 小时、6 小时、12 小时与 24 小时的未来污染物浓度。
        3. 与朴素基线比较：直接使用时刻 t 的浓度作为 t+1、t+6、t+12、t+24 的预测值。
    • 回归模型开发：选择两个或以上的回归算法（如线性回归、回归树、随机森林、梯度提升、神经网络或支持向量回归）来预测 CO、NMHC、C6H6、NOx、NO2 的浓度。
    • 分类模型开发：
        1. 以 CO(GT) 为目标变量，将其离散化为低 (<1.5 mg/m3)、中 (1.5≤CO<2.5)、高 (>2.5) 三类。
        2. 选择两个或以上的分类算法（如逻辑回归、决策树、随机森林、梯度提升、神经网络或支持向量机）。
        3. 与朴素基线比较：使用时刻 t 的离散化浓度预测 t+1、t+6、t+12、t+24。
    • 讨论：分析表现最佳的特征与建模策略，讨论传感器漂移、数据不平衡、时间概念漂移等现实部署中可能的局限。

提交指南
        • 报告与代码提交截止时间为 2025 年 11 月 23 日（周日）18:00。
        • 以 4-5 人小组形式完成，并在 Moodle 上登记。
        • 通过 Moodle 以团队形式提交报告（.pdf）与代码（.zip），仅需一次提交。
        • 迟交每天扣减 5%，超过 5 天记 0 分。

报告结构
最终报告（最多 6 页、12 号字体、1.5 倍行距）应遵循以下结构：
    1. 引言：问题动机、数据集描述、项目目标。
    2. 数据分析：统计摘要、数据清洗、模式探索。
    3. 方法：预处理、特征工程、模型选择理由。
    4. 实验结果：量化评估与可视化。
    5. 讨论：结果解读、局限与改进策略。
    6. 结论：发现总结与意义。
    7. 参考文献：列出使用的模型、算法或背景资料来源。

同伴互评
所有组员需在 2025 年 11 月 24 日（周一）18:00 之前完成同伴互评，未完成者成绩扣 10%。同伴评价将根据贡献度调整个人分数。

项目支持
可使用 scikit-learn、pandas、numpy、xgboost 等开源库。请参考官方文档与课程资料；一般问题可在 Moodle 的 COMP9417 项目讨论区提问。