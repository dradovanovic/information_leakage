# Socio-demographic / household attribute inference from smart meter load profiles (focus: Irish CER / ISSDA CBT)

Notes from a research agent, 2026-11-20. I verified full text (via publisher or author PDF) for Beckel et al. 2014, Wang et al. 2021 (federated, TSG), Zhang et al. 2022 (Frontiers), Wang Y. et al. 2023 (ICAID/Atlantis) and Radovanovic et al. 2025 (SMARTGREENS). For Wang et al. 2019 (TSG), Hopf et al., Viegas et al., Zhong & Tam and the 2025 NAS paper I saw only metadata and abstracts (paywall or HTTP 403). Statements about those papers are marked UNVERIFIED where relevant.

## Q1. Seminal and key works, and which ones use CER

### Takeaway
The canonical CER line runs Beckel et al. 2013 (CLASS) -> Beckel et al. 2014 (Energy; handcrafted features + SVM/LDA etc.) -> Hopf et al. 2014/2016 (feature extraction/selection) -> Wang et al. 2019 (TSG; CNN + SVM on weekly load) -> privacy-preserving and federated variants (Wang et al. 2021 TSG, Zhang et al. 2022) -> small CNN, CNN-LSTM and NAS/attention-FCN papers (2022-2025). I found no peer-reviewed Transformer or graph model for CER attribute inference specifically. Transformers do appear in CER-adjacent work on re-identification (Cretu et al. 2024).

### Cited Findings
- **Beckel, Sadamori, Staake, Santini (2014)**, "Revealing household characteristics from smart meter data," *Energy* 78:397-410, DOI 10.1016/j.energy.2014.10.025. **Uses CER**: 4,232 Irish households, 30-min resolution, July 2009 to December 2010 (75 weeks), pre- and post-trial questionnaires. — [EconPapers](https://econpapers.repec.org/RePEc:eee:energy:v:78:y:2014:i:c:p:397-410); [author PDF (ETH)](https://vs.inf.ethz.ch/publ/papers/beckel-2014-energy.pdf)
- **Beckel et al. 2013 (CLASS)**: Sadamori/Beckel/Santini, "Automatic socio-economic classification of households using electricity consumption data," e-Energy 2013 ("Future Energy" pp. 75-86 in a later citation). **Uses CER.** Later papers report CLASS accuracy above 70%. Exact venue/DOI not verified here. — [ResearchGate listing](https://www.researchgate.net/publication/262253537_Automatic_socio-economic_classification_of_households_using_electricity_consumption_data); [Frontiers 2022 citing it](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.992117/full); [ICAID 2023 reference list](https://www.atlantis-press.com/article/125977175.pdf)
- **Hopf, Sodenkamp, Kozlovskiy, Staake**, "Feature extraction and filtering for household classification based on smart electricity meter data," *Computer Science - Research and Development* 31(3):141-148 (online 2014, issue 2016), DOI 10.1007/s00450-014-0294-4. UNVERIFIED that it uses CER. Radovanovic et al. 2025 say Hopf et al. 2016 raised accuracy to about 80% "using the same dataset" (i.e., CER). — [Springer](https://link.springer.com/article/10.1007/s00450-014-0294-4); [Radovanovic et al. 2025](https://www.scitepress.org/Papers/2025/132174/132174.pdf)
- **Hopf, Sodenkamp, Staake (2018)**, *Electronic Markets*, DOI 10.1007/s12525-018-0290-9: 133 features from smart meter and weather data, Random Forest, 19 classes for 11 characteristics. **NOT CER**: 527 Swiss households. — [Springer](https://link.springer.com/10.1007/s12525-018-0290-9)
- **Wang Y., Chen Q., Gan D., Yang J., Kirschen D.S., Kang C. (2019)**, "Deep learning-based socio-demographic information identification from smart meter data," *IEEE Trans. Smart Grid* 10(3):2593-2602, DOI 10.1109/TSG.2018.2805723 (online Feb 2018). **Uses CER.** A deep CNN extracts features from load profiles and an SVM classifies them. — [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/8291011)
- **Wang Y., Bennani I.L., Liu X., Sun M., Zhou Y. (2021)**, "Electricity consumer characteristics identification: A federated learning approach," *IEEE Trans. Smart Grid* 12(4):3637-3647, DOI 10.1109/TSG.2021.3066577. **Uses CER** (4,232 residential consumers, 536 days). — [DTU Orbit PDF](https://backend.orbit.dtu.dk/ws/files/256582645/Electricity_Consumer_Characteristics_Identification_A_Federated_Learning_Approach_12_.pdf)
- **Zhang, Lu, Xu, Xu, Kong, Yu (2022)**, "Privacy-preserving deep learning for electricity consumer characteristics identification," *Frontiers in Energy Research* 10, DOI 10.3389/fenrg.2022.992117. **Uses CER**: 7x48 weekly input, secure multi-party computation (MPC) via MP-SPDZ. — [Frontiers](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.992117/full)
- **Wang Yibo, Wang Qian, Wu Z., Zhu B. (2023)**, "Socio-demographic information extraction from load profile using convolutional neural network," Proc. ICAID 2022, AHIS 7, pp. 703-715, Atlantis Press, DOI 10.2991/978-94-6463-010-7_72. **Uses CER**: 1D vs 2D CNN on 7x48 input. — [Atlantis PDF](https://www.atlantis-press.com/article/125977175.pdf)
- **Xu R. et al. (EMIE 2022)**, "A deep learning method for household characteristic classification from smart meter data," VDE proceedings. CNN-LSTM. **Uses CER** (abstract). Some papers cite a "Lin et al. 2021" CNN+LSTM work, which may be a citation mix-up with this one (UNVERIFIED). — [VDE Verlag](https://vde-verlag.de/proceedings-en/565961174.html)
- **NAS channel-attention FCN (2025)**, "Designing channel attention fully convolutional networks with neural architecture search for customer socio-demographic information identification using smart meter data," *AI* (MDPI) 6(1):9, DOI 10.3390/ai6010009. Targets the CER task according to search snippets. Authors, architecture and numbers UNVERIFIED (MDPI returned 403). — [DOI](https://doi.org/10.3390/ai6010009)
- **Viegas, Vieira, Melício, Mendes, Sousa (2016)**, "Classification of new electricity customers based on surveys and smart metering data," *Energy* 107:804-817. Uses smart meter data from more than 5,000 Irish customers (**CER**). Combines metering and survey/static data. Age, employment and appliances are reported as key. A secondary source reports 50-80% accuracy. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0360544216304789)
- **Zhong & Tam (2015)**, "Hierarchical classification of load profiles based on their characteristic attributes in frequency domain," *IEEE Trans. Power Systems* 30(5):2434-2441. Frequency-domain attributes (CAFD) and CART for load-profile classes. This is load-class hierarchy, not socio-demographic inference. CER use NOT confirmed. — [Semantic Scholar figure page](https://www.semanticscholar.org/paper/A-Frequency-Domain-Approach-to-Characterize-and-Zhong-Tam/8d9adf173f15f281aec94f982391963ed37a2c85/figure/5)
- **Radovanovic, Schirl, Unterweger, Eibl (2025)**, "Predicting socio-demographic characteristics from load profiles with varying time granularities," SMARTGREENS 2025, pp. 87-98, DOI 10.5220/0013217400003953. **NOT CER**: PEAK Load Data, 1,589 suburban Upper-Austrian households, 15-min data, Sept 2017 to Oct 2018. Uses Beckel features, XGBoost/SVM/MLP, MCC. — [SciTePress PDF](https://www.scitepress.org/Papers/2025/132174/132174.pdf)
- **Radovanovic, Schirl, Unterweger, Eibl (2026)**, "The impact of temporal granularity on socio-demographic inference from household load profiles," arXiv:2606.03358. Eight attributes, 1,589 households (1,208 after filtering, per a search snippet), handcrafted vs tsfresh vs CNN-autoencoder embeddings, XGBoost best. Same Austrian dataset, so **NOT CER**. — [arXiv](https://arxiv.org/abs/2606.03358)
- **Cretu, Rusu, de Montjoye (2024)**, "Re-pseudonymization strategies for smart meter data are not robust to deep learning profiling attacks," ACM CODASPY '24 (extended version arXiv:2404.03948). Tests six architectures on weekly records; Transformer and CNN-LSTM do best. Re-identification of 5,139 households (73.4% top-1 with electricity + gas, 54.5% with electricity only). Attribute inference not studied. The abstract does not name the dataset. — [arXiv](https://arxiv.org/abs/2404.03948)

### Inferences
- In this literature, "Wang et al." can mean three different groups: Yi Wang/Kang (TSG 2019, TSG 2021), Yibo Wang (ICAID 2023), and Zhang et al. (2022), who reuse the Wang setup. Cite carefully.
- The deep-learning work on CER is dominated by small 2-layer CNNs on 7x48 weekly matrices. I found no peer-reviewed CER attribute-inference papers with Transformer, GNN, or explicit multi-task/explainability designs, but my search was limited (see Gaps).

### Gaps
- I could not access the full text of Wang et al. 2019 (IEEE, ResearchGate and HKU all returned 403). Exact layer configuration, number of customers (a search snippet says 920 Irish customers, UNVERIFIED) and per-question accuracies are not confirmed here.
- Hopf et al. 2014/2016: dataset and results not verified.
- "Self-supervised CNN for user behavior analysis on smart meter data" (*Journal of Measurements in Engineering* 13(3), 2025) surfaced in search. I did not check its content or whether it uses CER. — [Extrica](https://www.extrica.com/article/24825)
- A Transformer, graph or multi-task CER attribute paper may exist in 2024-2026 venues not covered by my roughly 15-query budget.

## Q2. Architectures, input length/shape, resolution, features vs raw input

### Takeaway
There are two families. (a) Handcrafted features computed on one week of 30-min data: Beckel uses 22 features plus 10 PCA components, with SFFS selection capped at 8 features; Wang 2021 uses 73 features plus PCA and an ANN. (b) Raw 7x48 weekly matrices fed to very small CNNs with 2 conv layers, 8-16 kernels of size 2x3 or 3x3, one max-pool layer, and a dense/softmax or SVM head.

### Cited Findings
- **Beckel 2014 features**: five groups, each computed on one week of 30-min data: consumption figures (10), ratios (7), temporal dynamics (4), statistical properties (3), and the first 10 principal components. Most features use weekdays only. Each feature is log- or sqrt-transformed and then z-normalised. Classifiers are kNN, LDA, Mahalanobis, SVM (LIBSVM) and AdaBoost. Feature selection uses SFFS with at most 8 features, chosen per fold. — [Beckel 2014 PDF](https://vs.inf.ethz.ch/publ/papers/beckel-2014-energy.pdf)
- **Zhang et al. 2022 (CER, 7x48)**: binary tasks use "Network A", an MLP FC128-FC128-FC1 with sigmoid. Multi-class tasks use "Network B": Conv 16@3x3 -> Conv 16@3x3 -> MaxPool 2x2 -> FC32 -> softmax. Batch size 128, learning rate 0.01, 10 epochs, 16-bit fixed point under MPC. Weeks with missing data were dropped and weeks aligned to start on Friday, for 20,000 weekly samples in total. — [Frontiers](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.992117/full)
- **Wang Yibo et al. 2023 (CER, 48x7 input)**:
  - 1D model: Conv1D (kernel 3, 8 filters) -> Conv1D (16) -> MaxPool1D -> Dropout -> Dense. This treats a day as the sequence and the 7 days as channels.
  - 2D model: Conv2D 8@2x3 -> Conv2D 16 -> MaxPool 3x3 -> Dropout 0.5 -> Flatten (704) -> Dense softmax.
  - About 2.3k and 2.6k parameters. Cross-entropy loss, ReLU. "Data enhancement" means using every week as a separate sample: r weeks give r times the samples.
  - The authors justify one week as the input because it captures the weekday/weekend cycle and longer windows add little.
  — [Atlantis PDF](https://www.atlantis-press.com/article/125977175.pdf)
- **Wang et al. 2019 (TSG)**: a deep CNN extracts features from load profiles and an SVM classifies them. The 2025 NAS paper says Wang et al. studied 1D and 2D convolutions. Exact layers UNVERIFIED. — [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/8291011); [AI 2025 snippet](https://doi.org/10.3390/ai6010009)
- **Wang et al. 2021 (federated, CER)**: 73 handcrafted consumption features, privacy-preserving PCA, an ANN with one hidden layer and dropout, trained with Adam. Baselines are KNN, RF and XGBoost. — [DTU Orbit PDF](https://backend.orbit.dtu.dk/ws/files/256582645/Electricity_Consumer_Characteristics_Identification_A_Federated_Learning_Approach_12_.pdf)
- **Xu et al. (EMIE 2022)**: CNN-LSTM on CER. Details UNVERIFIED. — [VDE](https://vde-verlag.de/proceedings-en/565961174.html)
- **Granularity**: on Austrian data (not CER), 15-min to 1-h resolution gives nearly the same performance. Performance drops with coarser data, with a second plateau between 1 day and 7 days. Dwelling size can be inferred even from coarse data, while pool use needs fine data. — [Radovanovic et al. 2025](https://www.scitepress.org/Papers/2025/132174/132174.pdf); [arXiv:2606.03358](https://arxiv.org/abs/2606.03358)

### Inferences
- A one-week 7x48 matrix at 30 min is the de facto standard input for CER deep learning. Beckel also uses one week (week 26) for features. Published CER CNNs are tiny, with roughly 2-3k parameters in ICAID 2023. This is a reasonable "Wang-style" baseline to reproduce in PyTorch.

### Gaps
- Exact Wang 2019 CNN layers and SVM settings, and the NAS-FCN 2025 search space and final architecture, were not verified.

## Q3. Class imbalance handling and label definitions from the CER survey

### Takeaway
Beckel 2014 defines the reference label set: single = 1 adult and 0 children, family = more than 1 adult and at least 1 child, etc. It handles imbalance by random undersampling of training data, reports MCC next to accuracy, and compares against random-guess (RG) and biased-random-guess (BRG) baselines. Later DL papers mostly report accuracy only (Zhang 2022, ICAID 2023) or accuracy + MCC (Wang 2021). Explicit resampling is rarely described in the DL papers.

### Cited Findings
- **Beckel 2014 Table 2** (household counts in CER; age bounds as extracted from the PDF):
  - Single: 1 adult and 0 children, 859 vs 3,373.
  - Family: more than 1 adult and more than 0 children, 1,118 vs 3,114.
  - Children: at least 1 child, 1,229 vs 3,003.
  - #residents: few (2 or fewer) 2,199 vs many (3 or more) 2,033.
  - Age of house: old (more than 30 years) 2,151 vs new (30 or fewer) 2,077.
  - Employment of chief income earner (CIE): employed 2,536 vs not employed 1,696.
  - All adults employed: 1,013 vs 2,409.
  - Retirement of CIE: 1,285 vs 2,947.
  - Social class (NRS): AB 642, C1/C2 1,840, DE 1,593.
  - Age of CIE: <35 436, 35-65 2,819, >65 953.
  - Income: under 50k 940 vs 50k or more 997.
  - House type: detached/bungalow 2,189 vs semi-detached/terraced 1,964.
  - Floor area: 3 classes. #bedrooms: 4 classes. Cooking: electric 2,960 vs not 1,272.
  - Unoccupied more than 6 h/day: 885 vs 3,347.
  - Lightbulbs, #appliances.
  - Multi-class thresholds were chosen so that classes are roughly equal in size where possible.
  — [Beckel 2014 PDF](https://vs.inf.ethz.ch/publ/papers/beckel-2014-energy.pdf)
- **Beckel 2014 imbalance handling**: random undersampling to the size of the smallest class, optional. The accuracy runs use no undersampling and accuracy as the selection criterion. The MCC/ROC runs use undersampling and MCC as the criterion. Baselines are RG (1/K) and BRG (sum of squared class proportions). — [Beckel 2014 PDF](https://vs.inf.ethz.ch/publ/papers/beckel-2014-energy.pdf)
- **Wang 2021 labels** reuse the Beckel-style classes for 15 characteristics, including "live alone", #residents, age of house, and rent/own. The authors acknowledge strong imbalance and recommend MCC over accuracy, but use no resampling in training. — [DTU PDF](https://backend.orbit.dtu.dk/ws/files/256582645/Electricity_Consumer_Characteristics_Identification_A_Federated_Learning_Approach_12_.pdf)
- **Zhang 2022 labels** are by CER question number: #300 age of CIE (3 classes), #310 CIE retired, #401 social class (AB/C1C2/DE), #410 children, #450 house type, #453 house age (3 classes), #460 bedrooms (3 classes), #4704 cooking, #4905 energy-efficient bulbs. No imbalance handling is described. — [Frontiers](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.992117/full)
- **ICAID 2023** uses 13 pre-trial survey questions (200 sex, 300, 310, 401, 405 internet, 406 broadband, 410 household composition, 430 hours at home, 450, 4531 house age, 460, 4704, 4906 double glazing). Counts are per week-sample. Metrics are accuracy and micro-F1, with no resampling. — [Atlantis PDF](https://www.atlantis-press.com/article/125977175.pdf)
- **Radovanovic et al. 2025** (non-CER) undersample only in training and validation, leave the test set unaffected, and use MCC as the key metric, following Beckel. — [SciTePress PDF](https://www.scitepress.org/Papers/2025/132174/132174.pdf)

### Inferences
- For comparability with Beckel and Wang 2021, define single = 1 adult and 0 children (CER Q410 "live alone") and report MCC plus balanced accuracy against a BRG/majority baseline.
- Accuracy-only results on imbalanced CER labels are hard to interpret. Examples: type of home at about 93% (Wang 2021), and ICAID's >75% for cooking and double glazing, where the class ratios are about 3:1 and 15:1.

### Gaps
- I found no CER DL paper that uses class-weighted loss, focal loss or SMOTE, or that reports AUC. I could not verify whether Wang 2019 handled imbalance.

## Q4. Household-level vs week-level splits, leakage, aggregation

### Takeaway
The protocols differ and this matters. Beckel 2014 has one sample per household (one week), so the folds are household-disjoint, and it aggregates per household by majority vote over weeks. ICAID 2023 and Wang 2021 split by consumer ID (80/20). Zhang 2022 splits pooled weekly samples (17k/3k weeks) without stating household disjointness, which risks the same household appearing in train and test.

### Cited Findings
- **Beckel 2014**:
  - Single-week analysis (week 26: no holidays, includes households that joined in early 2010) with 4-fold CV over disjoint household subsets.
  - Results are stable across all 75 weeks (average accuracy std 0.017). Weeks 50-57 (summer holidays) are worse for occupancy- and size-related attributes.
  - Multi-week: a classifier is trained per week, and each test household gets the majority vote of its weekly predictions (only households with more than 50 weeks of data). This adds about 3 pp accuracy and 6 pp MCC on average, up to 10 pp.
  - The authors leave training on one week and testing on different weeks and households to future work.
  — [Beckel 2014 PDF](https://vs.inf.ethz.ch/publ/papers/beckel-2014-energy.pdf)
- **ICAID 2023**: from 929 demand-response households, 147 with fewer than 536 days were removed, leaving 782. Households are sorted by ID; the first 80% are train and the last 20% test, with 76 weeks per household as separate samples. This is a household-disjoint split with week-level samples. No per-household aggregation is reported. — [Atlantis PDF](https://www.atlantis-press.com/article/125977175.pdf)
- **Wang 2021**: the first 80% of the dataset (consumers) is used for training, split across 5 simulated retailers, and the remaining 20% for testing. The samples are consumers with features computed over their consumption. — [DTU PDF](https://backend.orbit.dtu.dk/ws/files/256582645/Electricity_Consumer_Characteristics_Identification_A_Federated_Learning_Approach_12_.pdf)
- **Zhang 2022**: 20,000 weekly samples, 17,000 for training and the rest for testing. Household disjointness is not mentioned. — [Frontiers](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.992117/full)
- **Radovanovic et al. 2025** criticise single-known-week protocols. They train on all 52 weekly snippets and test on a single arbitrary week, which they find harder. Their interpretable algorithm beats per-week classifiers and approaches whole-year approaches. — [SciTePress PDF](https://www.scitepress.org/Papers/2025/132174/132174.pdf)
- **Re-identification evidence**: weekly load records are highly household-specific. Deep embeddings re-identify the correct household among 5,139 in 54.5-73.4% of cases. A week-level random split therefore lets a model match a test week to a training week of the same household. — [Cretu et al. 2024](https://arxiv.org/abs/2404.03948)

### Inferences
- Week-level random splits, such as the likely protocol in Zhang 2022, probably inflate accuracy through household memorisation. Cretu et al.'s re-identification results show that weekly profiles act as fingerprints. A PhD project should split by household (GroupKFold) and report both week-level and household-aggregated (vote or mean-probability) metrics.

### Gaps
- Wang 2019's split protocol (household vs week) could not be verified. This is important because it is the most-cited DL baseline.
- I found no published CER study that directly quantifies the gap between household-disjoint and week-random splits.

## Q5. Reported accuracies / F1 per attribute on CER, and baselines

### Takeaway
On CER, the best-predicted attributes are household size and occupancy: single about 80-84% accuracy (MCC 0.43-0.50), #residents MCC about 0.45-0.51, retirement and employment about 70-77%. Dwelling attributes are weak: age of house about 60-64% (MCC about 0.18-0.28), floor area and bedrooms near chance. Social class and income are also weak. Gains over the biased random guess average about 14 pp (Beckel).

### Cited Findings
- **Beckel 2014, Table 3** (LDA; accuracy for week 26 / majority vote over the trial; MCC for week 26 / vote):

  | Characteristic | Acc wk26 | Acc vote | MCC wk26 | MCC vote |
  |---|---|---|---|---|
  | single | 80% | 82% | 0.43 | 0.50 |
  | all employed | 79% | 79% | 0.30 | 0.32 |
  | unoccupied | 76% | 76% | 0.29 | 0.38 |
  | family | 69% | 74% | 0.32 | 0.42 |
  | children | 67% | 73% | 0.31 | 0.41 |
  | cooking | 69% | 71% | 0.22 | 0.29 |
  | retirement | 69% | 73% | 0.35 | 0.43 |
  | #residents | 72% | 76% | 0.45 | 0.51 |
  | employment | 67% | 72% | 0.34 | 0.44 |
  | floor area | 45% | 50% | 0.15 | 0.21 |
  | age person | 55% | 59% | 0.21 | 0.30 |
  | age house | 60% | 64% | 0.19 | 0.28 |
  | house type | 59% | 59% | 0.20 | 0.19 |
  | income | 57% | 61% | 0.16 | 0.23 |
  | lightbulbs | 52% | 55% | 0.062 | 0.098 |
  | social class | 43% | 53% | 0.17 | 0.22 |
  | #appliances | 53% | 56% | 0.29 | 0.34 |
  | #bedrooms | 35% | 39% | 0.15 | 0.15 |
  | **Mean** | **62%** | **65%** | **0.26** | **0.32** |

  — [Beckel 2014 PDF](https://vs.inf.ethz.ch/publ/papers/beckel-2014-energy.pdf)
- **Beckel 2014, best classifier vs baselines**: the best classifier's accuracy exceeds RG by 4.0-33.8 pp (mean 22.0) and BRG by 4.0-22.3 pp (mean 14.2). It exceeds 80% for single, all employed and unoccupied. SVM is best for 13 of 18 characteristics. — [Beckel 2014 PDF](https://vs.inf.ethz.ch/publ/papers/beckel-2014-energy.pdf)
- **Wang et al. 2021 (CER, ANN + PCA, 15 characteristics)**: centralized ANN with PCA averages 68.53% accuracy and MCC 0.2989. "Live alone" reaches about 84% accuracy and #residents MCC 0.48. Employment, retirement, bedrooms, home type, entertainment appliances and cooking have MCC about 0.25-0.45. Age of house has MCC about 0.18 and rent/own about 0.11. The ANN matches XGBoost on accuracy but has higher MCC. KNN, RF and XGBoost baselines are reported. — [DTU PDF](https://backend.orbit.dtu.dk/ws/files/256582645/Electricity_Consumer_Characteristics_Identification_A_Federated_Learning_Approach_12_.pdf)
- **Zhang et al. 2022 (CER, 7x48, MPC-trained; week-level samples)**: #300 70.09%, #310 71.83%, #401 57.33%, #410 74.67%, #450 62.40%, #453 66.53%, #460 54.43%, #4704 66.87%, #4905 63.40%. Accuracy only. — [Frontiers](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.992117/full)
- **ICAID 2023 (CER, 1D/2D CNN)**: accuracy above 75% for broadband, cooking and double glazing. Household composition (Q410) and social class (Q401) are 50-60% with 1D convolution. 2D convolution improves 12 of 13 questions, by an average of 3.22% on the weaker ones. Per-question numbers are in figures only. — [Atlantis PDF](https://www.atlantis-press.com/article/125977175.pdf)
- **Viegas et al. 2016**: 50-80% (secondary report). **CLASS 2013**: more than 70%. **Hopf 2016**: about 80% (secondary). — [Radovanovic et al. 2025 related work](https://www.scitepress.org/Papers/2025/132174/132174.pdf); [Frontiers 2022](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.992117/full)

### Inferences
- The DL papers I verified do not clearly beat Beckel's handcrafted features + LDA/SVM on comparable attributes. For example, Zhang 2022 gets children 74.7% vs Beckel's vote result of 73%, and age of house 66.5% (3 classes) vs Beckel's 64% (2 classes). Different splits and label definitions make direct comparison unreliable.

### Gaps
- I could not extract Wang 2019's per-question accuracies. None of the verified DL papers reports a majority-class or consumption-level-only baseline.

## Q6. Do simple features (consumption level) explain most of the predictability?

### Takeaway
There is qualitative support: Beckel shows that single vs non-single is well separated by total weekly consumption alone, and that size and occupancy attributes, which drive consumption level, are the most predictable. However, I found no CER paper that formally ablates "mean consumption only" against full models.

### Cited Findings
- Beckel 2014 Fig. 3: the empirical CDFs of c_total (total consumption) differ strongly between Single and No single, so "single is well-separable" on that feature alone. A timing ratio (r_noon/day) separates the classes less well. — [Beckel 2014 PDF](https://vs.inf.ethz.ch/publ/papers/beckel-2014-energy.pdf)
- Beckel 2014 discussion: three groups are inferred well: occupancy (employment, unoccupied), number of persons (single, #residents, family) and appliance stock. Dwelling attributes (floor area, bedrooms) are hard because CER homes heat with oil or gas, so heating is absent from electricity data. Income is hard as well. — [Beckel 2014 PDF](https://vs.inf.ethz.ch/publ/papers/beckel-2014-energy.pdf)
- SFFS is limited to 8 features because more features often overfit, so a small feature subset already captures most of the signal. — [Beckel 2014 PDF](https://vs.inf.ethz.ch/publ/papers/beckel-2014-energy.pdf)
- Coarse data (daily to weekly totals, essentially consumption level) still predicts dwelling size in Austrian data. Performance plateaus from 1 day to 7 days and from 15 min to 1 h. — [arXiv:2606.03358](https://arxiv.org/abs/2606.03358); [Radovanovic et al. 2025](https://www.scitepress.org/Papers/2025/132174/132174.pdf)
- Beckel's regression of continuous attributes on consumption features gives low R² (about 0.08-0.30), e.g. RMSE 1.2 with R² 0.17 for one attribute. Most variance remains unexplained even though classification accuracy looks good. — [Beckel 2014 PDF](https://vs.inf.ethz.ch/publ/papers/beckel-2014-energy.pdf)

### Inferences
- The best-predicted CER attributes (single, #residents, family, children) are the ones most tied to total consumption. A "mean weekly kWh + logistic regression" baseline would likely recover a large share of reported performance, and DL papers should be benchmarked against it. This is a hypothesis, not an established result.

### Gaps
- I found no paper that reports a consumption-level-only baseline for CER attribute inference, or that decomposes predictability into level and shape (e.g., by training on load profiles normalised per household). This looks like an open gap the PhD project could fill.
- ISSDA dataset documentation (question numbering, pre-/post-trial survey files) was not fetched in this session. Question numbers above come from Zhang 2022 and ICAID 2023.
