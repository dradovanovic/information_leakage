# Subliminal learning and information leakage through knowledge distillation and shared initialisation

Scope: literature relevant to a PhD project testing whether a sensitive household attribute learned by a teacher leaks to a student that shares the teacher's initialisation and is distilled only on auxiliary logits on noise inputs (the Cloud et al. MNIST setup transferred to smart meter load data). Research date: 2026-11-20.

**Verification legend.**
- **[V]**: verified this session against the arXiv abstract or HTML page, or a publisher or press page.
- **[S]**: seen only in search-result snippets or secondary summaries; details unconfirmed.
- **[K]**: standard reference given from background knowledge. The venue, year and arXiv ID are believed correct but were not re-fetched this session; spot-check them before citing.

---

## Q1. Cloud et al. 2025/2026, "Subliminal Learning": claims, theorem, MNIST experiment, conditions, limitations

### Takeaway
A teacher's trait transfers to a student through data unrelated to the trait. The paper's theorem shows that one small step of distillation on any teacher outputs moves a student that shares the teacher's initialisation toward the teacher's own objective, or at worst leaves it orthogonal. In the MNIST version, a student distilled only on 3 auxiliary logits from noise images gets over 50% test accuracy on digits, but only when it shares the teacher's initialisation. This is almost exactly the setup the project transfers to smart meter data.

### Cited Findings
**Bibliographic details**
- arXiv preprint: Cloud, Le, Chua, Betley, Sztyber-Betley, Hilton, Marks, Evans. "Subliminal Learning: Language models transmit behavioral traits via hidden signals in data." arXiv:2507.14805, submitted 20 Jul 2025, cs.LG. The abstract page lists no venue in the comments field. [V] — [arXiv](https://arxiv.org/abs/2507.14805)
- Peer-reviewed version: "Language models transmit behavioural traits through hidden signals in data", *Nature* 652, 615–621 (2026), published 15 Apr 2026, DOI 10.1038/s41586-026-10319-8. I could not read the full text (paywall redirect). The citation details come from search snippets and press coverage. [S] — [Nature](https://www.nature.com/articles/s41586-026-10319-8); [Nature Asia press release](https://www.natureasia.com/en/info/press-releases/detail/9302); [The Register, 15 Apr 2026](https://theregister.com/2026/04/15/llms_inherit_bad_traits)
- Companion blog post on the Anthropic Alignment Science Blog. [V, existence] — [alignment.anthropic.com](https://alignment.anthropic.com/2025/subliminal-learning/)

**Core claims (LLM setting)**
- A teacher with a trait (a preference for owls, or misalignment) generates only number sequences. A student fine-tuned on them acquires the trait, even after references to the trait are filtered out. The effect also appears with code and with chain-of-thought (CoT) traces. [V] — [arXiv abs](https://arxiv.org/abs/2507.14805)
- The effect is not observed when teacher and student use different base models. [V] — [arXiv abs](https://arxiv.org/abs/2507.14805)
- Prompted LLM classifiers and in-context learning both fail to reliably detect the transmitted traits. [S] — [Nature Asia / press summary](https://www.natureasia.com/en/info/press-releases/detail/9302)

**Theorem 1 (single gradient step, shared initialisation)**
- **Assumptions.** Teacher and student start from the same parameters, θ_S⁰ = θ_T⁰.
  - The teacher takes one gradient step on any differentiable loss ℒ_T.
  - The student takes one gradient step with learning rate α on teacher-labelled samples x ~ 𝒟.
  - The student's loss ℒ_S is squared error or softmax cross-entropy.
- **Result.** One of two things holds:
  - the student update is orthogonal to the teacher update (Δθ_S · Δθ_T = 0), or
  - for small enough ε, the student's teacher loss falls below its starting value: ℒ_T(θ_S^ε) < ℒ_T(θ_S⁰).
- ℒ_T need not be related to ℒ_S or to 𝒟.
- Lemma 1 gives the underlying fact: Δθ_S · Δθ_T ≥ 0.

[V] — [arXiv HTML](https://arxiv.org/html/2507.14805)

**MNIST auxiliary-logit experiment** [V] — [arXiv HTML](https://arxiv.org/html/2507.14805)
- **Architecture.** A multilayer perceptron (MLP) with layers (784, 256, 256, 10+m), ReLU activations and m = 3 auxiliary logits. The 10 regular logits are softmax-normalised over digits.
- **Teacher.** A randomly initialised reference model, trained for 5 epochs with cross-entropy on MNIST. Only the 10 regular logits enter the loss.
- **Student.** A copy of the same reference initialisation, trained for 5 epochs to match the teacher's auxiliary logits on random-noise images. It sees "no class logits and no handwritten digit inputs".
- **Result.** On the MNIST test set, the student's never-trained regular logits reach "over 50% accuracy". Figure 10 shows 95% confidence intervals over 100 runs.
- **Controls.**
  - student trained on all logits;
  - student trained on auxiliary logits from a teacher with a different initialisation;
  - student trained on all logits from a teacher with a different initialisation.

  The cross-initialisation transfer fails. The only difference between the reference models is their initialisation.
- **Published version (unverified).** Secondary sources say the Nature version adds that transfer also occurs, "to a lesser extent", when the student is first trained to match the teacher's *initial* auxiliary logits. [S] — [search summary of Nature version](https://www.nature.com/articles/s41586-026-10319-8)

**Limitations stated by the authors** [V] — [arXiv HTML](https://arxiv.org/html/2507.14805)
- The distillation tasks and prompts are artificial and simplistic.
- The models had already been trained on GSM8K, which affects the CoT setting.
- It is unknown why some animals are not transmitted by some models, and what can be transmitted in general.
- The experiments violate the theorem's assumptions: they use many SGD steps on sampled outputs, not one step on full logits, and the outputs are filtered. The precise conditions for subliminal learning in practice remain open.

### Inferences
- The project's setup is the MNIST experiment with smart meter series in place of digits and household-attribute classes in place of digit classes. Theorem 1 applies directly to soft auxiliary-logit distillation, which is the regime closest to the theory. The LLM experiments use hard (sampled-token) distillation, which is further from it.
- The theorem is a first-order, local guarantee: it says the direction is non-negative, but says nothing about how large the effect is. Leakage size therefore has to be measured empirically, for example as attribute accuracy above chance in the student's untrained head against different-initialisation controls.
- The Nature finding that matching the teacher's *initial* auxiliary logits partly substitutes for a shared initialisation (unverified) matters for threat models. An attacker who cannot copy the initialisation but can query the teacher's pre-training outputs might still recover part of the channel.

### Gaps
- I could not read the Nature full text, so I could not confirm the exact control accuracies or the wording on initial-logit matching.
- The exact accuracies of the MNIST control students were not reported in the text I read; they appear only in a figure.

---

## Q2. Follow-ups and critiques, 2025–2026

### Takeaway
Most follow-ups use the MNIST auxiliary-logit setting, so they bear directly on the project. They agree the effect is real and seed-dependent, roughly 55% ± 10% accuracy. They disagree about whether a shared initialisation is strictly needed. Brockers et al. argue that compatible output heads matter more than shared hidden layers. Chauhan & Shah find that the channel sits in input-layer alignment and is predicted by centred kernel alignment (CKA) between representations. Early proposed defences are mixed: multiple diverse teachers and paraphrasing help, liminal training does not.

### Cited Findings
**Mechanism and theory, MNIST / MLP setting**
- **Brockers, Ventzke, Neuhaus, Hidalgo-Ogalde, Priesemann (MPI-DS / Göttingen).** "Learning Through Noise: Why Subliminal Learning Works and When It Fails", arXiv:2605.23645, 22 May 2026, preprint. [V] — [arXiv HTML](https://arxiv.org/html/2605.23645)
  - They state that "a closely matched initialization is not necessary". Reinitialising one or both hidden layers reduces efficiency, but transfer returns with enough auxiliary neurons.
  - Reinitialising the auxiliary head strongly suppresses transfer.
  - Reinitialising the class head eliminates transfer at every auxiliary size tested. The class head gets no gradient during distillation, so this cannot be corrected.
  - With m = 1, accuracy peaks at about 83% given many noise samples. With a fixed N = 1000 noise samples, accuracy plateaus at about 29% however large m becomes.
  - At large latent width d, student accuracy collapses even with a fully shared initialisation. They attribute this to auxiliary-head drift.
  - Transfer persists from an MLP teacher to a CNN student when the CNN is given Perlin (spatially correlated) noise. With uniform [-1, 1] noise it does not reliably work for CNNs.
  - On EMNIST, student accuracy falls faster than teacher accuracy as the number of classes grows.
- **Kitkana & Arora.** "Sustained Gradient Alignment Mediates Subliminal Learning in a Multi-Step Setting: Evidence from MNIST Auxiliary Logit Distillation", arXiv:2604.25779, 28 Apr 2026. Preprint; the acknowledgments mention reviewers of the Sci4DL Workshop. [V] — [arXiv HTML](https://arxiv.org/html/2604.25779v1)
  - Over 100 seeds the student reaches 55.28 ± 10.48% (range 26.31–76.04%).
  - The cosine similarity between the trait gradient and the distillation gradient is positive on about 78% of steps, but tiny on average (0.0075).
  - Projecting out the trait-aligned component of the distillation gradient drops accuracy to 10.14% (chance), while the KL curves are unchanged.
  - Liminal training (Yanagisawa et al. 2025) does not prevent transfer: 48.97 ± 9.63%.
- **Chauhan & Shah (UCSC / Google).** "Covert Trait Propagation Is Representation Alignment: Mechanistic Evidence from Hidden-Channel Distillation", arXiv:2607.04432, 5 Jul 2026, preprint. [V] — [arXiv HTML](https://arxiv.org/html/2607.04432)
  - **Teacher learning rate.** Student accuracy falls from about 50% to chance as the teacher learning rate goes from 1e-3 to 1e-2, while the teacher stays about 94.8% accurate. In other words, the further the teacher moves from the initialisation, the weaker the channel.
  - **Layer freezing.** Freezing the input projection drops the student from 42.3% to 15.2%. Freezing the output projection gives 38.2%, a change within noise.
  - **Multiple teachers.** Accuracy falls with more teachers: 64.3% with one, 34.8% with two, 19.8% with five. The label–auxiliary-logit mutual information stays roughly constant, at about 1.84–2.00 bits.
  - **CKA.** Over 16 interpolation points between shared and independent initialisations, CKA correlates with student accuracy at r = 0.98.
- **Li, Ying, Dong, Yang.** "Why Ghost Outputs Teach: A Kernel-Based Understanding of Subliminal Learning", arXiv:2609.23260, 20 Sep 2026, preprint. [V] — [arXiv abs](https://arxiv.org/abs/2609.23260)
  - They model the effect with a "chained cross-task kernel" linking the auxiliary ("ghost") outputs to the task outputs through the shared backbone.
  - With a shared initialisation, the transfer operator is positive semi-definite.
  - The number of ghost outputs acts as a rank bottleneck.
  - High-entropy synthetic inputs act as broadband probes that increase kernel overlap, which may explain why noise works better than structured data.

**Mechanism and critiques, LLM setting**
- **Schrodi, Kempf, Barez, Brox.** "Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer", arXiv:2509.23886 (v2, 5 Mar 2026), preprint. [V] — [arXiv HTML](https://arxiv.org/html/2509.23886)
  - Transfer occurs under greedy decoding and after removing entangled tokens.
  - It is driven by "divergence tokens": about 7.5% of fine-tuning tokens for Qwen and 18.3% for Gemma.
  - Fine-tuning a single early layer is enough to transmit the bias.
  - Meaning-preserving paraphrases usually suppress transfer.
  - Mixing in unbiased-teacher data weakens it: 10% substantially, 25% nearly eliminates it.
  - Some cross-family transfer occurs (an eagle preference from Gemma to Qwen), contrary to the strict shared-initialisation claim.
- **Other follow-ups (titles and claims from search snippets only).** [S] — [search results](https://arxiv.org/html/2608.05734)
  - "Subliminal Learning is Non-Semantic Distillation" (arXiv:2608.05734) reports no activation evidence of the latent signal, so activation-based audits fail.
  - "Subliminal Learning as Trait-Direction Drift" (arXiv:2609.01091) proposes a mechanism and a targeted control.
  - "Verbalizing Subliminal Learning Effects Using Text Optimization" (arXiv:2609.16927).
  - "Distillation for Incrimination and Distillation for Capabilities" (arXiv:2610.11012).
  - Snippets also mention: Zur et al. (token entanglement plus logit leakage); Draganov et al. 2026 (cross-model transfer with semantically broad query sets); "Subliminal Learning is a LoRA Artifact" (Nief et al.); "Subliminal Steering" (Morgulis & Hewitt); "Quantum Subliminal Learning" (arXiv:2605.29557).

**Defences**
- **Egan, Gilligan-Lee, O'Riordan, Zhu.** "Defence against the dark arts of distillation: Unsupervised discovery of subliminal learning", arXiv:2610.13536, 9 Oct 2026. [V] — [arXiv abs](https://arxiv.org/abs/2610.13536)
  - They recover a transferred trait from the parameter updates alone, using the fact that the trait is consistent across many teacher-generated datasets while task-specific effects vary.
  - The result is a direction in weight space. Projecting it out suppresses the trait with almost no loss on the task, tested on the Cloud et al. benchmarks.
- **Liminal training** (Yanagisawa et al., 2025) is a KL-regularisation defence. I know it only through Kitkana & Arora, where it failed in the MNIST setting. [S] — [Kitkana & Arora](https://arxiv.org/html/2604.25779v1)

### Inferences
- Design implications for the smart meter study:
  1. Sweep the number of auxiliary logits m and the number of noise queries N separately. Brockers et al. show both are bottlenecks.
  2. Choose the noise distribution with care. Plain i.i.d. noise may fail for convolutional or temporal architectures, much as Perlin noise was needed for CNNs. For load profiles, consider noise with realistic autocorrelation (e.g. 1/f or AR noise) as an alternative probe family.
  3. Include initialisation-interpolation controls and report CKA. Chauhan & Shah's r = 0.98 suggests CKA as a ready-made covariate for "how shared".
  4. Vary teacher learning rate and training length to test the distance-from-initialisation hypothesis.
  5. Ablate the class head and the auxiliary head separately. Brockers et al. locate the necessary component in head compatibility.
- Multi-teacher averaging and a different-initialisation student are cheap candidate defences to test. CKA monitoring and weight-space trait-direction projection (Egan et al.) are candidate detectors.

### Gaps
- Nearly all follow-ups are recent arXiv preprints, and I found no peer-reviewed venues. Several 2026 titles were seen only as snippets.
- I found no follow-up applying subliminal learning to time-series, tabular, or energy data. The project would appear to be first in this domain; a final check against Google Scholar "cited by" is advisable.
- I did not retrieve a full "cited by" list for Cloud et al.

---

## Q3. Related prior work: dark knowledge, what distillation transfers, data-free/noise distillation, model extraction, property inference, MIA on distilled models, privacy of KD

### Takeaway
Classic knowledge distillation (KD) work already establishes the ingredients of subliminal leakage:
- soft outputs carry far more than the argmax ("dark knowledge");
- students do not simply copy teacher functions;
- noise or synthetic inputs can be enough to distil;
- black-box outputs enable model extraction and property inference;
- distilled students leak membership of the teacher's training set, including through queries on unrelated examples.

Subliminal learning adds that a shared initialisation turns even task-irrelevant outputs into a channel.

### Cited Findings
**Dark knowledge and what distillation transfers**
- Hinton, Vinyals, Dean. "Distilling the Knowledge in a Neural Network", NIPS 2014 Deep Learning Workshop, arXiv:1503.02531 (2015). Soft targets carry "dark knowledge" about class similarity. [K] — [arXiv](https://arxiv.org/abs/1503.02531)
- Stanton, Izmailov, Kirichenko, Alemi, Wilson. "Does Knowledge Distillation Really Work?", NeurIPS 2021, arXiv:2106.05945. Students often fail to match the teacher's predictive distribution even when they have the capacity; fidelity and generalisation diverge. [K] — [arXiv](https://arxiv.org/abs/2106.05945)
- Ojha, Li, Sankaranarayanan, Lee et al. "What Knowledge Gets Distilled in Knowledge Distillation?", NeurIPS 2023, arXiv:2205.16004. Students inherit teacher properties beyond task accuracy, such as localisation, adversarial vulnerability and invariances. [K] — [arXiv](https://arxiv.org/abs/2205.16004)
- Furlanello et al. "Born-Again Neural Networks", ICML 2018, arXiv:1805.04770. Same-architecture self-distillation. [K] — [arXiv](https://arxiv.org/abs/1805.04770)

**Data-free and noise-input distillation**
- Lopes, Fenu, Starner. "Data-Free Knowledge Distillation for Deep Neural Networks", NIPS 2017 LLD Workshop, arXiv:1710.07535. Uses activation metadata. [K] — [arXiv](https://arxiv.org/abs/1710.07535)
- Nayak et al. "Zero-Shot Knowledge Distillation in Deep Neural Networks", ICML 2019, arXiv:1905.08114. [K] — [arXiv](https://arxiv.org/abs/1905.08114)
- Micaelli & Storkey. "Zero-shot Knowledge Transfer via Adversarial Belief Matching" (ZSKT), NeurIPS 2019, arXiv:1905.09768. [K] — [arXiv](https://arxiv.org/abs/1905.09768)
- Yin et al. "Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion", CVPR 2020, arXiv:1912.08795. [K] — [arXiv](https://arxiv.org/abs/1912.08795)
- A NeurIPS 2022 paper on noise-engineered data-free KD distils on pure Gaussian noise. It frames this as avoiding privacy concerns in sensitive domains, and identifies the shift in hidden-activation distribution under noise as the main limitation. [S] — [NeurIPS 2022 poster](https://neurips.cc/virtual/2022/poster/53095); [paper](https://papers.nips.cc/paper_files/paper/2022/file/1f96b24df4b06f5d68389845a9a13ed9-Paper-Conference.pdf)
- NAYER (CVPR 2024) notes that random noise inputs "inherently lack meaningful information" for data generation. [S] — [CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Tran_NAYER_Noisy_Layer_Data_Generation_for_Efficient_and_Effective_Data-free_CVPR_2024_paper.pdf)
- "Does Training with Synthetic Data Truly Protect Privacy?" (arXiv:2502.12976, 2025) reports that in data-free KD, synthetic data can activate the teacher's memorisation of private data and leak it to the student, even when the synthetic data look visually distinct. [S] — [arXiv](https://arxiv.org/pdf/2502.12976)
- Differentially private data-free distillation (IJCAI 2023), with a label-DP follow-up (arXiv:2409.12384). [S] — [IJCAI](https://www.ijcai.org/proceedings/2023/0243.pdf); [arXiv](https://arxiv.org/pdf/2409.12384)

**Model extraction from outputs**
- Tramèr, Zhang, Juels, Reiter, Ristenpart. "Stealing Machine Learning Models via Prediction APIs", USENIX Security 2016, arXiv:1609.02943. [K] — [arXiv](https://arxiv.org/abs/1609.02943)
- Orekondy, Schiele, Fritz. "Knockoff Nets: Stealing Functionality of Black-Box Models", CVPR 2019, arXiv:1812.02766. [K] — [arXiv](https://arxiv.org/abs/1812.02766)

**Property and attribute inference**
- Ateniese et al. "Hacking Smart Machines with Smarter Ones" (property inference from model parameters), IJSN 2015, arXiv:1306.4447. [K] — [arXiv](https://arxiv.org/abs/1306.4447)
- Ganju, Wang, Yang, Gunter, Borisov. "Property Inference Attacks on Fully Connected Neural Networks using Permutation Invariant Representations", ACM CCS 2018, DOI 10.1145/3243734.3243834. [K] — [ACM DL](https://doi.org/10.1145/3243734.3243834)
- Melis, Song, De Cristofaro, Shmatikov. "Exploiting Unintended Feature Leakage in Collaborative Learning", IEEE S&P 2019, arXiv:1805.04049. [K] — [arXiv](https://arxiv.org/abs/1805.04049)

**Membership inference and distillation privacy**
- Shokri, Stronati, Song, Shmatikov. "Membership Inference Attacks against Machine Learning Models", IEEE S&P 2017, arXiv:1610.05820. [K] — [arXiv](https://arxiv.org/abs/1610.05820)
- Jagielski, Nasr, Lee, Choquette-Choo, Carlini, Tramèr. "Students Parrot Their Teachers: Membership Inference on Model Distillation", NeurIPS 2023 (oral), arXiv:2303.03446. [V] — [arXiv](https://arxiv.org/abs/2303.03446); [NeurIPS](https://neurips.cc/virtual/2023/poster/71209)
  - Distillation alone gives only limited privacy; a LiRA-style attack works well against students.
  - Membership can be inferred from the student's predictions on *other, seemingly unrelated* examples (e.g. red-hued non-bird images).
  - Attacks are strongest when the student and teacher sets are similar, or when the teacher set can be poisoned.
- Papernot, Abadi, Erlingsson, Goodfellow, Talwar. "Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data" (PATE), ICLR 2017, arXiv:1610.05755. Scalable PATE: Papernot et al., ICLR 2018, arXiv:1802.08908. Distillation with a DP aggregator is the standard *private* KD baseline. [K] — [arXiv](https://arxiv.org/abs/1610.05755); [arXiv](https://arxiv.org/abs/1802.08908)

### Inferences
- The project's threat model is a mix of three lines of work:
  - property inference, because the leaked quantity is a population-level household attribute and not membership;
  - data-free KD, because the student only queries noise;
  - cross-example leakage, as in Jagielski et al., because leakage appears on inputs unrelated to the sensitive task.
- PATE and DP-SGD on the teacher are the natural baselines for defence experiments. Under DP, Theorem 1's channel should shrink in proportion to the teacher's clipped and noised updates; this is a testable prediction.
- The "Does training with synthetic data protect privacy?" result suggests a second leakage pathway, memorisation activated by synthetic inputs. That pathway exists even without a shared initialisation, so the project should separate initialisation-dependent leakage from memorisation leakage, for example with different-initialisation controls.

### Gaps
- Items marked [K] were not re-fetched; check the author lists and venues before final citation.
- I found no paper that directly measures membership or attribute leakage from noise-only (Gaussian or uniform) distillation students. This appears to be an open gap that the project fills.

---

## Q4. Shared initialisation and pretrained checkpoints: linear mode connectivity, task vectors, merging, NTK/lazy regime, privacy of public checkpoints

### Takeaway
Networks fine-tuned from a common checkpoint stay in one linearly connected basin and close to the initialisation. In that near-lazy regime, weight differences behave like task vectors that can be added and compared. This geometry is what makes a shared initialisation a "common coordinate system", and it is why Theorem 1 and its kernel follow-up predict leakage. Pre-trained checkpoints are also a known privacy attack surface: poisoned checkpoints amplify membership inference and data stealing.

### Cited Findings
**Basins and linear mode connectivity**
- Neyshabur, Sedghi, Zhang. "What is being transferred in transfer learning?", NeurIPS 2020, arXiv:2008.11687. Models fine-tuned from the same pre-trained checkpoint lie in the same basin and are linearly connected; models trained from scratch are not. [K] — [arXiv](https://arxiv.org/abs/2008.11687)
- Frankle, Dziugaite, Roy, Carbin. "Linear Mode Connectivity and the Lottery Ticket Hypothesis", ICML 2020, arXiv:1912.05671. Networks that share an early-training state become stable to SGD noise and linearly connected. [K] — [arXiv](https://arxiv.org/abs/1912.05671)
- Entezari et al. "The Role of Permutation Invariance in Linear Mode Connectivity", ICLR 2022, arXiv:2110.06296. [K] — [arXiv](https://arxiv.org/abs/2110.06296)
- Ainsworth, Hayase, Srinivasa. "Git Re-Basin", ICLR 2023, arXiv:2209.04836. Differently initialised networks can often be aligned by permutation. This suggests a possible attack that recovers a "shared" coordinate system without a shared initialisation. [K] — [arXiv](https://arxiv.org/abs/2209.04836)

**Task vectors and model merging**
- Ilharco et al. "Editing Models with Task Arithmetic", ICLR 2023, arXiv:2212.04089. [K] — [arXiv](https://arxiv.org/abs/2212.04089)
- Ortiz-Jiménez, Favero, Frossard. "Task Arithmetic in the Tangent Space", NeurIPS 2023, arXiv:2305.12827. Ties task arithmetic to weight disentanglement and to near-linear (NTK-like) fine-tuning. [K] — [arXiv](https://arxiv.org/abs/2305.12827)
- Wortsman et al. "Model Soups", ICML 2022, arXiv:2203.05482. Averaging works for models fine-tuned from the same initialisation. [K] — [arXiv](https://arxiv.org/abs/2203.05482)

**Lazy / NTK regime**
- Jacot, Gabriel, Hongler. "Neural Tangent Kernel", NeurIPS 2018, arXiv:1806.07572. [K] — [arXiv](https://arxiv.org/abs/1806.07572)
- Lee et al. "Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent", NeurIPS 2019, arXiv:1902.06720. [K] — [arXiv](https://arxiv.org/abs/1902.06720)
- Chizat, Oyallon, Bach. "On Lazy Training in Differentiable Programming", NeurIPS 2019, arXiv:1812.07956. [K] — [arXiv](https://arxiv.org/abs/1812.07956)
- Malladi et al. "A Kernel-Based View of Language Model Fine-Tuning", ICML 2023, arXiv:2210.05643. [K] — [arXiv](https://arxiv.org/abs/2210.05643)

**Direct links to subliminal transfer and distance from initialisation**
- Li et al. 2026 show the cross-task kernel is PSD under a shared initialisation; see Q2. [V] — [arXiv](https://arxiv.org/abs/2609.23260)
- Chauhan & Shah 2026 find the channel collapses at high teacher learning rates even though the teacher remains accurate, i.e. a distance-from-initialisation effect, and that CKA predicts transfer. [V] — [arXiv](https://arxiv.org/html/2607.04432)
- Kitkana & Arora 2026 report that gradient alignment is highest early in training. [V] — [arXiv](https://arxiv.org/html/2604.25779v1)

**Privacy of fine-tuning from public checkpoints**
- Tramèr, Kamath, Carlini. "Considerations for Differentially Private Learning with Large-Scale Public Pretraining", ICML 2024 (position paper), arXiv:2212.06470. [K] — [arXiv](https://arxiv.org/abs/2212.06470)
- Feng & Tramèr. "Privacy Backdoors: Stealing Data with Corrupted Pretrained Models", ICML 2024, arXiv:2404.00473. [K] — [arXiv](https://arxiv.org/abs/2404.00473)
- Wen et al. "Privacy Backdoors: Enhancing Membership Inference through Poisoning Pre-trained Models", arXiv:2404.01231 (2024; venue believed to be NeurIPS 2024, uncertain). [K, uncertain] — [arXiv](https://arxiv.org/abs/2404.01231)
- Tramèr et al. "Truth Serum: Poisoning Machine Learning Models to Reveal Their Secrets", ACM CCS 2022, arXiv:2204.00032. [K] — [arXiv](https://arxiv.org/abs/2204.00032)

### Inferences
- **Threat model.** In practice a shared initialisation corresponds to a public seed or a public pre-trained checkpoint, such as a foundation model for load forecasting. Neyshabur et al. and the task-vector literature imply that any two models fine-tuned from it share a basin, which is the regime where Theorem 1 and the PSD-kernel result apply. Using public checkpoints in energy ML may therefore open a subliminal attribute channel.
- **Testable hypothesis.** Leakage should fall as ‖θ_T − θ_0‖ grows (teacher learning rate, epochs, weight decay toward θ_0). Chauhan & Shah give initial MLP evidence. Measuring CKA and the cosine between the teacher's task vector and the student's distillation update would connect the experiment to the task-arithmetic literature.
- **Possible bypass.** Git Re-Basin-style permutation alignment might let an attacker partly undo a "different initialisation" defence. This is speculative and untested.

### Gaps
- I did not find a paper that explicitly links distance from initialisation to leakage through distillation, other than the subliminal-learning MLP work above.
- No privacy study of energy foundation models or shared load-forecasting checkpoints was found.

---

## Q5. Energy and smart meter settings: privacy of smart meter ML and federated learning leakage on load data

### Takeaway
Inferring household socio-demographics from raw load profiles is well established. Privacy of trained *models* on smart meter data (federated learning (FL) gradient leakage, property inference) is discussed mostly in surveys and differential-privacy defence papers, with few dedicated attack evaluations. I found no work on distillation-based or subliminal leakage in energy. That gap is the project's niche.

### Cited Findings
**Attribute inference from raw load profiles**
- Radovanovic, Schirl, Unterweger, Eibl. "The Impact of Temporal Granularity on Socio-Demographic Inference from Household Load Profiles", arXiv:2606.03358, 2 Jun 2026 (book chapter). [V] — [arXiv](https://arxiv.org/abs/2606.03358)
  - Data: 1,589 households over one year, granularities from 15 minutes to 7 days, 8 attributes.
  - Accuracy plateaus between 15 minutes and 1 hour, and between 1 and 7 days.
  - XGBoost performs best; handcrafted and tsfresh features are competitive with CNN-autoencoder embeddings.
  - Dwelling size is inferable even from coarse data; detecting a swimming pool needs fine-grained data.
  - This is the project team's own prior work and supplies the attribute-inference baseline and candidate sensitive attributes.
- Wang et al. "Deep Learning-Based Socio-Demographic Information Identification From Smart Meter Data", IEEE Trans. Smart Grid (2019), CNN on the Irish CER data. IEEE document 8291011 appeared in search; I did not confirm that it is this paper. [S/K] — [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/8291011)
- Beckel, Sadamori, Staake, Santini. "Revealing household characteristics from smart meter data", *Energy* 78 (2014), DOI 10.1016/j.energy.2014.10.025. [K] — [DOI](https://doi.org/10.1016/j.energy.2014.10.025)
- Molina-Markham et al. "Private Memoirs of a Smart Meter", ACM BuildSys 2010. [K] — [ACM DL](https://doi.org/10.1145/1878431.1878446) (DOI from memory, uncertain)
- Privacy-preserving deep learning (multi-party computation) for identifying consumer characteristics, *Frontiers in Energy Research* (2022), DOI 10.3389/fenrg.2022.992117. [S] — [Frontiers](https://www.frontiersin.org/articles/10.3389/fenrg.2022.992117/full)

**Model and FL leakage on load data**
- A survey, "Federated Learning for Smart Grid: A Survey on Applications and Potential Vulnerabilities" (arXiv:2409.10764), discusses how membership and property inference can reveal routines and attributes from fine-grained load data. [S] — [arXiv](https://arxiv.org/html/2409.10764v2)
- A 2025 MDPI paper on federated smart-meter theft detection argues that FL updates can expose household-specific details, revealing occupancy, appliance use and socioeconomic status, and that DP accounting is needed. [S] — [MDPI BDCC](https://www.mdpi.com/2504-2289/10/4/113)
- DP-enhanced federated household load forecasting (ResearchGate entry). The snippet says the authors report that updates leak unintended information and construct passive and active inference attacks. [S, attribution uncertain] — [ResearchGate](https://www.researchgate.net/publication/357890174_A_Differential_Privacy-enhanced_Federated_Learning_Method_for_Short-Term_Household_Load_Forecasting_in_Smart_Grid)
- Privacy-preserving FL for energy disaggregation (IET Cyber-Physical Systems, 2025). [S] — [Wiley](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cps2.70013)

### Inferences
- The project combines two established results: attributes are inferable from load profiles, and distillation with a shared initialisation transmits a teacher's task through task-irrelevant outputs. Together they suggest that a teacher trained to predict an attribute, or one that has simply learned it as a side feature, could pass it to a student that is never shown load data or labels.
- Realistic deployment analogues:
  - a utility shares a forecasting model's auxiliary outputs, such as quantiles or embeddings, on synthetic probes;
  - federated or split pipelines share a common initialisation, which FL does by design, since all clients start from the global model.

  FL's shared global initialisation every round makes subliminal transfer between clients and the server a plausible, unstudied channel.

### Gaps
- I found no published work on knowledge-distillation privacy, data-free KD, or subliminal learning on smart meter or other energy time-series.
- Dedicated property-inference attacks against FL load-forecasting models were not confirmed. A targeted follow-up search is advisable, for example "attribute inference federated load forecasting" or Husnoo et al. 2023, cited in the FL smart grid survey.
- Bibliographic details for Wang et al. 2019 and Molina-Markham et al. 2010 need confirming.
