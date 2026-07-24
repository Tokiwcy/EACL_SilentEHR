# SilentEHR: Auditing Closed-World Extrapolation in Frozen Patient Representations

## 3 Methods

### 3.1 Problem formulation

Given a patient record \(x\) and a clinical proposition \(h\), the clinically
appropriate evidence judgment is \(y(x,h)\in\{\textsc{Supported},
\textsc{Refuted},\textsc{Unknown}\}\). We study a common closed-world pipeline
that is trained only on Supported and Refuted examples and must therefore
extrapolate when the available record does not determine the proposition. Our
primary question is directional: whether genuine non-observation is
systematically mapped to Refuted. We separately analyze the magnitude and
calibration of the resulting confidence.

### 3.2 SilentEHR controlled benchmark

SilentEHR crosses 16 clinical concepts with 80 target-neutral synthetic patient
backgrounds. The concepts cover symptoms, allergies, medication exposure,
disease states, and safety-relevant conditions. Four concepts (dyspnea,
contrast allergy, opioid therapy, and liver dysfunction) are reserved for
concept-held-out evaluation. Concept-specific contamination rules exclude six
of 1,280 candidate background--concept combinations that contain direct or
indirect target evidence.

For each of the remaining 1,274 matched combinations, we generate two
Supported, two Refuted, two Not-assessed, and two Uncertain variants using
independent training and held-out templates. We additionally retain one
Unmentioned variant in which the target concept is not inserted. The resulting
benchmark contains 11,466 patient--proposition pairs. Unmentioned profiles are
not duplicated, and all variants derived from one background--concept pair
remain in the same fold.

We report four protocols. The primary new-background protocol trains on 56
backgrounds and evaluates on 12 held-out backgrounds; a further 12 backgrounds
form the validation set. IID, held-out assertion-template, and held-out concept
protocols isolate group-level sampling, template generalization, and concept
generalization, respectively. The template protocol does not contain
Unmentioned examples and is treated only as an assertion-generalization test.

### 3.3 Frozen encoders

We compare a sparse lexical baseline (TF--IDF), a clinical language model not
optimized as an embedding model (Bio_ClinicalBERT), general dense encoders
(Qwen3-Embedding-0.6B and Jina v5 text matching), a medical semantic encoder
(BioLORD-2023), a biomedical retrieval encoder (MedCPT Article Encoder), and a
clinical narrative encoder (MedAI-HS/med-gte-hybrid). All neural parameters are
frozen. Sentence-transformer outputs are L2 normalized.
Bio_ClinicalBERT uses attention-mask mean pooling; MedCPT uses its CLS
representation. These two encoders use a maximum input length of 512 tokens.
Jina inputs receive the model-required `Document:` prefix. TF--IDF uses at most
4,096 unigram and bigram features and is fitted only on the training data.

### 3.4 Closed-world classifier

For patient embedding \(e_x\) and proposition embedding \(e_h\), the pair
representation is

\[
r(x,h)=[e_x;e_h;|e_x-e_h|;e_x\odot e_h].
\]

The primary head is class-balanced logistic regression fitted after feature
standardization. It receives only Supported and Refuted training examples;
Unknown examples are reserved for evaluation. The primary protocol uses seed
0. Robustness analyses vary regularization, remove scaling, reduce features
with PCA, replace the linear head with nearest-centroid or cosine 5-nearest
neighbors, and apply temperature scaling selected on held-out known examples.

### 3.5 Controlled metrics

Known-state accuracy verifies that a model can distinguish explicit positive
and negative evidence. The primary failure metric is the Unmentioned-to-
Refuted Rate (UNR). HCUNR@0.9 records the subset assigned Refuted with
probability above 0.9, but is interpreted as calibration-sensitive. We measure
the detectability of non-observation using AUROC with low maximum confidence as
the score and report risk--coverage curves.

For mechanism analysis, we report decision-axis logits by epistemic state and
PCA--Ledoit--Wolf Mahalanobis distance. Geometry is summarized by matched
\(\mathrm{ECI}=1-d(U,R)/d(S,R)\), where \(S,R,U\) denote Supported, Refuted,
and Unmentioned representations. ECI is secondary because global distance need
not agree with the supervised decision projection.

Confidence intervals use 2,000 percentile bootstrap replicates. The
background--concept pair, rather than an individual template, is the
resampling unit. We bootstrap known accuracy, Unmentioned UNR, Unmentioned
HCUNR, and the paired Unmentioned-minus-Refuted decision-logit difference.

### 3.6 Structured-grounded MIMIC validation

We join MIMIC-IV discharge summaries with laboratory events and define the
clinical state from the last value in the 24 hours before discharge:
potassium \(>5.5\) for hyperkalemia, hemoglobin \(<10\) for anemia, and
platelets \(<100\) for thrombocytopenia. This yields 38,043
admission--concept pairs. Using the last near-discharge measurement avoids
labeling corrected transient abnormalities as current conditions.

Observation status is assigned by prompting **DeepSeek V4 Flash** on the full
discharge note for each concept. The model returns one of seven statuses:
observed positive, observed negative, unmentioned, uncertain, historical only,
admission only, or temporal conflict. Mixed, uncertain, historical, and
conflicting cases are excluded from classifier experiments. Full discharge
notes are encoded with chunk mean-pooling (4,800 characters per chunk, 400
character overlap) rather than hard truncation.

Subjects are split 70/30, so no patient contributes admissions to both train
and test. Training uses only concordant observed cells, \(C^+O^+\) and
\(C^-O^+\). Evaluation includes concordant observed cases and both
\(C^+O^-\) and \(C^-O^-\). Anemia, hyperkalemia, and thrombocytopenia each
meet the minimum of 20 observed training examples per clinical class. Neural
models use a fixed cap of 500 cases per concept--observation--clinical-state
cell. We report observed and undocumented positive recall, their documentation
gap, undocumented-positive false-negative rate (UPFNR), high-confidence wrong
negative rate (HCWE@0.9), clinical-state AUROC within unmentioned cases, and
AUROC for detecting non-observation from confidence.

### 3.7 Mitigation

We compare raw binary prediction, temperature scaling, confidence and distance
abstention, and three-way learning with 10, 50, or 100 Unknown semantic groups.
Unknown supervision is sampled at the background--concept--state group level
over ten seeds. We evaluate both transfer to held-out backgrounds and transfer
from 12 training concepts to four held-out concepts. An oracle observation gate
maps Unknown cases to abstention while retaining binary predictions for known
cases; it is an upper bound, not a deployable extraction system.

### 3.8 MIMIC validity audit

We prepare a 300-case audit stratified by concept (anemia or
thrombocytopenia) and structured clinical state. Two annotators independently
assign seven-class observation status aligned with DeepSeek V4 Flash
(`observed_positive/negative`, `unmentioned`, `admission_only`,
`historical_only`, `uncertain`, `temporal_conflict`), with evidence spans for
non-unmentioned labels. We report inter-annotator agreement, per-annotator and
consensus agreement with DeepSeek V4 Flash, and binary mention agreement.

## 4 Results

### 4.1 Non-observation is projected in the negative direction

New-background results show strong but non-universal negative extrapolation.
Unmentioned UNR is 0.688 for TF--IDF, 0.597 for Bio_ClinicalBERT, 0.944 for
Qwen, 0.910 for Jina, 0.993 for BioLORD, 0.792 for MedCPT, and 0.382 for
med-gte. Known-state accuracy ranges from 0.892 to 1.000. Thus the strongest
failures cannot be attributed to an inability to classify explicit Supported
and Refuted evidence, but med-gte provides an important counterexample to a
universal encoder-level claim.

Extreme confidence is model dependent. Unmentioned HCUNR@0.9 ranges from
0.160 for med-gte to 0.986 for BioLORD. Qwen, Jina, BioLORD, and MedCPT
assign high-confidence Refuted predictions to 0.882, 0.806, 0.986, and 0.722
of Unmentioned cases, respectively. Final calibrated and alternative-head
results will determine how much of this magnitude survives probability
correction.

### 4.2 Confidence often fails to identify non-observation

Dense encoders differ sharply from the lexical baseline. Confidence detects
Unmentioned cases almost perfectly for TF--IDF (AUROC 0.999), but performs
near chance for Jina (0.498) and below chance for BioLORD (0.184) and MedCPT
(0.300). Qwen is only moderately separable (0.664). Bio_ClinicalBERT (0.866)
and med-gte (0.747) are more detectable, consistent with their weaker or
reversed negative extrapolation.

### 4.3 The failure is not equivalent to geometric collapse

Matched ECI is positive only for BioLORD (0.099); it is negative for the other
six models, including Qwen (-0.151), Jina (-0.943), MedCPT (-3.949),
and med-gte (-2.156). The controlled result therefore does not support a universal claim
that Unmentioned representations are nearest to Refuted in the original
embedding space. Instead, the evidence supports closed-world extrapolation
along a supervised decision direction. Final decision-axis and paired
bootstrap analyses test this interpretation directly.

### 4.4 Real discharge summaries show clinically consequential false negatives

On the DeepSeek V4 Flash-labeled, full-note MIMIC cohort with a matched 500-case cap
per cell, joint undocumented-positive false-negative rates range from 0.463 for
med-gte to 0.812 for TF--IDF. For the five completed dense encoders, UPFNR
ranges from 0.463 to 0.600; high-confidence wrong-negative rates range from
0.375 to 0.463. Confidence-based detection of unmentioned cases remains near
chance for every dense encoder (AUROC 0.439--0.456). TF--IDF is again more
detectable (AUROC 0.495) but still far from the controlled benchmark. These
findings do not imply that text models should recover an unavailable laboratory
value. They show that the absence of textual evidence is frequently converted
into a clinical negative rather than uncertainty.

The documentation gap is model dependent (0.131--0.629 in current runs) and
must be interpreted together with UPFNR, confidence, structured-state AUROC,
and the concept-prior controls.

Subject-level bootstrap intervals confirm that these errors are not driven by a
small number of patients. On the matched cohort, neural UPFNR ranges from 0.463
(95% CI 0.351--0.581) for med-gte to 0.600 (0.493--0.708) for
Bio_ClinicalBERT; Jina yields 0.488 (0.385--0.620), BioLORD 0.537
(0.435--0.641), and MedCPT 0.475 (0.368--0.603). Neural HCWE@0.9 ranges from
0.375 (0.297--0.458) to 0.463 (0.375--0.556). TF--IDF has UPFNR 0.812
(0.718--0.909), HCWE 0.650 (0.560--0.731), and detection AUROC 0.495
(0.453--0.537).

Qwen3-Embedding achieves the lowest dense-encoder UPFNR on this cohort (0.425;
95% CI 0.322--0.549) despite the strongest controlled-benchmark CWE.

### 4.4a Artifact-resistant controlled tier

On the same 1,274 background--concept folds, three controls test whether CWE
is an artifact of length mismatch, fixed final-sentence insertion, or direct
lexical templates. Length-matched Unmentioned profiles add concept-neutral
fillers (mean 70.2 vs 69.6 words for other states). Random-position evidence
inserts Supported/Refuted sentences at pseudo-random sentence indices. Indirect
evidence replaces direct assertions with lab- and medication-style sentences.

Unmentioned UNR remains high for Qwen (0.882 length-matched; 0.944 random-position;
0.785 indirect), Jina (0.799; 0.993; 0.910), and BioLORD (0.993; 1.000; 0.861).
TF--IDF is unchanged under length and position controls (0.688) but rises to
1.000 under indirect evidence, confirming lexical rather than semantic coverage
exploitation.

### 4.4b Human validity audit

A 300-case dual-annotator audit calibrates DeepSeek V4 Flash observation labels.
Inter-annotator agreement is 85.0% (Cohen's \(\kappa=0.80\)). Individual
agreement with DeepSeek V4 Flash is 53.7% (Human1) and 54.7% (Human2); mean 54.2%;
consensus (both agree, \(n=255\)) 59.2% (\(\kappa=0.44\)). Binary mention
agreement is 88.3%, 87.3%, and 89.0%, respectively. Remaining errors
concentrate on `admission_only`, which has no equivalent in the LLM schema.
Thrombocytopenia is harder than anemia (48% vs 61% per annotator).

### 4.5 Direction is more robust than extreme confidence

The negative direction survives every tested regularization, scaling, PCA, kNN,
and nearest-centroid configuration for BioLORD (Unmentioned UNR
0.972--0.993), Jina (0.771--0.979), and Qwen (0.708--0.965). It is not
head-invariant for MedCPT (0.285--0.792), Bio_ClinicalBERT (0.354--0.646),
TF--IDF (0.306--0.688), or med-gte (0.215--0.722). The primary-head
background--concept bootstrap intervals exclude 0.5 for BioLORD
(0.979--1.000), Jina (0.861--0.951), Qwen (0.903--0.979), MedCPT
(0.722--0.854), TF--IDF (0.611--0.757), and narrowly Bio_ClinicalBERT
(0.514--0.674), whereas med-gte lies below chance (0.306--0.459).

In contrast, HCUNR@0.9 spans zero under at least one configuration for every
model. Temperature scaling leaves UNR unchanged, as expected for a monotonic
logit transformation, but reduces HCUNR to 0.597 for Qwen, 0.667 for Jina,
0.451 for MedCPT, 0.160 for Bio_ClinicalBERT, and zero for med-gte. BioLORD
remains an exception with calibrated HCUNR 0.965. These results support the
paper's central distinction: negative direction is robust for a subset of
modern dense encoders, while the extremity of confidence is substantially
head- and calibration-dependent.

### 4.6 Explicit Unknown supervision outperforms generic abstention

Train-calibrated confidence abstention is unreliable for the encoders with the
strongest negative extrapolation. At thresholds retaining 95.1%, 95.8%, and
85.4% of known test cases, it rejects only 3.5% of BioLORD, 28.5% of Qwen,
and 39.6% of Jina Unmentioned cases, respectively. A Mahalanobis threshold
calibrated to retain 95% of known training examples transfers even less
reliably: it retains all known test cases but rejects none of the Unmentioned
cases for six models and only 25.0% for med-gte. Distance ranking can still be
informative for individual models, but a training-derived absolute OOD
threshold does not generalize across backgrounds.

In contrast, 50 labeled Unknown semantic groups substantially reduce
new-background UNR while largely preserving known accuracy. Qwen falls from
0.944 to 0.058 UNR with 0.888 Unknown recall and 0.982 known accuracy; Jina
falls from 0.910 to 0.224 with 0.667 recall and 0.962 accuracy; BioLORD falls
from 0.993 to 0.294 with 0.587 recall and 0.989 accuracy. With 100 groups,
their UNRs fall further to 0.026, 0.112, and 0.135.

Transfer to held-out concepts is harder but not absent. With 50 Unknown
groups, Qwen obtains 0.204 UNR, 0.748 Unknown recall, and 0.974 known
accuracy; BioLORD obtains 0.302, 0.520, and 0.892. Jina improves Unknown
behavior but loses known accuracy (0.266 UNR, 0.511 recall, 0.741 accuracy).
At 100 groups, MedCPT and med-gte also incur substantial held-out-concept
known-accuracy losses. These results identify explicit Unknown supervision as
the strongest simple mitigation, while showing that coverage semantics do not
transfer uniformly across encoder and concept families.

## 5 Discussion

The structured-grounded MIMIC experiment validates closed-world extrapolation
for two near-discharge laboratory states, anemia and thrombocytopenia. It does
not establish that the same prevalence holds for symptoms, allergies,
medication exposure, or safety concepts. Those concept families are covered by
the controlled benchmark, which supports mechanism analysis but not prevalence
claims about real documentation. We therefore treat the MIMIC result as
laboratory-state validation and avoid generalizing its effect size to all
clinical concepts. Adding a third concept is justified only if its structured
state, temporal semantics, and observation labels meet the same validity
standard; a noisier concept would weaken rather than broaden the evidence.

