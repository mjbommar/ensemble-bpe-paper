# Bootstrap Aggregating and Bagging in NLP: Applications to Text Preprocessing and Tokenization

## Summary

Bootstrap aggregating (bagging) is a foundational ensemble learning technique that creates multiple diverse models by training on bootstrap-sampled datasets and combining their predictions through voting or averaging. While extensively studied in traditional machine learning tasks, its direct application to text preprocessing and tokenization remains relatively unexplored. However, several related concepts have emerged that embody similar principles: subword regularization as a form of ensemble training, ensemble preprocessing methods that combine multiple text transformation strategies, and byte-level probability frameworks that enable ensembling models with different tokenizers.

The research reveals an important distinction: rather than creating multiple separate tokenizer models and voting on outputs (traditional bagging), the NLP community has largely adopted alternative approaches. These include probabilistic tokenization methods that sample from multiple segmentation candidates during training (subword regularization), data augmentation techniques that apply bootstrap sampling principles to text, and meta-ensemble methods that aggregate predictions from models using different vocabularies by converting to a common byte-space representation. These techniques demonstrate that ensemble thinking pervades modern NLP, even when not explicitly framed as bootstrap aggregating.

The implications for ensemble BPE tokenization are significant: multiple complementary strategies exist for incorporating diversity and robustness into tokenization pipelines, from training-time regularization through probabilistic sampling to inference-time ensembling across different vocabulary spaces. The choice depends on whether diversity should exist within a single model (regularization) or across multiple independent models (traditional ensembling).

## Key Findings

- **Bootstrap aggregating reduces variance in unstable learners** by training multiple models on randomly sampled subsets (with replacement) and aggregating predictions, particularly effective for high-variance algorithms like decision trees and neural networks (Breiman, 1996; Machine Learning Mastery, 2024)

- **Subword regularization acts as ensemble training for tokenization** by exposing models to multiple probabilistic segmentations during training rather than deterministic sequences, achieving +1-2 BLEU improvements in low-resource settings and larger gains on out-of-domain data (Kudo, 2018)

- **Byte-level probability frameworks enable ensembling across incompatible vocabularies**, allowing models with different tokenizers to be combined by converting predictions to universal byte-space, yielding up to 3.7% improvement over individual models (ArXiv 2410.09303, 2024)

- **Ensemble preprocessing methods outperform single preprocessing techniques** by combining multiple complementary transformations that remove artifacts left by individual methods, applicable beyond spectral data to all preprocessing scenarios including NLP (TrAC Trends in Analytical Chemistry, 2020)

- **Random forests successfully apply bagging to text classification** when combined with bag-of-words or TF-IDF representations, with the ensemble of decision trees handling high-dimensional sparse text features effectively (Multiple sources, 2024)

- **BooStSa demonstrates bootstrap sampling for NLP model evaluation**, enabling statistical significance testing for both hard labels and soft probability distributions across multiple experimental conditions (ACL 2022)

- **Text data augmentation leverages ensemble-like principles** through consistency regularization and multiple augmented views of the same text, with particularly strong effects when training data is limited to 500-5000 examples (Journal of Big Data, 2021)

- **Ensemble transformer architectures with stacking achieve substantial gains** over single models, with combinations of DeBERTa, RoBERTa, T5, and GPT-2 reducing RMSE by 24% and improving F1-score by 43% through meta-learning aggregation (ArXiv 2409.19013, 2024)

## Relevant Research & Papers

### Bootstrap Aggregating (Bagging) - Wikipedia & Machine Learning Mastery
- **Authors/Source**: Breiman (1996), Machine Learning Mastery (Jason Brownlee)
- **Year**: 1996 (original), 2024 (contemporary sources)
- **Key Contributions**:
  - Formalized bootstrap aggregating as creating m datasets of size n' by sampling with replacement from original dataset D
  - Each bootstrap sample contains ~63.2% unique observations with duplicates comprising remainder
  - Reduces variance through "wisdom of crowds" - combining semi-independent models dampens individual overfitting
  - Particularly effective for unstable procedures (neural networks, decision trees) but may degrade stable methods (k-NN)
  - Cannot predict beyond training data range (extrapolation limitation)
- **Relevance to Ensemble BPE**: Provides theoretical foundation for creating diverse tokenizer models through bootstrap sampling of training corpora, though direct application to tokenization not yet extensively studied

### Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates
- **Authors/Source**: Taku Kudo (2018), ArXiv 1804.10959
- **Year**: 2018
- **Key Contributions**:
  - Proposes on-the-fly probabilistic sampling of subword segmentations during training rather than deterministic tokenization
  - Introduces Unigram Language Model approach that assigns probabilities to segmentations, enabling sampling
  - Frames approach as "variant of ensemble training" - similar to dropout but for input representations
  - Achieves +1-2 BLEU points in low-resource settings (IWSLT), +2 BLEU on out-of-domain evaluation
  - Demonstrates both source and target-side regularization provide complementary value
  - Language-agnostic improvements across Vietnamese, Chinese, Japanese, French, Arabic, German, Czech
- **Relevance to Ensemble BPE**: Demonstrates ensemble principles applied at segmentation level rather than model level - single model exposed to multiple tokenization variants acts as implicit ensemble

### BooStSa: Hard and Soft Evaluation of NLP Models with Bootstrap Sampling
- **Authors/Source**: ACL Anthology 2022.acl-demo.12
- **Year**: 2022
- **Key Contributions**:
  - Simplifies statistical significance testing for NLP model evaluation
  - Implements bootstrap sampling procedure to compute significance levels
  - Handles both hard labels (categorical) and soft labels (probability distributions)
  - Automates tedious calculations across multiple experimental conditions
  - Emphasizes robust model selection requires statistical rigor beyond raw metrics
- **Relevance to Ensemble BPE**: Provides methodology for rigorously evaluating whether ensemble tokenizer approaches yield statistically significant improvements over baselines

### Text Data Augmentation for Deep Learning
- **Authors/Source**: Journal of Big Data (SpringerOpen), 2021
- **Year**: 2021
- **Key Contributions**:
  - Categorizes augmentation into symbolic (rule-based: EDA, MixUp, graph-based) and neural (back-translation, generative models)
  - Identifies that augmentation addresses expensive annotation costs (23 hours for 124 QA pairs cited)
  - Shows EDA particularly effective with 500-5000 labeled examples, diminishing returns beyond
  - Introduces consistency regularization (enforcing prediction invariance across transformations) and contrastive learning
  - Discusses augmentation controllers (AutoAugment, RandAugment) that automate technique selection
  - Notes "NLP is at an early stage in applying Data Augmentation compared to Computer Vision"
- **Relevance to Ensemble BPE**: Bootstrap sampling of training data for augmentation could create diverse training sets for multiple BPE models; consistency regularization applicable to tokenization robustness

### Improving Academic Skills Assessment with NLP and Ensemble Learning
- **Authors/Source**: ArXiv 2409.19013, 2024
- **Year**: 2024
- **Key Contributions**:
  - Combines DeBERTa, RoBERTa, ALBERT, BART, ELECTRA, GPT-2, BERT, T5 via stacking ensemble
  - Two-stage approach: Stage 1 (SVR/Ridge feature extraction), Stage 2 (Ridge/LightGBM meta-learning)
  - Pseudo-labeling enhances training data through semi-supervised learning
  - Achieves 24% RMSE reduction, 43% F1-score improvement over single DeBERTa baseline
  - Demonstrates complementary strengths of different transformer architectures
- **Relevance to Ensemble BPE**: Meta-learning aggregation strategies applicable to combining predictions from models with different tokenizers; shows substantial gains from architectural diversity

### Exact Byte-Level Probabilities from Tokenized Language Models for Model Ensembles
- **Authors/Source**: ArXiv 2410.09303, 2024
- **Year**: 2024
- **Key Contributions**:
  - Enables ensembling models with incompatible vocabularies via byte-space conversion
  - Byte-Token Representation (BTR) Lemma establishes mappings between token and byte distributions
  - Identifies "tokenization bias" - statistically equivalent models have different byte-level predictions
  - FIM code completion: 63.9% pass rate (byte-level) vs 45.0% (token-level), 18% improvement
  - Model ensemble improvements up to 3.7% over individual models
  - Works with BPE, Maximum Prefix Encoding, and other deterministic tokenizers
- **Relevance to Ensemble BPE**: Directly enables ensembling multiple BPE models with different vocabularies - solves previously intractable aggregation problem for multi-tokenizer ensembles

### Tokenizer Summary - Hugging Face Documentation
- **Authors/Source**: Hugging Face Team
- **Year**: 2024 (current documentation)
- **Key Contributions**:
  - Compares BPE (frequency-based merging), WordPiece (likelihood maximization), Unigram (loss-based pruning)
  - BPE: builds up from characters, deterministic, used in GPT-2, RoBERTa, GPT
  - WordPiece: evaluates statistical impact of merges, used in BERT, DistilBERT, ELECTRA
  - Unigram: prunes from large vocabulary, probabilistic, combined with SentencePiece in ALBERT, XLNet, T5
  - SentencePiece treats input as raw character streams, solving space-free language tokenization
  - No true ensemble methods discussed - tokenizers operate as alternatives not combined systems
- **Relevance to Ensemble BPE**: Documents standard tokenization approaches that could be combined in ensemble; highlights fundamental differences that might provide complementary strengths

### New Data Preprocessing Trends Based on Ensemble of Multiple Preprocessing Techniques
- **Authors/Source**: TrAC Trends in Analytical Chemistry, 2020
- **Year**: 2020
- **Key Contributions**:
  - Ensemble preprocessing removes artifacts left by single techniques through complementary combination
  - Multiple preprocessing in complementary way achieves better artifact removal than sequential application
  - Applicable beyond spectral data to all preprocessing scenarios
  - Selection of techniques and combinations leads to improved downstream models
- **Relevance to Ensemble BPE**: Conceptual framework for combining multiple tokenization preprocessing strategies (normalization, cleaning, segmentation) rather than relying on single approach

## Technical Details

### Bootstrap Aggregating Algorithm

1. **Bootstrap Sampling**: For training set D of size n, create m bootstrap samples D₁, D₂, ..., Dₘ by sampling n' instances with replacement
2. **Model Training**: Train base model M on each bootstrap sample to create ensemble members M₁, M₂, ..., Mₘ
3. **Prediction Aggregation**:
   - Regression: Average predictions ŷ = (1/m) Σ Mᵢ(x)
   - Classification: Majority voting ŷ = mode(M₁(x), M₂(x), ..., Mₘ(x))

Mathematical foundation: With replacement sampling means E[unique samples] ≈ 0.632n for large n

### Subword Regularization Approach

**Training Objective**: Maximize expected log-likelihood over segmentation distribution
```
θ* = argmax_θ Σ E_x~P(x|X) [log P(y|x; θ)]
```

**Practical Implementation**:
- Sample k=1 segmentation per training example per iteration
- With sufficient iterations, approximates full marginalized objective
- Unigram LM: P(x) = Π p(xᵢ) enables probabilistic sampling
- EM algorithm optimizes subword probabilities, iteratively prunes low-value tokens

**BPE-Dropout Extension**: Applies dropout to BPE merge operations during encoding to generate multiple segmentations, making deterministic BPE probabilistic

### Byte-Level Ensemble Framework

**Byte-Token Representation Lemma**: For deterministic tokenizer τ, exact byte-level probabilities extractable from token-level distribution

**Cover Encoding Search**: Identifies all valid token sequences representing target byte sequence, computes marginal probabilities

**Ensemble Aggregation**:
```
P_ensemble(next_byte|context) = (1/N) Σᵢ P_modelᵢ(next_byte|context)
```

Enables mixture-of-experts, weighted averaging, and other combination strategies across incompatible vocabularies

### Random Forest Text Classification

**Pipeline**:
1. Text → CountVectorizer/TF-IDF → High-dimensional sparse matrix
2. Bootstrap sample features and instances for each tree
3. Train decision trees with random feature subsets at each split
4. Aggregate predictions via majority voting

**Considerations**: All features potentially relevant in TF-IDF matrix (unlike typical RF assumptions of sparse feature importance), may require different hyperparameter tuning

## Implications for Ensemble BPE Tokenization

### Multiple Pathways for Ensemble Tokenization

The research reveals several distinct approaches to incorporating ensemble principles into tokenization:

1. **Model-Level Ensembling**: Train multiple BPE models on bootstrap samples of training corpus, aggregate tokenization decisions via voting or statistical combination (traditional bagging approach)

2. **Regularization-Based Ensembling**: Single BPE model with probabilistic segmentation sampling during training (subword regularization approach) - implicit ensemble within one model

3. **Vocabulary-Level Ensembling**: Train models with different tokenizers (BPE, WordPiece, Unigram), convert predictions to byte-space for aggregation (byte-level probability framework)

4. **Preprocessing Ensembling**: Combine multiple complementary preprocessing transformations before or during tokenization to remove artifacts and improve robustness

### Training Data Diversity Strategies

Bootstrap sampling provides multiple mechanisms for BPE training data diversity:

- **Corpus-level bootstrapping**: Sample documents/sentences with replacement to create training sets for different BPE models
- **Token-level bootstrapping**: Sample from token frequency distributions to create variant merge priority orderings
- **Domain-level bootstrapping**: Sample from multiple domains to create domain-robust or domain-specific ensemble members
- **Augmentation bootstrapping**: Apply data augmentation (back-translation, paraphrasing) to expand training corpus variations

Each strategy creates different types of tokenizer diversity that may capture complementary linguistic patterns.

### Variance Reduction in Tokenization

Bagging reduces variance by combining diverse models. For tokenization, variance manifests as:

- **Segmentation instability**: Small corpus changes causing large vocabulary shifts
- **Rare word handling**: Inconsistent subword decomposition of infrequent terms
- **Domain transfer**: Poor generalization to out-of-distribution text

Ensemble BPE could reduce these variances by:
- Averaging merge decisions across multiple bootstrap-trained vocabularies
- Maintaining ensemble members specialized on different corpus subsets
- Combining deterministic BPE with probabilistic regularization for robust segmentation

### Evaluation and Significance Testing

BooStSa methodology applicable to tokenization evaluation:

- **Bootstrap resampling**: Evaluate tokenization quality metrics (compression rate, downstream task performance) across resampled test sets
- **Significance testing**: Determine if ensemble tokenizer improvements over single BPE are statistically robust
- **Multi-metric evaluation**: Assess both hard metrics (exact match) and soft metrics (embedding similarity) of tokenization outputs

### Practical Implementation Considerations

**Computational Costs**:
- Multiple BPE models increase training time linearly with ensemble size
- Inference requires running text through multiple tokenizers unless using regularization approach
- Byte-level conversion adds overhead but enables vocabulary-incompatible ensembling

**Memory Requirements**:
- Each ensemble member requires separate vocabulary storage
- Byte-level frameworks maintain multiple token-to-byte mappings
- Practical systems may need vocabulary compression or shared prefix trees

**Optimization Opportunities**:
- Parallel training of bootstrap-sampled BPE models (embarrassingly parallel)
- Cached tokenization for common sequences across ensemble members
- Pruning ensemble members with high correlation (diversity-based selection)

### Connection to Modern NLP Practices

The research suggests ensemble tokenization aligns with broader NLP trends:

- **Multi-model systems**: Following success of model ensembles (T5 + DeBERTa + GPT combinations), tokenizer ensembles provide similar diversity benefits at preprocessing stage
- **Robustness emphasis**: Byte-level fallbacks, character-aware models, and robust tokenization increasingly important as models deploy across diverse domains
- **Probabilistic frameworks**: Shift from deterministic pipelines to probabilistic approaches enables uncertainty quantification and better calibration

### Open Research Directions

Several promising directions emerge from this synthesis:

1. **Adaptive ensemble weighting**: Learn task-specific or domain-specific weights for ensemble tokenizer members
2. **Hierarchical ensembles**: Combine character, subword (BPE), and word-level tokenizers in multi-granularity ensemble
3. **Cross-lingual ensembling**: Bootstrap sample from multilingual corpora to create language-robust tokenizers
4. **Meta-learning tokenization**: Use meta-learning to select optimal ensemble configuration per task/domain
5. **Tokenization-aware training**: Co-train tokenizer ensemble and language model to optimize end-to-end performance

## Open Questions & Future Directions

### Unexplored Territory in Ensemble Tokenization

1. **Optimal ensemble size for BPE**: Traditional bagging uses 50-500 models; what is the bias-variance tradeoff for tokenizer ensembles? When do returns diminish?

2. **Bootstrap sampling granularity**: Should bootstrapping occur at document, sentence, or token level for BPE training? How does granularity affect ensemble diversity and quality?

3. **Voting mechanisms for tokenization**: How to aggregate tokenization decisions when ensemble members disagree? Simple majority voting, confidence-weighted voting, or learned aggregation?

4. **Combination with subword regularization**: Can model-level ensembles (multiple BPE models) be combined with training-level regularization (probabilistic segmentation) for multiplicative benefits?

5. **Domain adaptation via selective ensembling**: Can ensemble members specialize on different domains, with adaptive weighting based on input characteristics?

### Gaps in Current Research

1. **Lack of direct bagging applications to tokenization**: Most ensemble NLP research focuses on model predictions, not preprocessing stages like tokenization

2. **Limited evaluation of tokenization ensemble quality**: Need metrics beyond downstream task performance (e.g., consistency, coverage, robustness to noise)

3. **Computational efficiency unexplored**: No research on efficient inference for tokenizer ensembles, caching strategies, or pruning redundant members

4. **Integration with modern architectures**: How do tokenizer ensembles interact with byte-level models, character-aware transformers, or tokenization-free approaches?

5. **Theoretical analysis lacking**: No formal analysis of variance reduction in tokenization ensembles, generalization bounds, or sample complexity

### Promising Future Directions

1. **Hybrid ensemble-regularization approaches**: Combine multiple BPE models (ensemble) each using probabilistic segmentation (regularization) for robust tokenization at training and inference

2. **Curriculum ensembling**: Start with diverse ensemble members, progressively prune to most complementary subset based on validation performance

3. **Multi-objective tokenization ensembles**: Optimize ensemble for multiple objectives (compression, downstream accuracy, fairness across languages/domains)

4. **Active learning for ensemble construction**: Iteratively add ensemble members targeting weaknesses of current ensemble (e.g., poor rare word handling)

5. **Federated tokenizer ensembles**: Train ensemble members on different data sources without centralizing data, aggregate for privacy-preserving tokenization

6. **Interpretable ensemble analysis**: Use ensemble disagreement to identify ambiguous or problematic tokenization cases, guide vocabulary design

### Connections to Broader Research Areas

1. **Neural architecture search**: Can AutoML techniques automatically design optimal tokenizer ensemble configurations?

2. **Continual learning**: How should tokenizer ensembles evolve as new data arrives? Add new members vs. retrain existing?

3. **Uncertainty quantification**: Ensemble disagreement as proxy for tokenization uncertainty, useful for active learning and model calibration

4. **Fairness and bias**: Can ensemble tokenization reduce bias by combining tokenizers trained on balanced corpus subsamples?

5. **Compression and efficiency**: Relationship between ensemble tokenization and model compression - do better tokenizers enable smaller models?

### Practical Implementation Challenges

1. **Production deployment**: Engineering challenges of serving multiple tokenizer models in low-latency production systems

2. **Backward compatibility**: Migrating existing systems from single BPE to ensemble while maintaining consistency

3. **Version control**: Managing multiple tokenizer versions across ensemble members, ensuring reproducibility

4. **Monitoring and debugging**: Detecting when ensemble members diverge or degrade, root cause analysis for tokenization failures

## References

### Primary Research Papers

1. Kudo, T. (2018). Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. ArXiv 1804.10959. https://ar5iv.labs.arxiv.org/html/1804.10959

2. ArXiv 2410.09303 (2024). Exact Byte-Level Probabilities from Tokenized Language Models for FIM-Tasks and Model Ensembles. https://arxiv.org/html/2410.09303

3. ArXiv 2409.19013 (2024). Improving Academic Skills Assessment with NLP and Ensemble Learning. https://arxiv.org/html/2409.19013

4. Bjerva, J. et al. (2022). BooStSa: Hard and Soft Evaluation of NLP Models with Bootstrap Sampling. ACL Anthology 2022.acl-demo.12. https://aclanthology.org/2022.acl-demo.12/

5. Shorten, C., Khoshgoftaar, T.M., & Furht, B. (2021). Text Data Augmentation for Deep Learning. Journal of Big Data, 8, 101. https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00492-0

6. Engel, J. et al. (2020). New data preprocessing trends based on ensemble of multiple preprocessing techniques. TrAC Trends in Analytical Chemistry, 132, 116045. https://www.sciencedirect.com/science/article/pii/S0165993620302740

### Technical Documentation and Tutorials

7. Hugging Face Team (2024). Summary of the tokenizers. https://huggingface.co/docs/transformers/tokenizer_summary

8. Brownlee, J. (2024). Essence of Bootstrap Aggregation Ensembles. Machine Learning Mastery. https://machinelearningmastery.com/essence-of-bootstrap-aggregation-ensembles/

9. Towards Data Science (2024). Byte-Pair Encoding: Subword-based tokenization algorithm. https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0/

### Wikipedia and Reference Material

10. Wikipedia (2024). Bootstrap aggregating. https://en.wikipedia.org/wiki/Bootstrap_aggregating

11. Wikipedia (2024). Bag-of-words model. https://en.wikipedia.org/wiki/Bag-of-words_model

### Additional Resources

12. Medium - Jay Jo (2024). Ensemble Learning: Bagging/Bootstrap Aggregation. https://medium.com/@wjj1019/ensemble-learning-bagging-bootstrap-aggregation-b7900e58caa2

13. DataCamp (2024). A Guide to Bagging in Machine Learning. https://www.datacamp.com/tutorial/what-bagging-in-machine-learning-a-guide-with-examples

14. Analytics Vidhya (2020). What is Bootstrap Sampling in Statistics and Machine Learning? https://www.analyticsvidhya.com/blog/2020/02/what-is-bootstrap-sampling-in-statistics-and-machine-learning/

15. GeeksforGeeks (2024). Bag of words (BoW) model in NLP. https://www.geeksforgeeks.org/nlp/bag-of-words-bow-model-in-nlp/

16. CodeSignal (2024). Mastering Random Forest for Text Classification. https://codesignal.com/learn/courses/introduction-to-modeling-techniques-for-text-classification/lessons/mastering-random-forest-for-text-classification

17. GitHub - google/sentencepiece. Unsupervised text tokenizer for Neural Network-based text generation. https://github.com/google/sentencepiece

18. GitHub - bheinzerling/bpemb. Pre-trained subword embeddings in 275 languages, based on Byte-Pair Encoding. https://github.com/bheinzerling/bpemb

19. GroundAI (2024). Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. https://www.groundai.com/project/subword-regularization-improving-neural-network-translation-models-with-multiple-subword-candidates/

20. Medium - Sieun Park (2024). Improving subword tokenization with Subword Regularization. https://sieunpark77.medium.com/subword-regularization-6aace5e6a165
