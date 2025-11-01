# SentencePiece Unigram Language Model and EM Algorithm for Tokenization

## Summary

SentencePiece's unigram language model represents a fundamentally different approach to subword tokenization compared to traditional methods like Byte Pair Encoding (BPE). Instead of building vocabulary bottom-up through iterative merging, the unigram model starts with a large seed vocabulary and uses the Expectation-Maximization (EM) algorithm to iteratively prune tokens based on their contribution to corpus likelihood. This probabilistic approach enables multiple valid segmentations of the same text, which can be leveraged as a regularization technique during neural network training.

The unigram model, introduced by Taku Kudo in the 2018 ACL paper "Subword Regularization," treats tokenization as a maximum likelihood estimation problem. Each possible tokenization of a text is assigned a probability based on the product of individual token probabilities under a unigram language model. The Viterbi algorithm efficiently finds the optimal segmentation during encoding, while the EM algorithm optimizes token probabilities during training. This probabilistic framework enables "subword regularization," where multiple segmentation candidates are sampled during training to improve model robustness, particularly in low-resource and out-of-domain settings.

Research comparing unigram tokenization to BPE reveals significant advantages in morphological preservation. Unigram models better recover common English morphemes like suffixes ('ly', 's', 'ing') and correctly segment complex words (e.g., "de-stabiliz-ing" vs BPE's "dest-ab-il-izing"). This linguistic accuracy, combined with the method's probabilistic flexibility, makes unigram tokenization particularly valuable for tasks requiring understanding of word structure and composition, though BPE remains widely adopted due to ecosystem support and established infrastructure.

## Key Findings

- **EM Algorithm Core**: The unigram tokenization training process uses Expectation-Maximization to iteratively estimate token probabilities from corpus frequencies, segment the corpus using Viterbi decoding, compute loss for each token, and prune 10-20% of low-impact tokens until reaching the desired vocabulary size (Source: Hugging Face LLM Course, Data Science Stack Exchange)

- **Mathematical Formulation**: The unigram model maximizes P(X) = ∏ᵢ₌₁ᴹ P(xᵢ) where tokens (x₁, x₂, ..., xₙ) are chosen to maximize the product of token probabilities. The overall loss is L = -∑ᵢ₌₁ᴺ log(∑ₓ ∈ S(xᵢ) p(x)), where S(xᵢ) represents all possible tokenizations of word xᵢ (Source: Data Science Stack Exchange, Guillaume Be blog)

- **Viterbi Decoding**: The encoding process uses a two-phase Viterbi algorithm with a forward pass identifying maximum-likelihood tokens at each character position and a backward pass reconstructing the optimal sequence. Efficient DAG-based implementations achieve 13x speed improvements over naive nested-loop approaches (Source: Guillaume Be Rust SentencePiece implementation)

- **Morphological Superiority**: Testing against 8,000 words using Merriam-Webster syllabication showed unigram tokenizers substantially outperform BPE-based tokenizers in preserving morphological structure, better recovering common suffixes and prefix-root relationships (Source: Nick Dingwall blog)

- **Subword Regularization Benefits**: Training with multiple probabilistically sampled segmentations (enabled by the unigram model's probabilistic nature) shows consistent improvements especially on low-resource and out-of-domain settings. BPE-dropout, a related technique, achieves improvements of up to 2.3 BLEU over standard BPE (Source: Kudo ACL 2018, Provilkov et al. ACL 2020)

- **Top-Down vs Bottom-Up**: Unigram starts with a large vocabulary (e.g., from Suffix Array algorithm) and prunes downward, contrasting with BPE's bottom-up merging approach. This enables counterintuitive efficiency gains at larger vocabulary sizes since the algorithm prunes rather than builds incrementally (Source: Hugging Face LLM Course, Nick Dingwall blog)

- **Implementation Differences**: SentencePiece treats inputs as raw streams (no pre-tokenization required) and makes encoding/decoding lossless by preserving whitespace information. The unigram model shows slightly better text compression ratios than BPE and is more robust when no pretokenizer is applied (Source: SentencePiece GitHub, experiments documentation)

## Relevant Research & Papers

### Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates
- **Authors/Source**: Taku Kudo (Google)
- **Year**: 2018
- **Publication**: ACL 2018 (56th Annual Meeting of the Association for Computational Linguistics), Melbourne, Australia, July 2018, pages 66-75
- **DOI**: 10.18653/v1/P18-1007
- **arXiv**: 1804.10959
- **Key Contributions**:
  - Introduced subword regularization method that trains models with multiple subword segmentations probabilistically sampled during training
  - Proposed unigram language model-based subword segmentation algorithm as an alternative to BPE
  - Demonstrated consistent improvements especially on low-resource and out-of-domain settings across multiple corpora
  - Showed how to harness segmentation ambiguity as regularization noise to improve NMT robustness
- **Relevance to Ensemble BPE**: This work directly addresses the concept of leveraging multiple tokenization strategies. The unigram model's ability to produce multiple valid segmentations with associated probabilities provides a theoretical foundation for ensemble approaches. The demonstrated benefits in low-resource settings suggest ensemble methods combining different tokenization strategies could provide similar regularization benefits.

### BPE-Dropout: Simple and Effective Subword Regularization
- **Authors/Source**: Provilkov et al.
- **Year**: 2020
- **Publication**: ACL 2020
- **arXiv**: 1910.13267
- **Key Contributions**:
  - Introduced BPE-dropout, a stochastic corruption method for BPE segmentation that produces multiple segmentations within the same framework
  - Showed improvements up to 2.3 BLEU over standard BPE and 0.9 BLEU over previous subword regularization
  - Demonstrated larger improvements (1.6-2.3 BLEU) on misspelled test sets without exposure to misspellings during training
  - Proved subword regularization benefits extend to large datasets for practical noisy-input applications
- **Relevance to Ensemble BPE**: BPE-dropout demonstrates that stochastic variation in segmentation improves robustness. An ensemble approach could systematically leverage different tokenization strategies (BPE, unigram, WordPiece) rather than randomly perturbing a single method, potentially capturing complementary benefits of each algorithm.

### SentencePiece: A Simple and Language Independent Subword Tokenizer
- **Authors/Source**: Taku Kudo and John Richardson (Google)
- **Year**: 2018
- **Publication**: EMNLP 2018
- **Repository**: https://github.com/google/sentencepiece
- **Key Contributions**:
  - Implemented unified framework supporting both BPE and unigram language model algorithms
  - Introduced language-independent tokenization treating text as raw input streams without pre-tokenization
  - Achieved lossless encoding/decoding by preserving whitespace information
  - Provided C++ implementation with Python bindings for production-grade performance
- **Relevance to Ensemble BPE**: SentencePiece's unified framework demonstrates that multiple tokenization algorithms can share common infrastructure. This suggests ensemble approaches combining BPE and unigram within a single training pipeline are architecturally feasible.

### Tokenization for Language Modeling: Byte Pair Encoding vs Unigram Language Modeling
- **Authors/Source**: Nick Dingwall
- **Year**: 2024 (blog post)
- **URL**: https://ndingwall.github.io/blog/tokenization
- **Key Contributions**:
  - Empirical comparison showing unigram tokenizers beat BPE by substantial margin in morphological preservation
  - Demonstrated unigram correctly segments complex words (e.g., "de-stabiliz-ing") while BPE produces linguistically incorrect splits
  - Showed training time under 1 hour for 10M sentences on modest hardware
  - Identified negligible inference speed differences between methods
- **Relevance to Ensemble BPE**: The morphological analysis suggests BPE and unigram capture different linguistic properties. An ensemble approach could combine BPE's compression efficiency with unigram's morphological awareness, potentially achieving better overall representation quality than either method alone.

### A Rust SentencePiece Implementation
- **Authors/Source**: Guillaume Becquin
- **Year**: 2020
- **URL**: https://guillaume-be.github.io/2020-05-30/sentence_piece
- **Key Contributions**:
  - Detailed technical exposition of unigram training with EM algorithm
  - Comparison of Viterbi implementation strategies (HashMap vs DAG-based)
  - Demonstrated 13x performance improvement using DAG-based approach (12µs vs 614.5µs for longer sentences)
  - Provided clear mathematical formulations: P(X) = ∏P(xᵢ) with forward-backward Viterbi decoding
- **Relevance to Ensemble BPE**: The implementation details reveal computational considerations for unigram tokenization. For ensemble approaches, understanding these performance characteristics is critical for designing efficient multi-tokenizer systems, particularly regarding memory and compute trade-offs.

### Hugging Face LLM Course - Unigram Tokenization
- **Authors/Source**: Hugging Face
- **Year**: 2023-2024
- **URL**: https://huggingface.co/learn/llm-course/en/chapter6/7
- **Key Contributions**:
  - Step-by-step algorithmic description of unigram training process
  - Explanation of loss computation as sum of -log(P(word)) over corpus
  - Details on iterative pruning (typically 10-20% per iteration) while preserving base characters
  - Usage examples for models like AlBERT, T5, mBART, Big Bird, and XLNet
- **Relevance to Ensemble BPE**: The tutorial format provides clear implementation guidance that could inform ensemble tokenizer development. The list of models using unigram tokenization demonstrates its production viability and suggests ensemble approaches could build on proven infrastructure.

## Technical Details

### Expectation-Maximization Algorithm for Unigram Training

The EM algorithm for unigram tokenization follows this iterative process:

1. **Initialization**: Generate a large seed vocabulary using algorithms like Suffix Array from the training corpus. The initial vocabulary size is larger than the desired final vocabulary.

2. **E-Step (Expectation)**: Estimate the probability of each vocabulary token using frequency counts:
   ```
   p(xᵢ) = count(xᵢ) / Σⱼ count(xⱼ)
   ```
   where the sum is over all tokens in the current vocabulary.

3. **Tokenization Step**: Use the Viterbi algorithm to segment the corpus with current token probabilities, finding the most likely tokenization for each word.

4. **M-Step (Maximization)**: Compute the loss for each vocabulary token, defined as the decrease in overall likelihood L if that token were removed:
   ```
   L = -Σᵢ₌₁ᴺ log(Σₓ ∈ S(xᵢ) p(x))
   ```
   where S(xᵢ) represents all possible tokenizations of word xᵢ.

5. **Pruning**: Sort tokens by loss and keep only the top x% (typically 80%, removing 20%) with the lowest loss. Single-character tokens are always preserved to prevent out-of-vocabulary issues.

6. **Iteration**: Repeat steps 2-5 until the vocabulary reaches the desired size.

### Viterbi Decoding Algorithm

The Viterbi algorithm finds the optimal tokenization in two phases:

**Forward Pass**: For each position k in the word, compute the best score for all segmentations ending at position k:
```
Score(k, token) = Best_score(start_position) + log(p(token))
```

This can be implemented two ways:
- **HashMap approach**: O(N²) complexity, checking all possible substrings against vocabulary
- **DAG approach**: Constructs directed acyclic graph from vocabulary with common prefix search, reducing to O(k) where k is DAG depth

**Backward Pass**: Starting from the end position, follow stored backpointers to reconstruct the token sequence that achieved the maximum likelihood.

### Mathematical Formulation

The unigram language model makes the independence assumption that each subword occurs independently. For a sequence of tokens (x₁, x₂, ..., xₙ), the probability is:

```
P(X) = ∏ᵢ₌₁ⁿ P(xᵢ)
```

In practice, log probabilities are used to avoid numerical underflow:

```
log P(X) = Σᵢ₌₁ⁿ log P(xᵢ)
```

The optimal tokenization x* for input X from the set of all possible segmentations S(X) is:

```
x* = argmax_{x ∈ S(X)} P(x) = argmax_{x ∈ S(X)} ∏ᵢ₌₁ⁿ P(xᵢ)
```

### Subword Regularization Mechanism

Unigram's probabilistic nature enables sampling multiple segmentations during training:

1. For each training sentence, instead of using only the most likely segmentation, sample from the distribution of possible segmentations
2. The sampling probability for segmentation x is proportional to P(x)
3. Different segmentations of the same sentence serve as training augmentation
4. This regularization improves robustness to segmentation errors and domain shift

BPE-dropout achieves similar effects by stochastically corrupting the BPE merge order during segmentation, producing multiple outputs from the same vocabulary.

### Comparison of Tokenization Algorithms

| Aspect | BPE | Unigram | WordPiece |
|--------|-----|---------|-----------|
| **Direction** | Bottom-up (merge) | Top-down (prune) | Bottom-up (merge) |
| **Determinism** | Deterministic | Probabilistic | Deterministic |
| **Selection Criterion** | Frequency of pair | Likelihood maximization | Likelihood maximization |
| **Vocabulary Growth** | Incremental | Decremental | Incremental |
| **Morphology Preservation** | Poor | Good | Medium |
| **Training Complexity** | O(n log n) | O(n) with pruning | O(n log n) |
| **Inference Speed** | Fast | Fast | Fast |
| **Multiple Segmentations** | Requires dropout | Native support | Requires modification |

### Implementation Considerations

**SentencePiece Framework**:
- Treats input as raw byte stream without pre-tokenization
- Handles spaces as regular characters (marked as '_')
- Provides lossless encoding/decoding
- Supports both BPE and unigram algorithms in unified API
- Implemented in C++ for performance with Python bindings

**Vocabulary Initialization**:
- Suffix Array algorithm commonly used for seed vocabulary
- Alternative: Use BPE to generate initial large vocabulary
- Must be substantially larger than target size (e.g., 2-10x)

**Hyperparameters**:
- Target vocabulary size: Typically 8k-64k tokens
- Pruning rate: 10-20% per iteration
- Character coverage: 0.9995 (controls rare character handling)
- Sampling alpha (for regularization): 0.1-0.5

## Implications for Ensemble BPE Tokenization

### Theoretical Foundations for Ensemble Approaches

The unigram language model's probabilistic framework provides strong theoretical justification for ensemble tokenization. Since multiple valid segmentations exist with associated probabilities, combining outputs from different tokenization algorithms (BPE, unigram, WordPiece) could capture complementary linguistic properties:

1. **BPE**: Optimized for compression efficiency and frequency patterns
2. **Unigram**: Optimized for likelihood and morphological coherence
3. **WordPiece**: Balanced approach with good OOV handling

An ensemble could weight or sample from these different perspectives during training, similar to how subword regularization samples different segmentations from a single unigram model.

### Regularization Through Diversity

The success of subword regularization (2.3 BLEU improvement) and BPE-dropout (0.9-2.3 BLEU improvement) demonstrates that variation in tokenization improves model robustness. Ensemble approaches could:

- Train with different tokenizers for different mini-batches
- Use consensus voting or probability averaging across tokenizers
- Dynamically select tokenizer based on input characteristics
- Employ different tokenizers for encoder vs decoder in sequence-to-sequence models

### Morphological and Semantic Complementarity

Research shows BPE and unigram excel at different aspects:
- **Unigram**: Better morphological preservation, recovers linguistic structure
- **BPE**: Better established infrastructure, strong empirical track record

An ensemble could leverage both:
- Use unigram for languages with rich morphology
- Use BPE for languages where compression is more important
- Combine both for multilingual models to handle diverse linguistic properties

### Practical Implementation Strategies

Several ensemble architectures are feasible based on existing research:

1. **Parallel Tokenization**: Encode inputs with multiple tokenizers, concatenate or average embeddings
2. **Mixture of Tokenizers**: Route inputs to different tokenizers based on learned gating function
3. **Hierarchical Tokenization**: Use coarse tokenizer (e.g., BPE) followed by fine tokenizer (e.g., unigram)
4. **Probabilistic Ensemble**: Sample tokenizer for each training example based on learned or fixed distribution

### Low-Resource and Domain Adaptation

The demonstrated benefits of unigram and subword regularization in low-resource settings suggest ensemble tokenizers could be particularly valuable for:

- Few-shot learning scenarios
- Domain adaptation with limited target-domain data
- Zero-shot cross-lingual transfer
- Handling noisy or misspelled inputs (where BPE-dropout showed 1.6-2.3 BLEU gains)

### Computational Considerations

The Rust SentencePiece implementation showing 13x speedups reveals that efficient implementations are crucial for ensemble approaches. Key considerations:

- **Training Cost**: Unigram training is fast (<1 hour for 10M sentences), making ensemble training feasible
- **Inference Cost**: Negligible differences between BPE and unigram inference suggest ensemble inference overhead would be minimal per tokenizer
- **Memory**: Multiple vocabularies require more memory but modern systems can handle this
- **Parallelization**: Independent tokenizers can process in parallel on different cores/GPUs

### Research Directions Inspired by Unigram

1. **Adaptive Ensembles**: Learn which tokenizer(s) to use for different inputs or tasks
2. **Probabilistic Token Embeddings**: Weight token embeddings by segmentation probabilities from unigram model
3. **Cross-Tokenizer Alignment**: Develop methods to align token spaces from different tokenizers for ensemble combination
4. **Meta-Learning Tokenizers**: Use EM-like algorithms to optimize ensemble weights based on downstream task performance

## Open Questions & Future Directions

### Algorithmic Questions

1. **Optimal Ensemble Combination**: What is the best way to combine outputs from multiple tokenizers? Should they be:
   - Concatenated in parallel?
   - Averaged probabilistically?
   - Selected via learned gating mechanisms?
   - Used sequentially in hierarchical fashion?

2. **EM for Ensemble Training**: Can the EM algorithm used in unigram training be extended to optimize ensemble tokenizer weights? Could we jointly train token probabilities across multiple tokenization strategies?

3. **Convergence Guarantees**: Unigram's EM algorithm provides certain convergence properties. Do ensemble approaches maintain similar theoretical guarantees?

4. **Vocabulary Size Interactions**: How do vocabulary sizes of different tokenizers in an ensemble interact? Should all tokenizers have the same vocabulary size, or could complementary sizes be beneficial?

### Empirical Research Needs

1. **Comprehensive Benchmarking**: Systematic comparison of unigram vs BPE vs ensemble approaches across diverse:
   - Languages (morphologically rich vs isolating)
   - Domains (news, social media, technical, medical)
   - Task types (translation, generation, classification)
   - Resource levels (high vs low resource)

2. **Morphological Analysis**: Extend the Merriam-Webster syllabication analysis to:
   - Multiple languages with different morphological properties
   - Compound words in Germanic languages
   - Agglutinative languages (Turkish, Finnish, Japanese)
   - Evaluate ensemble tokenizers on same metrics

3. **Subword Regularization vs Ensemble**: Direct comparison of:
   - Single unigram model with subword regularization
   - BPE with BPE-dropout
   - Ensemble of BPE + unigram without regularization
   - Ensemble with regularization from all tokenizers

4. **Computational Profiling**: Detailed measurement of:
   - Training time for ensemble tokenizers
   - Inference latency with different ensemble architectures
   - Memory requirements for multiple vocabularies
   - GPU utilization patterns

### Implementation Challenges

1. **Framework Integration**: How to integrate ensemble tokenizers into existing training pipelines (Transformers, fairseq, etc.)?

2. **Vocabulary Management**: Efficient data structures for multiple vocabularies with potential overlap

3. **Gradient Flow**: For end-to-end training, how do gradients flow back through multiple tokenization paths?

4. **Caching and Optimization**: Can we pre-compute and cache multiple tokenizations? What are storage trade-offs?

### Theoretical Understanding

1. **Information Theory**: What is the information-theoretic relationship between different tokenization strategies? Are they capturing orthogonal information?

2. **Generalization Bounds**: Can we derive generalization bounds for ensemble tokenizers building on existing PAC learning theory?

3. **Optimal Tokenization**: Is there a theoretical "optimal" tokenizer, or is the answer task-dependent? Does ensemble approach the theoretical optimum?

4. **Bayesian Perspective**: Can ensemble tokenization be framed as Bayesian model averaging over tokenization strategies?

### Application-Specific Questions

1. **Multilingual Models**: Do different language families benefit from different tokenizers? Should multilingual ensembles use language-specific tokenizer selection?

2. **Code and Mixed Content**: How do different tokenizers handle code-switching, URLs, code snippets? Could ensembles improve robustness?

3. **Long-Context Models**: As context windows grow to 100k+ tokens, do tokenization strategies matter more or less?

4. **Multimodal Models**: For vision-language or audio-language models, how should text tokenization interact with other modalities?

### Beyond Text Tokenization

1. **Protein and DNA Sequences**: The paper "Linguistic Laws Meet Protein Sequences" compares tokenizers for biological sequences. Could ensemble approaches help?

2. **Time Series**: Can EM-based unigram approach extend to time series segmentation?

3. **Graph Tokenization**: For graph neural networks, could similar ensemble principles apply to node/subgraph tokenization?

### Reproducibility and Standards

1. **Benchmark Datasets**: Need for standardized benchmarks specifically evaluating tokenization quality (not just end-task performance)

2. **Evaluation Metrics**: Better metrics for tokenization quality beyond downstream task scores:
   - Morphological accuracy
   - Semantic compositionality
   - Cross-lingual consistency
   - Robustness to noise

3. **Open Implementations**: Public implementations of ensemble tokenizers for reproducibility and comparison

## References

### Primary Research Papers

- Kudo, T. (2018). Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018), pages 66-75, Melbourne, Australia. DOI: 10.18653/v1/P18-1007. arXiv:1804.10959. https://aclanthology.org/P18-1007/

- Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. EMNLP 2018. https://github.com/google/sentencepiece

- Provilkov, I., Emelianenko, D., & Voita, E. (2020). BPE-Dropout: Simple and Effective Subword Regularization. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020). arXiv:1910.13267. https://aclanthology.org/2020.acl-main.170/

- Bostrom, K., & Durrett, G. (2020). Byte Pair Encoding is Suboptimal for Language Model Pretraining. Findings of EMNLP 2020.

### Technical Documentation & Tutorials

- Hugging Face. (2023-2024). Unigram tokenization - LLM Course. https://huggingface.co/learn/llm-course/en/chapter6/7

- Hugging Face. (2024). Summary of the tokenizers. https://huggingface.co/docs/transformers/en/tokenizer_summary

- Dingwall, N. (2024). Tokenization for language modeling: Byte Pair Encoding vs Unigram Language Modeling. https://ndingwall.github.io/blog/tokenization

- Becquin, G. (2020). A Rust SentencePiece implementation. Rust NLP tales. https://guillaume-be.github.io/2020-05-30/sentence_piece

### Educational Resources

- Kernes, J. (2021). SentencePiece Tokenizer Demystified. Towards Data Science. https://towardsdatascience.com/sentencepiece-tokenizer-demystified-d0a3aac19b15

- Vaidhya, T. (2021). Divergence - Tale of Sentencepiece Library. https://tejasvaidhyadev.github.io/blog/Sentencepiece

- Machine Learning Mastery. (2024). Tokenizers in Language Models. https://machinelearningmastery.com/tokenizers-in-language-models/

### Stack Exchange & Community Resources

- Data Science Stack Exchange. (2021). Unigram tokenizer: how does it work? https://datascience.stackexchange.com/questions/88824/unigram-tokenizer-how-does-it-work

- everdark. (2023). On Subword Units: Segmentation for Natural Language Modeling. https://everdark.github.io/k9/notebooks/ml/natural_language_understanding/subword_units/subword_units.nb.html

### Software Repositories

- Google SentencePiece: https://github.com/google/sentencepiece
- BPE-Dropout Implementation: https://github.com/VProv/BPE-Dropout
- Hugging Face Tokenizers: https://github.com/huggingface/tokenizers
- Subword NMT (BPE): https://github.com/rsennrich/subword-nmt
- Unigram Tokenization from Scratch: https://github.com/DmitryAsdre/UnigramTokenization
- merge-tokenizers (Alignment): https://github.com/symanto-research/merge-tokenizers

### Related Work

- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. Proceedings of ACL 2016. arXiv:1508.07909

- Papers with Code - Unigram Segmentation: https://paperswithcode.com/method/unigram-segmentation

- Papers with Code - BPE-Dropout: https://paperswithcode.com/paper/191013267

### arXiv Preprints

- arXiv:1804.10959 - Subword Regularization
- arXiv:1910.13267 - BPE-Dropout
- arXiv:1508.07909 - Neural Machine Translation with Subword Units
- arXiv:2506.01687 - StochasTok: Improving Fine-Grained Subword Understanding
- arXiv:2404.08335 - Toward a Theory of Tokenization in LLMs
- arXiv:2507.07824 - Conditional Unigram Tokenization with Parallel Data
- arXiv:2411.17669 - Linguistic Laws Meet Protein Sequences (comparative tokenization analysis)
