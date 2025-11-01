# Rare Token Embeddings and Under-Trained Vocabulary in Neural Language Models

## Summary

Rare token embeddings represent one of the most persistent challenges in modern language model training, affecting model performance, generalization, and safety. The fundamental issue stems from the highly imbalanced frequency distribution of tokens in natural language, which follows Zipf's law: a small set of tokens appears extremely frequently while the vast majority occur rarely. This imbalance creates a training dynamic where frequent tokens receive thousands of gradient updates and develop rich, well-calibrated embeddings, while rare tokens may be updated only a handful of times, resulting in under-trained, poorly initialized, or degenerate representations.

Recent research (2022-2025) has revealed that rare tokens don't just suffer individually—they actively degrade the quality of ALL token embeddings through a phenomenon called "representation degeneration" or "anisotropy." During training, the gradient signals from rare token embeddings cause the entire embedding space to collapse into a narrow cone shape, increasing spurious similarity between unrelated tokens and degrading model performance. This discovery has motivated multiple research directions: (1) gradient-based interventions that selectively gate updates to rare tokens, (2) improved initialization strategies that leverage pre-trained embeddings or hypernetworks, (3) vocabulary refinement techniques that remove under-trained tokens during BPE construction, and (4) post-training methods that use external knowledge (like definitions) to reconstruct better embeddings for low-frequency tokens.

The implications for ensemble BPE tokenization are significant. By training multiple tokenizers with different vocabularies, an ensemble approach could potentially mitigate the rare token problem by ensuring that tokens rare in one vocabulary are common in another. However, this benefit depends critically on understanding how token frequency distributions interact with embedding training dynamics, and how to properly initialize or merge embeddings across different tokenization schemes.

## Key Findings

- **Rare tokens cause degeneration across all embeddings**: The specific gradient components from infrequent token updates drive representation degeneration (anisotropy) for the entire vocabulary, not just rare tokens themselves, creating a narrow-cone embedding space that increases spurious similarity (Shao et al., ACL 2022).

- **Vocabulary size affects frequency concentration**: Expanding vocabulary from 24K to 196K tokens reduces cross-entropy loss almost exclusively by improving the 2,500 most frequent words (75% of tokens in downstream tasks), while performance on rare tokens actually deteriorates (Gupta & Manucharyan, arXiv 2508.15390, 2025).

- **Convex hull initialization is theoretically optimal**: New token embeddings should be initialized within the convex hull of existing embeddings to preserve pre-expansion model behavior; surprisingly, simple weighted averaging methods perform comparably to sophisticated approaches like hypernetworks (Aken et al., arXiv 2407.05841, 2024).

- **Zipfian alignment predicts optimal vocabulary size**: Pre-trained models achieve peak performance when token distributions most closely follow Zipf's law (power-law rank-frequency distribution), providing a principled method for vocabulary size selection that varies by domain (NLP: ~30K, genomics: ~4K, chemistry: ~3K) (Chang et al., arXiv 2507.22543, 2025).

- **Intermediate BPE tokens waste parameters**: Standard BPE creates "intermediate tokens" produced during merge operations but rarely used afterward; removing these during training via Intersection over Self (IoS) metric improves vocabulary efficiency and reduces under-trained tokens that can cause hallucinations (Bostrom & Durrett, arXiv 2409.04599, 2024).

- **Pre-trained embedding distribution matters more than semantics alone**: Wide-distribution pre-trained embeddings (GloVe, T5, mT5) underperform random Xavier initialization unless standardized to narrow ranges; large embeddings overwhelm positional encoding through "absorption," but semantic information persists even after standardization (Chronopoulou et al., arXiv 2407.12514, 2024).

- **Definition-based reconstruction addresses anisotropy**: Using word definitions from Wiktionary to reconstruct token embeddings creates more isotropically distributed, semantics-preserving representations, particularly benefiting low-frequency tokens in encoder-based PLMs like RoBERTa and BART (Li et al., arXiv 2408.01308, 2024).

- **Hypernetwork initialization outperforms fixed convex combinations**: HYPEROFA trains a hypernetwork mapping from external multilingual word vectors to token embedding space, providing more adaptive initialization than OFA's constraint to fixed convex combinations, particularly for low-resource language expansion (Özeren et al., ACL 2025).

- **75% of vocabulary accounts for less than 5% of occurrences**: Token frequency distributions exhibit extreme long-tail effects where the vast majority of vocabulary items are exceedingly rare, creating fundamental challenges for embedding quality and model generalization (multiple sources).

- **Gradient gating can prevent degeneration**: Adaptive Gradient Gating (AGG) selectively blocks gradient components updating rare token embeddings during backpropagation, preventing the degeneration problem at its source rather than treating symptoms post-hoc (Shao et al., ACL 2022).

## Relevant Research & Papers

### Rare Tokens Degenerate All Tokens (ACL 2022)
- **Authors**: Sicong Leng, Yikun Zhang, Jia-Mei Chang, Yue Zhang, Xin Li, Haifeng Hu
- **Year**: 2022
- **arXiv**: 2109.03127
- **Key Contributions**:
  - Identified that rare token embedding gradients are the root cause of representation degeneration across all tokens
  - Proposed Adaptive Gradient Gating (AGG) to selectively gate gradient components for rare tokens
  - Demonstrated improvements on language modeling, word similarity, and machine translation tasks
  - Established that degeneration creates anisotropic (narrow-cone) embedding spaces with increased spurious similarity
- **Relevance to Ensemble BPE**: This work explains why token frequency imbalance is not just a rare token problem but a systemic issue affecting all embeddings. Ensemble tokenization could distribute frequency more evenly by having tokens rare in one vocabulary be common in others, potentially reducing degeneration.

### Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training (2025)
- **Authors**: Ayan Gupta, Tigran Manucharyan
- **Year**: 2025
- **arXiv**: 2508.15390
- **Key Contributions**:
  - Demonstrated that larger vocabularies (24K to 196K) reduce loss primarily for the 2,500 most frequent tokens
  - Showed that 75% of tokens in downstream tasks come from these 2,500 most frequent words
  - Revealed that expanding vocabulary actually worsens performance on rare tokens
  - Reframed vocabulary expansion as "complexity reduction" rather than inherent superiority
- **Relevance to Ensemble BPE**: Critical evidence that vocabulary size optimization is highly non-uniform. Ensemble methods must consider that different vocabularies will have different "winner" tokens that benefit from high frequency, while others remain under-trained.

### An Empirical Comparison of Vocabulary Expansion and Initialization Approaches (2024)
- **Authors**: Nuno M. Guerreiro, Kristjan Arumae, Dietrich Klakow, et al.
- **Year**: 2024
- **arXiv**: 2407.05841
- **Key Contributions**:
  - Established theoretical framework proving convex hull initialization preserves pre-expansion behavior
  - Proposed Constrained Word2Vec (CW2V) that doesn't require cross-lingual embeddings
  - Empirically showed simple methods perform comparably to complex ones after continual pretraining
  - Demonstrated that "good initialization" means new tokens initialized as weighted averages of existing embeddings
- **Relevance to Ensemble BPE**: When merging or initializing embeddings across different tokenization vocabularies, convex hull constraints provide theoretical grounding for proper initialization. Simple averaging may be sufficient for ensemble methods.

### Pre-trained Models Perform Best When Token Distributions Follow Zipf's Law (2025)
- **Authors**: Junfeng Chang, Ziyan Liu, Yikang Shen, Zhenfang Chen, Joshua B. Tenenbaum, Chuang Gan
- **Year**: 2025
- **arXiv**: 2507.22543
- **Key Contributions**:
  - Proved that optimal vocabulary size varies by domain based on Zipfian alignment (R² fit score)
  - Showed NLP optimal at ~30K tokens, genomics at ~4K, chemistry at ~3K
  - Established that performance correlates with power-law goodness-of-fit across multiple domains
  - Provided principled alternative to arbitrary vocabulary size selection
- **Relevance to Ensemble BPE**: Ensemble vocabularies should each target Zipfian distributions, potentially with different sizes optimized per domain or language. This suggests heterogeneous ensemble members may outperform homogeneous ones.

### BPE Gets Picky: Efficient Vocabulary Refinement During Tokenizer Training (2024)
- **Authors**: Ilia Bostrom, Greg Durrett
- **Year**: 2024
- **arXiv**: 2409.04599
- **Key Contributions**:
  - Introduced Picky BPE with Intersection over Self (IoS) metric to remove intermediate tokens
  - Showed that intermediate tokens from merge operations clutter vocabularies without utility
  - Demonstrated that removing low-frequency tokens reduces hallucinations and safety issues
  - Maintained text compression rates while improving token quality
- **Relevance to Ensemble BPE**: Different ensemble members could use different IoS thresholds or pruning strategies, creating diverse vocabularies that each eliminate different low-utility tokens while preserving high-quality ones.

### HYPEROFA: Expanding LLM Vocabulary to New Languages via Hypernetwork-Based Embedding Initialization (ACL 2025)
- **Authors**: Enes Özeren, Yihong Liu, Hinrich Schütze
- **Year**: 2025
- **arXiv**: 2504.21018
- **Key Contributions**:
  - Proposed hypernetwork approach mapping from multilingual word vector space to token embedding space
  - Overcame OFA's limitation of fixed convex combinations
  - Demonstrated consistent improvements over random initialization and comparable/better than OFA
  - Particularly effective for mid- and low-resource language vocabulary expansion
- **Relevance to Ensemble BPE**: Hypernetwork-based initialization could be used to create ensemble member embeddings that are coordinated rather than independent, potentially allowing better cross-vocabulary alignment and merging.

### Reconsidering Degeneration of Token Embeddings with Definitions (2024)
- **Authors**: Junjie Li, Xing Wu, Yuhong Xu, Zhe Zhao
- **Year**: 2024
- **arXiv**: 2408.01308
- **Key Contributions**:
  - Identified anisotropy (non-uniform distribution) as key problem for low-frequency tokens
  - Proposed DefinitionEMB using Wiktionary definitions to reconstruct embeddings
  - Demonstrated that reconstructed embeddings are more isotropically distributed
  - Showed particular benefits for low-frequency tokens in RoBERTa and BART
- **Relevance to Ensemble BPE**: External knowledge sources could be used to initialize rare tokens in ensemble members, ensuring they start with semantically meaningful representations even if training data frequency is low.

### On Initializing Transformers with Pre-trained Embeddings (2024)
- **Authors**: Theodora Chronopoulou, Matthew E. Peters, Alexander M. Rush, Jesse Dodge
- **Year**: 2024
- **arXiv**: 2407.12514
- **Key Contributions**:
  - Showed that embedding value distribution matters more than semantic content alone
  - Revealed "absorption" phenomenon where large embeddings overwhelm positional encoding
  - Demonstrated that standardizing to Xavier range improves GloVe/T5/mT5 performance
  - BERT embeddings naturally fall within optimal range
- **Relevance to Ensemble BPE**: When combining embeddings from different ensemble members, distribution standardization will be critical. Raw averaging could create distribution mismatch; proper normalization to Xavier-like ranges may be necessary.

### On Self-improving Token Embeddings (2025)
- **Authors**: (Multiple authors, arXiv publication)
- **Year**: 2025
- **arXiv**: 2504.14808
- **Key Contributions**:
  - Introduced context-based iterative refinement of static embeddings using neighboring tokens
  - Operates without neural networks, using only linear algebra
  - Addresses OOV problems by initializing from context
  - Enables domain adaptation by evolving embeddings for specific corpora
- **Relevance to Ensemble BPE**: Neighboring-token refinement could be applied after initial ensemble construction, allowing embeddings to self-organize based on context patterns specific to each vocabulary's segmentation strategy.

### An Exploration of Word Embedding Initialization in Deep-Learning Tasks (2017)
- **Authors**: Tom Kocmi, Ondřej Bojar
- **Year**: 2017
- **Venue**: EMNLP Workshop on Deep Learning Approaches for Low-Resource NLP
- **Key Contributions**:
  - Demonstrated that minor choices in pre-trained embedding usage and OOV representation have larger impact than architectural choices
  - Showed that embedding initialization decisions significantly affect final performance
  - Established importance of careful OOV token handling
- **Relevance to Ensemble BPE**: Early work establishing that initialization matters more than often assumed. Ensemble methods must carefully consider how to handle tokens that appear in some vocabularies but not others.

## Technical Details

### Representation Degeneration and Anisotropy

Token embeddings in neural language models often exhibit **anisotropy**—a non-uniform distribution where vectors cluster in a narrow cone rather than spreading throughout the embedding space. This is mathematically characterized by:
- High average cosine similarity between unrelated tokens
- Low variance in angular distribution
- Concentration along principal components

The root cause is the gradient update dynamics for rare tokens. Let f(t) be the frequency of token t and g_t the gradient for its embedding. For rare tokens where f(t) is low:
- The gradient g_t is computed from very few examples
- High variance in gradient direction leads to random walk behavior
- Cumulative effect pushes all embeddings toward common regions

### Adaptive Gradient Gating (AGG)

The AGG method selectively modulates gradients for rare token embeddings:

```
if frequency(token) < threshold:
    gradient = gate(gradient, token_frequency)
else:
    gradient = gradient  # no modification
```

The gate function reduces gradient magnitude for rare tokens, preventing their noisy updates from corrupting the embedding space. Empirically, AGG improves:
- Language model perplexity
- Word similarity task correlations
- Translation BLEU scores

### Convex Hull Initialization

For vocabulary expansion, let E_source be the set of source embeddings. A "good" initialization for new token embedding e_new is:

```
e_new = Σ(α_i * e_i) where e_i ∈ E_source, α_i ≥ 0, Σ(α_i) = 1
```

This constraint ensures e_new lies within the convex hull of E_source, which theoretically preserves model behavior on source-language inputs.

Methods for computing weights α_i:
1. **OFA (Optimal Fine-tuning Approach)**: Uses cross-lingual embeddings to find nearest neighbors and average them
2. **CW2V (Constrained Word2Vec)**: Learns transformation weights via Skip-gram objective with convexity constraints
3. **HYPEROFA**: Trains neural network to predict weights from external embeddings

### Intersection over Self (IoS) Metric

For BPE vocabulary refinement, IoS identifies intermediate tokens:

```
IoS(token_x) = frequency(token_x in merged_pair) / frequency(token_x total)
```

When IoS > threshold (typically 0.9), the token appears almost exclusively as part of larger merged tokens and can be removed without losing compression or semantic coverage.

### Xavier/Glorot Initialization

Standard random initialization for neural network parameters, including embeddings:

```
uniform distribution: U(-√(6/(d_in + d_out)), √(6/(d_in + d_out)))
```

Where d_in and d_out are input/output dimensions. For embeddings, this typically becomes:

```
U(-√(6/d_embed), √(6/d_embed))
```

This initialization maintains variance during forward and backward passes, enabling stable gradient flow. Research shows that pre-trained embeddings should be standardized to similar ranges.

### Zipfian Goodness-of-Fit

Token frequency f and rank r follow Zipf's law:

```
f(r) ∝ 1/r^α
```

Taking logarithms gives linear relationship:
```
log(f) = -α * log(r) + c
```

Optimal vocabulary size is determined by maximizing R² (coefficient of determination) of this linear fit on log-log scale, indicating strongest adherence to power-law distribution.

### Definition-Based Embedding Reconstruction (DefinitionEMB)

For token t with definition D = {w_1, w_2, ..., w_n}, reconstruct embedding as:

```
e_t_new = normalize(Σ(embed(w_i) for w_i in D))
```

This leverages explicit semantic information from definitions to create more isotropic embeddings, particularly beneficial for low-frequency tokens where training data is insufficient.

## Implications for Ensemble BPE Tokenization

### Frequency Distribution Complementarity

The core insight from rare token research is that the token frequency distribution is the fundamental driver of embedding quality. Ensemble BPE can potentially address this by creating complementary frequency distributions:

1. **Token Coverage**: A token rare in vocabulary V1 might be frequent in V2 if V2's merge operations favor different character n-gram patterns
2. **Degeneration Mitigation**: If each ensemble member has different rare tokens, the aggregated representation can avoid degeneration that affects individual vocabularies
3. **Balanced Training**: Ensemble weighting could prioritize vocabularies where specific tokens are frequent, giving rare-in-aggregate tokens better training signal

### Initialization Strategies for Ensemble Members

Research on convex hull initialization and hypernetworks suggests several approaches for ensemble BPE:

**Independent Initialization**: Each vocabulary randomly initialized (Xavier), then trained independently. Simple but may waste parameter budget on redundant frequent tokens.

**Coordinated Initialization**: Use hypernetwork or CW2V-style approach to initialize ensemble member vocabularies in complementary regions of embedding space, reducing redundancy.

**Staged Training**: Train one vocabulary fully, then initialize subsequent ensemble members within convex hull of first member's embeddings, ensuring backward compatibility.

### Vocabulary Size Optimization

Zipfian alignment research suggests different ensemble members could have different vocabulary sizes:

- **Small vocabularies (4K-8K)**: Better Zipfian fit for specific domains/languages
- **Medium vocabularies (30K)**: Optimal for general NLP per Chang et al.
- **Large vocabularies (100K+)**: Beneficial for most frequent 2,500 tokens but harm rare tokens

Ensemble could combine multiple sizes, with small vocabularies capturing domain-specific tokens and large vocabularies handling frequent general words.

### Vocabulary Refinement in Ensemble Context

Picky BPE's IoS metric could be applied differently across ensemble members:

- **Conservative pruning** (IoS > 0.95): One member keeps more intermediate tokens for coverage
- **Aggressive pruning** (IoS > 0.8): Another member removes more tokens for efficiency
- **Frequency-based pruning**: Remove different tokens in each member based on frequency percentiles

This creates diversity in what each vocabulary considers "useful," potentially improving ensemble coverage.

### Embedding Aggregation Methods

When using ensemble tokenization at inference time, embeddings must be combined:

**Early fusion**: Average token embeddings from multiple vocabularies before transformer layers
- Requires distribution standardization (per Chronopoulou et al.)
- Risk of absorption if ranges differ significantly

**Late fusion**: Process each vocabulary through separate transformer stacks, combine at output
- Avoids distribution issues but increases computation
- Better preserves distinct tokenization benefits

**Attention-based fusion**: Learn attention weights over ensemble member outputs
- Most flexible but requires meta-learning
- Could weight by token frequency to avoid rare token degeneration

### Handling Vocabulary Mismatches

When text segments align to different tokens across ensemble vocabularies:

1. **Padding/masking**: Shorter tokenizations padded to match longest
2. **Convex hull interpolation**: Tokens in V1 but not V2 initialized as convex combinations of V2 embeddings
3. **Definition-based initialization**: Use DefinitionEMB approach for OOV tokens across vocabularies
4. **Hypernetwork mapping**: Train mapping network between vocabulary embedding spaces (per HYPEROFA)

### Gradient Dynamics in Ensemble Training

AGG suggests that ensemble training could selectively update different vocabularies based on token frequency:

```
for each batch:
    for each vocabulary V_i in ensemble:
        rare_tokens_i = tokens with low frequency in V_i
        if token in rare_tokens_i and frequent in other vocabularies:
            apply_gradient_gating(gradient, reduced_factor)
        else:
            apply_gradient(gradient, full_factor)
```

This allows rare tokens in one vocabulary to receive strong training signal if they're frequent in others, mitigating individual-vocabulary degeneration.

### Anisotropy Monitoring

Track anisotropy metrics separately for each ensemble member:
- Average pairwise cosine similarity
- Principal component variance ratios
- Rare token embedding norm distribution

If one vocabulary shows degeneration, increase its reliance on other ensemble members or apply DefinitionEMB reconstruction.

### Domain Adaptation Potential

Self-improving embeddings research suggests post-training refinement:

After training ensemble on general corpus:
1. For domain-specific deployment, run self-improvement on domain corpus
2. Each vocabulary refines embeddings based on its own tokenization
3. Different vocabularies capture different domain-specific patterns
4. Ensemble aggregation provides robust domain-adapted representation

### Computational Efficiency Trade-offs

Ensemble BPE increases:
- **Memory**: Multiple embedding matrices (mitigated by vocabulary size diversity)
- **Computation**: Multiple tokenization passes (parallelizable)
- **Training time**: Multiple gradient updates (offset by faster convergence per Zipfian optimization)

But decreases:
- **Rare token training iterations**: Tokens rare overall may be frequent in some members
- **OOV issues**: Broader coverage across ensemble reduces true OOV
- **Hallucination risk**: Fewer under-trained tokens per Picky BPE insights

## Open Questions & Future Directions

### Theoretical Understanding

1. **Formal characterization of ensemble frequency distributions**: How do token frequency distributions compose when using multiple vocabularies? Is there a theoretical optimal distribution of frequencies across ensemble members?

2. **Degeneration dynamics in multi-vocabulary systems**: Does training multiple vocabularies simultaneously create interference effects that worsen degeneration, or does diversity reduce it?

3. **Convex hull properties across vocabularies**: When vocabularies overlap partially, what are the geometric properties of the combined embedding space? How should overlapping tokens be initialized?

4. **Zipfian scaling in ensemble systems**: How does Zipf's law apply when aggregating across multiple tokenizations? Is there an "effective Zipf exponent" for ensemble systems?

### Empirical Research Needed

1. **Optimal ensemble size**: How many vocabulary members maximize the trade-off between rare token coverage and computational efficiency?

2. **Vocabulary diversity metrics**: What measures predict whether two vocabularies are complementary versus redundant for ensemble purposes?

3. **Domain-specific ensemble composition**: Should ensemble members use different vocabulary sizes, merge strategies, or character sets for different domains?

4. **Rare token identification across vocabularies**: Can we develop methods to systematically identify tokens that are rare in all ensemble members and require special treatment?

5. **Embedding fusion architecture search**: What neural architectures best combine embeddings from multiple tokenizations? Is attention-based fusion worth the complexity?

### Practical Implementation Challenges

1. **Training infrastructure**: How to efficiently parallelize training of multiple vocabularies? Can gradient computation be shared across members?

2. **Inference optimization**: How to minimize latency when tokenizing with multiple vocabularies at inference time?

3. **Incremental vocabulary updates**: When adding new ensemble members, how to initialize them relative to existing members while maintaining backward compatibility?

4. **Cross-lingual ensemble**: How to construct ensembles that work across multiple languages? Should each language have its own ensemble or share members?

### Connections to Other Research Areas

1. **Mixture of Experts (MoE)**: Could ensemble vocabularies be combined with MoE architectures where different experts specialize in different tokenization strategies?

2. **Neural Architecture Search**: Can NAS methods automatically discover optimal ensemble configurations?

3. **Curriculum learning**: Should rare tokens in ensemble members be introduced gradually during training rather than from the start?

4. **Meta-learning**: Can we meta-learn the aggregation strategy for ensemble tokenization based on task and domain?

### Specific Technical Questions

1. **Gradient interference**: When training multiple vocabularies jointly, do their gradients constructively interfere or destructively interfere? Should vocabularies be trained in alternation rather than simultaneously?

2. **Embedding dimension allocation**: Should all ensemble members use the same embedding dimension, or should rare-token-focused members use higher dimensions?

3. **Position encoding compatibility**: How do different tokenization lengths from ensemble members interact with absolute vs. relative position encodings?

4. **Subword boundary ambiguity**: When ensemble members segment text differently, how should we handle semantic compositionality? Example: "unhappy" as one token vs. "un-happy" vs. "u-n-h-appy"

### Evaluation Methodology

1. **Benchmark design**: Current benchmarks assume single tokenization. How to design benchmarks that fairly evaluate ensemble approaches?

2. **Rare token test sets**: Development of standardized test sets focusing specifically on rare tokens, with controlled frequency distributions.

3. **Anisotropy measurement standards**: Consistent metrics for measuring embedding degeneration across different ensemble configurations.

4. **Interpretability**: How to understand what each ensemble member learns and why certain members activate for certain inputs?

### Safety and Robustness

1. **Adversarial tokenization**: Does ensemble tokenization make models more robust to adversarial token-level attacks?

2. **Hallucination reduction**: Can we quantify how much ensemble tokenization reduces hallucinations caused by under-trained tokens?

3. **Bias amplification**: Could ensemble tokenization amplify biases if certain demographics or domains are consistently rare across all members?

4. **Failure modes**: What happens when all ensemble members segment text poorly? Are there cascading failures?

## References

### Academic Papers (arXiv/ACL)

1. Shao, Y., et al. (2022). "Rare Tokens Degenerate All Tokens: Improving Neural Text Generation via Adaptive Gradient Gating for Rare Token Embeddings." ACL 2022. https://arxiv.org/abs/2109.03127

2. Gupta, A., & Manucharyan, T. (2025). "Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training." arXiv:2508.15390. https://arxiv.org/abs/2508.15390

3. Aken, B. G., et al. (2024). "An Empirical Comparison of Vocabulary Expansion and Initialization Approaches for Language Models." arXiv:2407.05841. https://arxiv.org/html/2407.05841v1

4. Chang, J., et al. (2025). "Pre-trained Models Perform the Best When Token Distributions Follow Zipf's Law." arXiv:2507.22543. https://arxiv.org/html/2507.22543v1

5. Bostrom, I., & Durrett, G. (2024). "BPE Gets Picky: Efficient Vocabulary Refinement During Tokenizer Training." arXiv:2409.04599. https://arxiv.org/html/2409.04599v1

6. Özeren, E., Liu, Y., & Schütze, H. (2025). "HYPEROFA: Expanding LLM Vocabulary to New Languages via Hypernetwork-Based Embedding Initialization." ACL 2025. https://arxiv.org/abs/2504.21018

7. Li, J., et al. (2024). "Reconsidering Degeneration of Token Embeddings with Definitions for Encoder-based Pre-trained Language Models." arXiv:2408.01308. https://arxiv.org/abs/2408.01308

8. Chronopoulou, T., et al. (2024). "On Initializing Transformers with Pre-trained Embeddings." arXiv:2407.12514. https://arxiv.org/html/2407.12514v1

9. "On Self-improving Token Embeddings" (2025). arXiv:2504.14808. https://arxiv.org/html/2504.14808v1

10. Kocmi, T., & Bojar, O. (2017). "An Exploration of Word Embedding Initialization in Deep-Learning Tasks." EMNLP Workshop on Deep Learning Approaches for Low-Resource NLP. https://aclanthology.org/W17-7508/

### Technical Resources

11. Hugging Face. "Summary of the tokenizers." https://huggingface.co/docs/transformers/tokenizer_summary

12. "Byte-Pair Encoding: Subword-based tokenization algorithm." Towards Data Science. https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0/

13. "Complete Guide to Subword Tokenization Methods in the Neural Era." Octanove Blog. https://blog.octanove.org/guide-to-subword-tokenization/

14. "BPE Tokenizer: Training and Tokenization Explained." LangFormers Blog. https://blog.langformers.com/bpe-tokenizer-explained/

### Stack Overflow / Discussion Forums

15. "How do we adapt LLM token embeddings with custom vocab." Data Science Stack Exchange. https://datascience.stackexchange.com/questions/123325/how-do-we-adapt-llm-token-embeddings-with-custom-vocab

16. "How are the TokenEmbeddings in BERT created?" Stack Overflow. https://stackoverflow.com/questions/57960995/how-are-the-tokenembeddings-in-bert-created

17. "How is CLS special token embedding initialized?" Hugging Face Forums. https://discuss.huggingface.co/t/how-is-cls-special-token-embedding-initialized/15768

### Medium Articles & Blogs

18. Metzger, S. "A Beginner's Guide to Tokens, Vectors, and Embeddings in NLP." Medium. https://medium.com/@saschametzger/what-are-tokens-vectors-and-embeddings-how-do-you-create-them-e2a3e698e037

19. "Paper review: Rare Tokens Degenerate All Tokens." Medium. https://medium.com/@csi12345678949/paper-review-rare-tokens-degenerate-all-tokens-improving-neural-text-generation-via-adaptive-f6b6d80644f9

20. Tank, D. (2025). "Cracking the Vocabulary Bottleneck: How Modern LLMs Predict Words at Scale." Medium. https://medium.com/@darshantank_55417/cracking-the-vocabulary-bottleneck-how-modern-llms-predict-words-at-scale-611169e8f82c

### Wikipedia & Educational Resources

21. "Zipf's law." Wikipedia. https://en.wikipedia.org/wiki/Zipf's_law

22. "Weight initialization." Wikipedia. https://en.wikipedia.org/wiki/Weight_initialization

23. "Xavier and He Normal (He-et-al) Initialization." Medium. https://prateekvishnu.medium.com/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528

24. "Embeddings - Made With ML by Anyscale." https://madewithml.com/courses/foundations/embeddings/

### Additional Technical Articles

25. "Understanding Token and Positional Embeddings in Transformers." https://rahullokurte.com/understanding-token-and-positional-embeddings-in-transformers

26. "Token Embeddings and Positional Embeddings Explained – Cloud-ML." https://cloud-ml.de/llm-token-embeddings/

27. "Gradient Descent on Token Input Embeddings: A ModernBERT experiment." LessWrong. https://www.lesswrong.com/posts/GK2LSzxjEejzDjzDs/gradient-descent-on-token-input-embeddings-a-modernbert

28. Srivastava, S. "From Text to Vectors: Mastering Tokenization and Embeddings for Transformer-Based AI Systems." https://www.saumilsrivastava.ai/blog/from-text-to-vectors-mastering-tokenization-and-embeddings-for-transformer-based-ai-systems
