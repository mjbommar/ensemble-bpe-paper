# Vocabulary Pruning and Filtering in BPE Tokenization: A Comprehensive Review

## Summary

Vocabulary pruning and filtering represent critical post-processing and in-training strategies for refining byte-pair encoding (BPE) tokenizers. Traditional BPE suffers from a fundamental flaw: its bottom-up, frequency-driven merge process inadvertently produces "intermediate" or "junk" tokens—subwords that primarily serve as stepping stones to form longer tokens but rarely appear independently in the final tokenized output. These intermediate tokens clutter vocabularies, waste model capacity, and can lead to under-trained embeddings that contribute to poor downstream performance and hallucinations.

Recent research reveals a nuanced landscape where different pruning strategies yield vastly different outcomes. Post-hoc vocabulary trimming—removing rare tokens after training—consistently fails to improve and often degrades model performance across neural machine translation tasks. In contrast, in-training refinement methods that dynamically remove intermediate tokens during the BPE learning process show promising results, maintaining or improving model quality while producing cleaner, more semantically meaningful vocabularies. Alternative approaches like tree-based tokenization with entropy-based pruning and vocabulary transfer for domain adaptation demonstrate that thoughtful vocabulary reduction can enhance both efficiency and performance when properly executed.

The research landscape indicates that vocabulary pruning is not inherently beneficial or harmful—its effectiveness depends critically on when, how, and why tokens are removed. Strategies that respect token semantics, morphological structure, and usage patterns outperform naive frequency-based thresholding applied after the fact.

## Key Findings

- **Post-hoc trimming is harmful**: Vocabulary trimming as a post-processing step (removing rare tokens after BPE training) consistently fails to improve performance and frequently causes substantial degradation in neural machine translation tasks, contradicting widespread assumptions about its benefits (Cognetta et al., 2024).

- **Intermediate tokens are pervasive**: Standard BPE vocabularies contain approximately 6-10% "junk" or "scaffold" tokens that serve primarily to construct longer tokens but rarely appear in final output, representing wasted model capacity (Lian et al., 2024; Chizhov et al., 2024).

- **In-training refinement works**: Dynamic vocabulary refinement during BPE training using metrics like Intersection over Self (IoS) successfully removes intermediate tokens while maintaining or improving translation performance and text compression rates (Chizhov et al., 2024).

- **Tree-based approaches excel at morphology**: TreeTok's combination of tree-aware BPE construction followed by entropy-based Unigram pruning achieves 37.9% morphological segmentation accuracy versus standard BPE's 19.5%, while producing more compact vocabularies (Zhu et al., 2024).

- **Domain-specific vocabulary transfer enables compression**: Fast Vocabulary Transfer achieves up to 75% vocabulary reduction with minimal performance loss by adapting pre-trained models to smaller, domain-specific tokenizers, with inference speedups of 1.07-1.40x depending on the task (Gee et al., 2024).

- **Compression correlates with performance**: Better tokenizer compression (fewer tokens per character) strongly predicts downstream model performance, with correlations of r=-0.976 to -0.996 for generation tasks, establishing compression as a meaningful proxy for tokenizer quality (Goldman et al., 2024).

- **Scaffold removal improves token distribution**: Scaffold-BPE's dynamic removal mechanism achieves 76.40% frequency improvement and gains of +0.5-0.6 BLEU in machine translation while maintaining BPE's simplicity through parameter-free modifications (Lian et al., 2024).

## Relevant Research & Papers

### BPE Gets Picky: Efficient Vocabulary Refinement During Tokenizer Training

- **Authors**: Pavel Chizhov, Catherine Arnett, Elizaveta Korotkova, Ivan P. Yamshchikov
- **Year**: 2024
- **Key Contributions**:
  - Introduces Intersection over Self (IoS) metric to identify intermediate tokens: IoS measures how often a token appears as part of a specific merge pair relative to its total occurrences
  - Removes intermediate tokens dynamically during training using IoS threshold (τ), not as post-processing
  - Maintains chronological event order of merges and removals for consistent inference
  - Achieves 2-10% token removal depending on threshold while maintaining comparable or superior COMET scores
  - Preserves text compression (0.992-0.999 relative to vanilla BPE) unlike competing methods
  - Increases proportion of meaningful word-initial tokens and improves mean token length (5.38→5.50 characters at τ=0.6)
- **Relevance to Ensemble BPE**: Demonstrates that vocabulary quality can be improved during training without post-hoc trimming, suggesting ensemble methods could incorporate IoS-based filtering as each component tokenizer is trained to ensure cleaner merged vocabularies.

### An Analysis of BPE Vocabulary Trimming in Neural Machine Translation

- **Authors**: Marco Cognetta, Tatsuya Hiraoka, Naoaki Okazaki, Rico Sennrich, Yuval Pinter
- **Year**: 2024 (Fifth Workshop on Insights from Negative Results in NLP)
- **Key Contributions**:
  - Comprehensive empirical analysis showing vocabulary trimming fails to improve performance across extensive hyperparameter configurations
  - Tested on IWSLT14 German→English and Europarl English→French datasets
  - Examined vocabulary sizes (5k-30k), trimming thresholds, split vs. joint vocabularies, and terminal subword preservation
  - Found that trimming optimal baselines typically reduces BLEU by 0.2-0.3 points, up to 0.46 in worst cases
  - Trimming only showed minimal improvements (+0.37 BLEU maximum) in severely suboptimal baseline configurations
  - Recommends against vocabulary trimming as post-processing; advocates for directly initializing smaller vocabularies instead
- **Relevance to Ensemble BPE**: Critical negative result showing that naive frequency-based filtering after tokenizer training is counterproductive. Ensemble tokenization should avoid post-hoc trimming of rare tokens from merged vocabularies, instead focusing on intelligent selection during vocabulary construction or merging phases.

### Unsupervised Morphological Tree Tokenizer (TreeTok)

- **Authors**: Qingyang Zhu, Xiang Hu, Pengyu Ji, Wei Wu, Kewei Tu (ShanghaiTech University, Ant Group)
- **Year**: 2024 (ACL 2025 Findings)
- **Key Contributions**:
  - Two-phase approach: tree-based BPE variant for vocabulary construction, followed by tree-based Unigram variant for pruning
  - MorphOverriding mechanism allows subword representations to override compositional structures when matching morpheme vocabulary
  - Counts adjacent token pairs sharing the same parent in tree structure, respecting morphological boundaries
  - Entropy-based pruning removes lowest k% of vocabulary by information entropy gain
  - Achieves 37.9% morphological segmentation accuracy vs. BPE's 19.5%
  - Improves perplexity (107.26 vs. 107.76) and reduces tokens per sentence (25.99 vs. 26.58)
  - Eliminates intermediate tokens through top-down matching, producing more compact vocabularies
- **Relevance to Ensemble BPE**: TreeTok's approach of building a large initial vocabulary then pruning based on entropy and morphological structure offers a principled alternative to frequency-based methods. Ensemble tokenizers could adopt similar hierarchical construction with structured pruning to ensure component vocabularies are morphologically coherent.

### Scaffold-BPE: Enhancing Byte Pair Encoding with Simple and Effective Scaffold Token Removal

- **Authors**: Haoran Lian, Yizhe Xiong, Jianwei Niu (Beihang University, Tsinghua University)
- **Year**: 2024
- **Key Contributions**:
  - Identifies "scaffold tokens" as intermediate tokens appearing infrequently on their own (6.07% of 32K vocabulary on Pile dataset)
  - Dynamic detection during training: marks tokens as scaffolds when frequency falls below next candidate pair
  - Two-phase encoding: (1) Scaffolding uses all tokens for merging, (2) Demolishing decomposes scaffolds into shortest non-scaffold sequences
  - Parameter-free, computation-light, O(1) demolishing time per token
  - Achieves +0.5 BLEU on WMT'14 EN-DE, +0.6 on EN-FR
  - 76.40% frequency improvement for 32K vocabularies
  - Produces embeddings with better uniformity and higher Shannon entropy
- **Relevance to Ensemble BPE**: Scaffold-BPE's lightweight detection mechanism could be applied to ensemble vocabularies to identify and filter low-utility intermediate tokens. The two-phase encoding approach (building with scaffolds, removing for output) might inform ensemble strategies that use different vocabularies during training versus inference.

### Fast Vocabulary Transfer for Language Model Compression

- **Authors**: Leonidas Gee, Andrea Zugarini, Leonardo Rigutini, Paolo Torroni
- **Year**: 2024
- **Key Contributions**:
  - Vocabulary Transfer (VT) adapts pre-trained models to domain-specific, smaller tokenizers
  - Simple initialization: copy embeddings for shared tokens, average pre-trained embeddings for new tokens
  - Achieves up to 75% vocabulary reduction with limited performance drops
  - Combined with knowledge distillation: up to 55% overall model size reduction
  - Inference speedups: 1.40x (medical), 1.21x (legal), 1.07x (news)
  - Domain specialization significantly impacts effectiveness—more specialized domains show better compression-to-performance ratios
  - 15%+ parameter reduction with minimal F1 loss; surprisingly, reducing vocabulary size sometimes improves performance
- **Relevance to Ensemble BPE**: Demonstrates that vocabulary size is a tunable hyperparameter and domain-specific vocabularies can be more efficient than large general-purpose ones. Ensemble approaches could leverage vocabulary transfer to create domain-specialized component tokenizers that collectively provide broad coverage while maintaining compression efficiency.

### Unpacking Tokenization: Evaluating Text Compression and its Correlation with Model Performance

- **Authors**: Goldman, Caciulariu, Eyal, Cao, Szpektor, Tsarfaty (Bar-Ilan University, Google Research/DeepMind)
- **Year**: 2024
- **Key Contributions**:
  - Demonstrates strong correlation between tokenizer compression and downstream model performance
  - Controlled compression by training BPE with varying amounts of supporting documents (1M down to 0)
  - Poor tokenizers produced texts 121% longer (1-doc) to 323% longer (character-level)
  - Generation tasks show stronger correlation (r=-0.976 to -0.996) than classification (r=-0.710 to -0.714)
  - Smaller models suffer disproportionately from poor compression
  - Compression differences concentrate in rare words; frequent terms show minimal divergence
  - Frames compression as "0-gram language modeling," establishing it as a reasonable proxy for tokenizer quality
- **Relevance to Ensemble BPE**: Provides theoretical and empirical justification for optimizing tokenizer compression as a primary objective. Ensemble methods should measure and optimize for compression efficiency across the merged vocabulary, particularly focusing on rare word handling where compression differences are most pronounced.

### Thunder-Tok: Minimizing Tokens per Word in Tokenizing Korean Texts for Generative Language Models

- **Authors**: Not fully extracted in search
- **Year**: 2024
- **Key Contributions**:
  - Korean-specific tokenizer reducing fertility (tokens per word) by ~10% compared to BPE
  - Achieves 10% inference speed improvement without performance compromise
  - Demonstrates language-specific optimization can improve compression and efficiency
- **Relevance to Ensemble BPE**: Shows that tokenizer optimization for specific language characteristics (morphology, writing systems) can yield significant efficiency gains. Ensemble approaches could incorporate language-specific component tokenizers optimized for different linguistic features.

### BytePiece: Novel Training Algorithm Based on Byte-based N-gram Language Model

- **Authors**: Not fully detailed in search
- **Year**: Recent (referenced in 2024 searches)
- **Key Contributions**:
  - Uses Byte-based N-gram Language Model (BNLM) for training
  - Achieves higher compression rates than existing tokenizers
  - Supports multiprocessing acceleration
- **Relevance to Ensemble BPE**: Alternative training algorithm that could inform ensemble vocabulary construction methods, particularly for optimizing compression through n-gram modeling rather than pure frequency-based approaches.

## Technical Details

### Intersection over Self (IoS) Metric

The IoS metric from "BPE Gets Picky" identifies intermediate tokens by calculating how often a token appears as part of a specific merge pair relative to its total occurrences:

**Formula**: For a merge pair (x₁, x₂) creating token x₁₂, the IoS for x₁ is:
```
IoS(x₁) = frequency(x₁ in pair (x₁, x₂)) / total_frequency(x₁)
```

**Interpretation**: When IoS ≈ 1, token x₁ almost exclusively appears as part of forming x₁₂, indicating it's an intermediate token with little independent utility.

**Threshold-based Removal**: Tokens with IoS above threshold τ (typically 0.5-0.8) are removed from the vocabulary during training, with their component tokens used instead.

**Implementation**: Maintains chronological event order of merges and removals to ensure consistent tokenization during inference.

### Scaffold Token Detection Algorithm

Scaffold-BPE identifies intermediate tokens dynamically during training:

1. **During BPE training**: When merging pair (a,b) into token t, frequencies of a and b decrease
2. **Scaffold marking**: If frequency(a) or frequency(b) drops below the frequency of the next candidate pair at the head of the priority queue, mark as scaffold
3. **Two-phase encoding**:
   - **Scaffolding phase**: Use all tokens (including scaffolds) to merge according to vocabulary rank
   - **Demolishing phase**: Decompose scaffold tokens into shortest non-scaffold component sequences

**Complexity**: O(1) time for demolishing individual tokens, making it computationally efficient.

### Tree-Based BPE Construction and Entropy Pruning

TreeTok's two-phase approach:

**Phase 1: Tree-Based BPE Variant**
- Counts adjacent token pairs that share the same parent in the tree structure
- Merges pairs exceeding a threshold, respecting morphological boundaries
- Only merges siblings in the induced parse tree, preventing cross-morpheme combinations
- Produces initial large vocabulary with morphological awareness

**Phase 2: Tree-Based Unigram Pruning**
- Calculates information entropy gains for each vocabulary entry
- Iteratively removes the lowest k% from vocabulary based on entropy contribution
- Preserves meaningful morphemes while eliminating intermediate tokens
- Uses "pruned version of deep inside encoder" with linear space complexity

**Dual Self-Supervised Objectives**:
1. Auto-encoding loss: predicting masked morphemes
2. Auto-regression loss: next-token prediction leveraging sentence context

### Unigram Language Model Pruning

Standard Unigram approach:

1. **Initialize**: Start with large vocabulary of potential subwords
2. **Compute loss**: Calculate corpus loss L(vocabulary) under unigram language model
3. **Evaluate removals**: For each symbol s, compute ΔL(s) = L(vocabulary \ {s}) - L(vocabulary)
4. **Prune**: Remove 10-20% of symbols with smallest ΔL (those whose removal increases loss least)
5. **Iterate**: Repeat until desired vocabulary size reached

**Advantage**: Makes intermediate tokens less likely to survive since they contribute little to the unigram language model likelihood.

### Vocabulary Transfer Embedding Initialization

Fast Vocabulary Transfer's simple but effective approach:

1. **Shared tokens**: Directly copy embeddings from pre-trained model
2. **New tokens**: Initialize as weighted average of embeddings for the token's decomposition under the original tokenizer
   ```
   emb(new_token) = average([emb(t) for t in original_tokenizer(new_token)])
   ```
3. **Optional refinement**: One epoch of Masked Language Modeling on in-domain data before fine-tuning

**Key insight**: Preserves pre-trained knowledge while accommodating domain-specific vocabulary.

### Compression Metrics

Several compression metrics are used across the literature:

- **Fertility**: Average number of tokens per word (lower is better)
- **Compression ratio**: Character-to-token ratio or percentage reduction in sequence length
- **Relative compression**: Ratio of tokenized length to baseline (e.g., vanilla BPE)
- **Tokens per sentence**: Direct measure of sequence length

**Correlation with performance**: Strong negative correlation (r=-0.976 to -0.996) for generation tasks indicates better compression predicts better model performance.

## Implications for Ensemble BPE Tokenization

### Avoid Post-Hoc Trimming of Merged Vocabularies

The strong negative results from Cognetta et al. (2024) indicate that after merging vocabularies from multiple BPE tokenizers, applying frequency-based trimming will likely harm performance. Instead:

- Perform vocabulary filtering during component tokenizer training (using methods like IoS or scaffold detection)
- Select high-quality tokens during the ensemble merging process rather than trimming afterward
- Consider compression metrics and token utility rather than raw frequency when deciding which tokens to include

### Incorporate Quality Metrics During Vocabulary Construction

Ensemble BPE training should integrate quality metrics to produce cleaner component vocabularies:

- **IoS metric**: Filter intermediate tokens during individual tokenizer training
- **Morphological alignment**: Consider tree-based or structure-aware merging strategies
- **Compression efficiency**: Optimize for tokens that improve compression on target corpora
- **Embedding quality**: Monitor embedding norm distributions to identify under-trained tokens

### Leverage Complementary Vocabulary Structures

Different component tokenizers could be optimized for different objectives:

- **Compression-optimized tokenizer**: Maximize text compression using techniques from Goldman et al.
- **Morphology-optimized tokenizer**: Use TreeTok-style construction for morphologically rich languages
- **Domain-specific tokenizers**: Apply vocabulary transfer to create specialized components for different domains
- **Frequency-balanced tokenizer**: Use Scaffold-BPE to ensure uniform token frequency distributions

The ensemble would then select the most appropriate tokenization based on context, domain, or frequency characteristics.

### Dynamic Vocabulary Refinement During Ensemble Training

Rather than training component tokenizers independently and merging:

1. **Iterative refinement**: Train component tokenizers with periodic cross-vocabulary quality checks
2. **Shared quality metrics**: Apply consistent IoS or scaffold detection across all components
3. **Coordinated pruning**: Remove intermediate tokens that appear across multiple component vocabularies
4. **Entropy-based selection**: During merging, use entropy gains (à la Unigram) to select tokens from component vocabularies

### Optimize for Compression as Primary Objective

Given the strong correlation between compression and performance:

- Measure compression efficiency of the merged ensemble vocabulary
- Prioritize tokens that improve compression, especially for rare words where differences are most pronounced
- Consider using compression as a selection criterion when multiple component tokenizers provide overlapping coverage
- Monitor task-specific compression needs (generation vs. classification require different optimization)

### Handle Intermediate Tokens Across Ensemble Components

Intermediate tokens may appear in some component vocabularies but not others:

- **Cross-vocabulary detection**: A token might be intermediate in one component but meaningful in another
- **Ensemble-level IoS**: Calculate IoS across the merged vocabulary to identify globally intermediate tokens
- **Selective removal**: Keep tokens that are meaningful in any component vocabulary
- **Two-phase encoding**: Adopt Scaffold-BPE's approach of using different vocabularies for training vs. inference

### Domain Adaptation Through Vocabulary Transfer

Ensemble approaches could incorporate vocabulary transfer:

1. Start with general-purpose pre-trained component tokenizers
2. Apply vocabulary transfer to adapt each component to different domains
3. Ensemble provides broad coverage while maintaining domain-specific efficiency
4. Reduces overall vocabulary size while improving domain performance

### Model Size Considerations

Smaller models suffer more from poor compression:

- Ensemble tokenizers for smaller models should prioritize compression optimization
- Larger models may tolerate more vocabulary diversity
- Adjust ensemble strategy based on target model size

### Quality Metrics for Ensemble Evaluation

Beyond standard downstream task performance, ensemble tokenizers should be evaluated on:

- **Compression efficiency**: Tokens per character, fertility rates
- **Morphological alignment**: Accuracy on morphological segmentation tasks
- **Token frequency distribution**: Shannon entropy, uniformity metrics
- **Intermediate token ratio**: Percentage of vocabulary that are scaffolds/junk tokens
- **Embedding quality**: Norm distributions, representational uniformity
- **Domain coverage**: Performance across specialized vs. general domains

## Open Questions & Future Directions

### Optimal Ensemble Merging Strategies with Pruning

How should ensemble tokenizers merge multiple component vocabularies while filtering intermediate tokens?

- Should filtering happen before merging, during merging, or after?
- How do we identify ensemble-level intermediate tokens that might not be intermediate in individual components?
- What metrics best predict which tokens from overlapping vocabularies should be retained?

### Interaction Between Ensemble Diversity and Vocabulary Quality

There may be tension between ensemble diversity and vocabulary cleanliness:

- Does filtering intermediate tokens reduce the diversity benefits of ensemble approaches?
- Can we maintain morphological, compression, and frequency diversity while removing junk tokens?
- How do we measure and optimize the quality-diversity tradeoff?

### Dynamic Pruning During Ensemble Inference

Could ensemble tokenizers adaptively select vocabularies based on input characteristics?

- Use compression-optimized vocabulary for rare words, morphology-optimized for complex morphology
- Switch vocabularies based on domain detection
- Learn to select optimal component tokenizer based on input features

### Theoretical Foundations for Vocabulary Pruning in Ensemble Context

Most pruning research focuses on single tokenizers:

- What is the theoretical impact of vocabulary pruning on ensemble tokenizer performance?
- How does pruning affect ensemble diversity metrics?
- Can we develop formal criteria for when pruning helps vs. harms in ensemble settings?

### Cross-Lingual Ensemble Tokenizers with Pruning

How do pruning strategies interact with multilingual tokenizers?

- Do intermediate tokens manifest differently across languages?
- Should pruning be language-specific or cross-lingual?
- How does TreeTok's morphological approach extend to typologically diverse languages?

### Compression-Morphology Tradeoff

Goldman et al. show compression correlates with performance, but TreeTok shows morphological alignment also improves performance:

- Is there a tradeoff between compression efficiency and morphological accuracy?
- Can ensemble approaches optimize both simultaneously by having specialized components?
- How do we balance these objectives when they conflict?

### Scalability of Quality-Based Pruning

Methods like IoS calculation and scaffold detection add computational overhead:

- How do these methods scale to very large vocabularies (100K+)?
- Can pruning be parallelized effectively for ensemble training?
- What are the computational tradeoffs between quality pruning and training larger vocabularies?

### Transfer Learning with Pruned Ensemble Vocabularies

How does vocabulary transfer interact with ensemble tokenization?

- Can we transfer from a large ensemble tokenizer to smaller domain-specific ensembles?
- How do we initialize embeddings when component vocabularies have been pruned differently?
- Does pruning before or after transfer yield better results?

### Negative Results and Best Practices

More negative results like Cognetta et al. are needed:

- Which seemingly reasonable pruning strategies actually harm performance?
- Under what conditions (if any) does post-hoc trimming help?
- What are the failure modes of in-training pruning methods?

### Integration with Neural Architecture

How should vocabulary pruning strategies inform model architecture?

- Should models have different embedding dimensions for scaffolds vs. meaningful tokens?
- Can we use pruning signals to guide attention mechanisms?
- How do pruned vocabularies interact with subword regularization during training?

## References

### Primary Research Papers

1. Chizhov, P., Arnett, C., Korotkova, E., & Yamshchikov, I. P. (2024). BPE Gets Picky: Efficient Vocabulary Refinement During Tokenizer Training. arXiv:2409.04599v1. https://arxiv.org/html/2409.04599v1

2. Cognetta, M., Hiraoka, T., Okazaki, N., Sennrich, R., & Pinter, Y. (2024). An Analysis of BPE Vocabulary Trimming in Neural Machine Translation. In Fifth Workshop on Insights from Negative Results in NLP. https://arxiv.org/html/2404.00397 | https://aclanthology.org/2024.insights-1.7/

3. Zhu, Q., Hu, X., Ji, P., Wu, W., & Tu, K. (2024). Unsupervised Morphological Tree Tokenizer. In ACL 2025 Findings. arXiv:2406.15245. https://arxiv.org/html/2406.15245v1 | https://github.com/martianmartina/TreeTokenizer

4. Lian, H., Xiong, Y., Niu, J., et al. (2024). Scaffold-BPE: Enhancing Byte Pair Encoding with Simple and Effective Scaffold Token Removal. arXiv:2404.17808v1. https://arxiv.org/html/2404.17808v1

5. Gee, L., Zugarini, A., Rigutini, L., & Torroni, P. (2024). Fast Vocabulary Transfer for Language Model Compression. arXiv:2402.09977v1. https://arxiv.org/html/2402.09977v1

6. Goldman, Caciulariu, Eyal, Cao, Szpektor, & Tsarfaty. (2024). Unpacking Tokenization: Evaluating Text Compression and its Correlation with Model Performance. arXiv:2403.06265v1. https://arxiv.org/html/2403.06265v1

### Additional Resources

7. Towards Data Science. (2024). Byte-Pair Encoding: Subword-based tokenization algorithm. https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0/

8. Octanove Blog. Complete Guide to Subword Tokenization Methods in the Neural Era. https://blog.octanove.org/guide-to-subword-tokenization/

9. Hugging Face. (2024). Summary of the tokenizers. https://huggingface.co/docs/transformers/tokenizer_summary

10. Hugging Face. (2024). Byte-Pair Encoding tokenization - LLM Course. https://huggingface.co/learn/llm-course/en/chapter6/5

11. Hugging Face. (2024). Unigram tokenization - LLM Course. https://huggingface.co/learn/llm-course/en/chapter6/7

12. Google. SentencePiece: Unsupervised text tokenizer for Neural Network-based text generation. https://github.com/google/sentencepiece

13. Medium. (2024). Word tokenization as compression. https://medium.com/@matti.kwan/word-tokenization-as-compression-2540260f6eda

14. Medium. (2024). Token Efficiency and Compression Techniques in Large Language Models. https://medium.com/@anicomanesh/token-efficiency-and-compression-techniques-in-large-language-models-navigating-context-length-05a61283412b

15. BytePiece GitHub Repository. https://github.com/bojone/bytepiece/blob/main/README_en.md

16. Wikipedia. Byte pair encoding. https://en.wikipedia.org/wiki/Byte_pair_encoding

17. Dingwall, N. Tokenization for language modeling: Byte Pair Encoding vs Unigram Language Modeling. https://ndingwall.github.io/blog/tokenization

18. Data Science Stack Exchange. Unigram tokenizer: how does it work? https://datascience.stackexchange.com/questions/88824/unigram-tokenizer-how-does-it-work

### Related Work (Not Fully Analyzed)

19. Thunder-Tok: Minimizing Tokens per Word in Tokenizing Korean Texts for Generative Language Models. arXiv:2506.15138. https://arxiv.org/html/2506.15138

20. MorphPiece: A Linguistic Tokenizer for Large Language Models. arXiv:2307.07262. https://arxiv.org/html/2307.07262v2

21. T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings. arXiv:2406.19223. https://arxiv.org/html/2406.19223v1

22. StochasTok: Improving Fine-Grained Subword Understanding in LLMs. arXiv:2506.01687. https://arxiv.org/html/2506.01687

23. Tokenization Is More Than Compression. arXiv:2402.18376. https://arxiv.org/abs/2402.18376

24. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. arXiv:1508.07909. https://arxiv.org/pdf/1508.07909
