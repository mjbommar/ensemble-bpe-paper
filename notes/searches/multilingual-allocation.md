# Multilingual Vocabulary Allocation and Cross-Lingual Token Sharing: A Comprehensive Research Review

## Summary

Multilingual vocabulary allocation and cross-lingual token sharing represent critical challenges in developing fair and efficient multilingual language models. The fundamental problem is that traditional frequency-based tokenization algorithms like Byte Pair Encoding (BPE) inherently favor high-resource languages, resulting in severe computational and performance disparities across languages. Recent research has revealed that these disparities—often manifesting as 3-5x higher token counts for low-resource languages—translate directly into increased training costs, reduced context capacity, and diminished model performance for underrepresented languages.

The field has converged on several key insights: (1) vocabulary overlap between languages can facilitate cross-lingual transfer when semantically meaningful, but the quality of overlap matters more than quantity; (2) language-specific vocabulary allocation significantly outperforms uniform allocation strategies; and (3) structural constraints in tokenization (particularly whitespace pre-tokenization) introduce systematic biases that affect languages differently based on their orthographic conventions.

Emerging solutions include parity-aware algorithms that balance compression rates across languages, parallel tokenizers that explicitly align semantically equivalent tokens, vocabulary capacity allocation methods that tailor vocabulary sizes to linguistic characteristics, and novel pre-tokenization strategies that eliminate language-biased constraints. These advances demonstrate that careful tokenizer design can substantially improve multilingual model fairness without sacrificing overall performance.

## Key Findings

- **Token inequality is pervasive and costly**: Low and medium-resource languages experience 3-5x higher tokenization rates (fertility) compared to high-resource languages, translating to proportionally higher computational costs during training and inference, and reduced effective context window utilization (Arnett et al., 2025; Foroutan et al., 2024).

- **Vocabulary allocation drives task-specific performance**: Language-specific token coverage in multilingual vocabularies significantly impacts word-level tasks (POS tagging, dependency parsing), while sentence-level tasks (NER, NLI, cross-lingual retrieval) benefit from vocabulary sharing across languages (Limisiewicz et al., 2023).

- **Semantic similarity of shared tokens is critical**: Token overlap enables embedding spaces to capture cross-lingual semantic relationships, with models containing any overlap consistently outperforming disjoint-vocabulary models on zero-shot transfer tasks. However, semantically meaningful overlap provides greater benefits, especially for linguistically distant language pairs (Kallini et al., 2025).

- **Whitespace pre-tokenization creates systematic bias**: Different languages use varying amounts of whitespace, causing whitespace-based pre-tokenization boundaries to disproportionately constrain tokenization for some languages. SuperBPE, which allows cross-whitespace merges, reduces token premiums and achieves 27% inference compute reduction (SuperBPE, 2025).

- **Parity-aware algorithms achieve fairness without sacrificing performance**: Parity-aware BPE reduces the Gini coefficient from 0.064 to 0.011 across 30 languages while maintaining competitive global compression and negligible downstream accuracy changes (median +0.19 percentage points) (Foroutan et al., 2024).

- **Fertility is an incomplete metric**: While widely used, fertility (average tokens per word) obscures vocabulary allocation patterns across languages and domains. The Single Token Retention Rate (STRR)—measuring the proportion of words preserved as single tokens—provides better visibility into distributional fairness and systematic language prioritization (Nayeem et al., 2025).

- **Language-specific vocabulary sizing is optimal**: Assigning different vocabulary sizes to different languages based on linguistic characteristics drastically reduces token count disparities compared to uniform sizing (p<0.001), with some languages benefiting more from increased allocation than others (Arnett et al., 2025; Zheng et al., 2021).

## Relevant Research & Papers

### Tokenization Impacts Multilingual Language Modeling: Assessing Vocabulary Allocation and Overlap Across Languages
- **Authors/Source**: Tomasz Limisiewicz, Jiří Balhar, David Mareček
- **Year**: 2023 (ACL Findings)
- **arXiv ID**: 2305.17179
- **Key Contributions**:
  - Developed novel criteria to evaluate lexical representation quality and vocabulary overlap in subword tokenizers
  - Demonstrated that vocabulary overlap effects are task-dependent: detrimental for word-level tasks (POS, dependency parsing) but beneficial for sentence-level tasks (NER, NLI, cross-lingual retrieval)
  - Showed that language-specific token coverage in multilingual vocabularies significantly impacts word-level task performance
  - Provided empirical guidelines for selecting tokenizers based on intended downstream applications
- **Relevance to Ensemble BPE**: Suggests ensemble tokenizers should consider task-specific optimization and that combining language-specific and shared vocabularies could balance the trade-offs between word-level and sentence-level task performance.

### Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization
- **Authors/Source**: Negar Foroutan, Clara Meister, Debjit Paul, Joel Niklaus, Sina Ahmadi, Antoine Bosselut, Rico Sennrich
- **Year**: 2024
- **arXiv ID**: 2508.04796
- **Key Contributions**:
  - Introduced parity-aware BPE, which performs "fair-max" updates that progressively equalize compression rates across languages
  - At each merge step, identifies the language with worst compression and selects the merge that most benefits that language
  - Achieved dramatic Gini coefficient reduction from 0.064 to 0.011 across 30 languages with 128K vocabulary
  - Demonstrated higher vocabulary utilization for low/medium-resource languages without performance degradation (median +0.19pp)
  - Introduced Tokenizer Fairness Gini Coefficient as a key metric for multilingual tokenization equality
- **Relevance to Ensemble BPE**: The fair-max selection strategy could inform ensemble weight allocation or vocabulary merging strategies. An ensemble approach could incorporate parity-aware objectives alongside other optimization criteria.

### Explaining and Mitigating Crosslingual Tokenizer Inequities
- **Authors/Source**: Arnett, C., Chang, T. A., Biderman, S., Bergen, B. K.
- **Year**: 2025
- **arXiv ID**: 2510.21909
- **Key Contributions**:
  - Identified three primary drivers of token premiums: data similarity (R²=0.239), mean token length in evaluation corpora (R²=0.168), and whitespace pre-tokenization (R²=0.157)
  - Trained ~7,000 monolingual tokenizers across 97 languages to isolate effects
  - Demonstrated that language-specific vocabulary sizing drastically reduces character-per-token disparities (F-test: p<0.001)
  - Introduced SuperBPE tokenizers that remove whitespace pre-tokenization boundaries, reducing both average token counts and cross-language variance
  - Showed BPE consistently outperforms Unigram across languages
- **Relevance to Ensemble BPE**: Provides empirical evidence for language-specific vocabulary sizing in ensemble components. SuperBPE's two-phase approach (subword then superword) could inspire ensemble architectures with hierarchical tokenization strategies.

### False Friends Are Not Foes: Investigating Vocabulary Overlap in Multilingual Language Models
- **Authors/Source**: Julie Kallini, Dan Jurafsky, Christopher Potts, Martijn Bartelds
- **Year**: 2025
- **arXiv ID**: 2509.18750
- **Key Contributions**:
  - Conducted controlled experiments with systematically varied vocabulary overlap settings across bilingual models
  - Demonstrated that models with any overlap outperform disjoint-vocabulary models on cross-lingual benchmarks (XNLI, XQuAD)
  - Showed overlap enables embedding spaces to capture cross-lingual semantic relationships
  - Found that semantic similarity of shared tokens matters more for linguistically distant languages
  - Concluded that substantial shared vocabulary remains beneficial for multilingual tokenizers
- **Relevance to Ensemble BPE**: Suggests ensemble strategies should promote controlled overlap rather than completely disjoint vocabularies. Semantic similarity could inform how ensemble components are combined or weighted.

### Beyond Fertility: Analyzing STRR as a Metric for Multilingual Tokenization Evaluation
- **Authors/Source**: Mir Tafseer Nayeem, Sawsan Alqahtani, Md Tahmid Rahman Laskar, Tasnim Mohiuddin, M Saiful Bari
- **Year**: 2025
- **arXiv ID**: 2510.09947
- **Key Contributions**:
  - Introduced Single Token Retention Rate (STRR), measuring proportion of words preserved as single tokens
  - Demonstrated that fertility obscures vocabulary allocation patterns across languages and domains
  - Analyzed 6 tokenizers across 7 languages and 2 domains, revealing systematic English prioritization, strong Chinese support, and Hindi fragmentation
  - Provided more interpretable diagnostics for fairness and efficiency than fertility alone
- **Relevance to Ensemble BPE**: STRR could serve as an evaluation metric for ensemble tokenizers, helping identify whether ensemble approaches improve single-token retention for underrepresented languages. Could guide ensemble composition strategies.

### Allocating Large Vocabulary Capacity for Cross-Lingual Language Model Pre-Training
- **Authors/Source**: Bo Zheng, Li Dong, Shaohan Huang, Saksham Singhal, Wanxiang Che, Ting Liu, Xia Song, Furu Wei
- **Year**: 2021 (EMNLP)
- **Key Contributions**:
  - Proposed VOCAP algorithm for principled vocabulary capacity distribution across languages
  - Used language-specific factors (character frequency, morphological complexity, corpus representation) to determine optimal vocabulary sizes
  - Demonstrated that equal vocabulary allocation is suboptimal for multilingual models
  - Showed tailored capacity distribution improves both efficiency and downstream task performance
- **Relevance to Ensemble BPE**: VOCAP provides a methodological framework for determining language-specific vocabulary sizes in ensemble components. Could inform how to allocate capacity across different tokenizers in an ensemble.

### Parallel Tokenizers: Rethinking Vocabulary Design for Cross-Lingual Transfer
- **Authors/Source**: Multiple authors
- **Year**: 2024
- **arXiv ID**: 2510.06128
- **Key Contributions**:
  - Addressed problem where semantically equivalent words receive distinct embeddings (e.g., "I eat rice" vs "Ina cin shinkafa")
  - Proposed framework training tokenizers monolingually then exhaustively aligning vocabularies using bilingual dictionaries or word-to-word translation
  - Ensures consistent indices for semantically equivalent words across languages
  - Demonstrated that semantic similarity of tokens is crucial for effective cross-lingual transfer
- **Relevance to Ensemble BPE**: Directly relevant to ensemble approaches. Suggests ensemble components could be monolingual tokenizers with explicit alignment mechanisms. Vocabulary alignment could be a key ensemble integration strategy.

### Overlap-based Vocabulary Generation Improves Cross-lingual Transfer Among Related Languages
- **Authors/Source**: Vaidehi Patil, Partha Talukdar, Sunita Sarawagi
- **Year**: 2022 (ACL)
- **arXiv ID**: 2203.01976
- **Key Contributions**:
  - Introduced OBPE (Overlap BPE), modifying BPE to enhance overlap across related languages
  - Leveraged lexical overlap among language families to overcome corpus limitations for low-resource languages
  - Increased representation of low-resource languages via tokens shared with high-resource languages
  - Demonstrated effectiveness through extensive experiments on multiple NLP tasks and datasets
- **Relevance to Ensemble BPE**: OBPE's explicit overlap optimization could inform how ensemble tokenizers share or merge vocabularies. Language family clustering could guide ensemble component organization.

### SuperBPE: Space Travel for Language Models
- **Authors/Source**: Multiple authors
- **Year**: 2025
- **arXiv ID**: 2503.13423
- **Key Contributions**:
  - Introduced two-phase tokenization: Stage 1 learns subwords (with whitespace pre-tokenization), Stage 2 learns superwords (without whitespace restrictions)
  - Allows creation of tokens that bridge multiple words, eliminating systematic bias from whitespace differences across languages
  - Achieved +4.0% absolute improvement over BPE baseline across 30 downstream tasks (+8.2% on MMLU)
  - Reduced inference compute by 27% while improving performance
  - Significantly reduced cross-lingual token premium effects
- **Relevance to Ensemble BPE**: The two-phase approach could inspire ensemble architectures with hierarchical tokenization levels. Could combine subword-focused and superword-focused tokenizers in an ensemble for optimal coverage.

### Trans-Tokenization and Cross-lingual Vocabulary Transfers
- **Authors/Source**: Multiple authors
- **Year**: 2024
- **arXiv ID**: 2408.04303v1
- **Key Contributions**:
  - Proposed novel cross-lingual vocabulary transfer strategy for adapting high-resource LLMs to new target languages
  - Initializes token embeddings using weighted average of semantically similar source language embeddings
  - Developed Hydra LLM achieving state-of-the-art zero-shot machine translation for Tatar without parallel data
  - Particularly significant for low-resource language adaptation
- **Relevance to Ensemble BPE**: Vocabulary transfer mechanisms could enable ensemble components to share learned representations. Suggests ensemble tokenizers could leverage cross-lingual initialization strategies.

## Technical Details

### Key Metrics for Multilingual Tokenization

1. **Fertility**: Average number of tokens per word. Widely used but obscures distributional patterns.
   - Formula: Total tokens / Total words
   - Limitation: Token-level average that hides vocabulary allocation inequities

2. **Single Token Retention Rate (STRR)**: Proportion of words preserved as single tokens
   - More interpretable than fertility for fairness diagnostics
   - Better reveals systematic language prioritization

3. **Tokenizer Fairness Gini Coefficient**: Measures token-cost inequality across languages
   - Near 0 = equal treatment; Near 1 = highly unequal
   - Parity-aware BPE: 0.064 → 0.011

4. **Compression Rate (CR)**: Per-language ratio of input length to token count
   - Directly comparable across languages
   - Reveals computational cost disparities

5. **Character-per-Token (CTC)**: Characters encoded per token
   - Lower CTC = higher tokenization cost
   - Key metric for token premium analysis

6. **Vocabulary Utilization**: Proportion of vocabulary actually used by a language
   - Reveals allocation efficiency
   - Important for understanding wasted capacity

7. **Parity Score**: Cross-lingual metric comparing token counts across two languages
   - Best score when counts are identical
   - Used in parity-aware algorithms

8. **Token Overlap**: Percentage of shared vocabulary between language pairs
   - Important for cross-lingual transfer
   - Semantic quality matters more than quantity

### Algorithms and Techniques

#### Parity-Aware BPE
```
At each merge iteration:
1. Compute compression rate for each language
2. Identify language with worst (lowest) compression rate
3. Select merge operation that most improves that language's compression
4. Apply merge globally across all languages
5. Repeat until target vocabulary size reached
```

Key innovation: "Fair-max" selection instead of greedy global frequency maximization.

Variants:
- **Parity-aware Hybrid**: Alternates between parity-aware and classical BPE
- **Parity-aware Window**: Uses parity-aware selection within sliding windows of merge operations

#### VOCAP (Vocabulary Capacity Allocation)
```
For each language L:
1. Analyze linguistic characteristics:
   - Character/morpheme frequency distributions
   - Morphological complexity
   - Corpus representation in training data
2. Compute required vocabulary capacity V_L
3. Allocate vocabulary size proportional to V_L
4. Train language-specific or weighted multilingual tokenizer
```

Principle: Different languages need different vocabulary sizes for comparable compression.

#### SuperBPE (Two-Phase Tokenization)
```
Phase 1 (Subword Learning):
- Apply whitespace pre-tokenization
- Run BPE until vocabulary size V_1
- Result: Standard subword vocabulary

Phase 2 (Superword Learning):
- Remove whitespace pre-tokenization constraint
- Continue BPE from V_1 to final size V_final
- Result: Vocabulary includes cross-word tokens
```

Benefit: Eliminates systematic bias from varying whitespace usage across languages.

Performance: +4.0% average accuracy, +8.2% on MMLU, 27% inference compute reduction.

#### OBPE (Overlap-Based BPE)
```
Modify BPE merge selection:
1. For each candidate merge, compute:
   - Frequency in current language
   - Overlap potential with related languages
2. Score = α * frequency + β * overlap_score
3. Select merge with highest score
4. Repeat until target vocabulary size
```

Application: Particularly effective for language families with lexical overlap.

Benefit: Increases low-resource language representation via shared tokens with high-resource relatives.

#### Parallel Tokenizers with Alignment
```
Training Phase:
1. Train separate monolingual tokenizers for each language
2. Create bilingual dictionaries or translation mappings
3. Align vocabularies by mapping semantically equivalent tokens
4. Assign consistent indices to aligned token pairs

Inference Phase:
- Each language uses its specialized tokenizer
- Embeddings for aligned tokens are shared or initialized similarly
- Enables semantic consistency across languages
```

Advantage: Combines monolingual quality with cross-lingual transfer capability.

### Identified Causes of Token Inequity

1. **Data Similarity** (R² = 0.239)
   - Train-evaluation dataset overlap is strongest predictor
   - Languages well-represented in training get better compression
   - Parallel data training provides minimal improvement (~1% CTC reduction)

2. **Mean Token Length in Evaluation** (R² = 0.168)
   - Token lengths in evaluation corpora matter more than vocabulary-wide averages
   - Languages with longer words need more vocabulary capacity
   - Morphologically complex languages disadvantaged

3. **Whitespace Pre-tokenization** (R² = 0.157)
   - Languages vary substantially in whitespace density
   - Pre-tokenization creates hard boundaries at different rates
   - Systematically constrains some languages more than others

4. **Script and Orthography**
   - Latin-script languages show consistently higher efficiency
   - Non-Latin scripts experience 3-5x token inflation
   - Script differences limit token overlap benefits

5. **Morphological Complexity**
   - Agglutinative and polysynthetic languages fragment more
   - Need larger vocabulary for comparable compression
   - Higher fertility and lower STRR

## Implications for Ensemble BPE Tokenization

### Direct Applications

1. **Language-Specific Component Sizing**
   - Research clearly shows uniform vocabulary allocation is suboptimal
   - Ensemble components should have language-specific vocabulary sizes based on VOCAP principles
   - Could allocate different ensemble "capacity" (weight, vocab size, or priority) to different languages

2. **Parity-Aware Ensemble Training**
   - Incorporate fairness objectives during ensemble component training
   - Use Gini coefficient or similar metrics to ensure balanced representation
   - Consider parity-aware merge selection for shared vocabulary components

3. **Controlled Vocabulary Overlap**
   - Research shows overlap benefits transfer but can hurt word-level tasks
   - Ensemble could maintain both shared and language-specific vocabularies
   - Route different tasks to appropriate ensemble components

4. **Hierarchical Tokenization Strategy**
   - SuperBPE's two-phase approach suggests ensemble could combine:
     - Subword-focused tokenizers (respect whitespace)
     - Superword-focused tokenizers (cross-word merges)
     - Different levels for different contexts

5. **Semantic Alignment Mechanisms**
   - Parallel tokenizer research suggests training components monolingually then aligning
   - Could train language-specific ensemble components with explicit semantic alignment
   - Use bilingual dictionaries or translation models to enforce consistent representations

### Novel Ensemble Architectures

1. **Task-Aware Ensemble Routing**
   - Based on Limisiewicz et al.: overlap helps sentence-level, hurts word-level tasks
   - Route word-level tasks to language-specific components
   - Route sentence-level tasks to overlap-heavy components
   - Dynamic weighting based on task type

2. **Fairness-Optimized Ensemble**
   - Multiple components optimized for different fairness objectives
   - Parity component (minimizes Gini)
   - Efficiency component (maximizes compression)
   - Balance via learned weights or selection policy

3. **Language-Family Clustering**
   - OBPE shows related languages benefit from shared tokens
   - Create ensemble components for language families
   - Within-family overlap, between-family specialization
   - Hierarchical structure: language-specific → family → universal

4. **Multi-Metric Optimization**
   - Different components optimize different metrics (STRR, fertility, compression)
   - Ensemble combines to achieve multi-objective balance
   - Could use Pareto-optimal selection across components

### Evaluation Framework

Based on reviewed research, ensemble BPE should be evaluated on:

1. **Fairness Metrics**:
   - Gini coefficient across languages
   - Per-language STRR comparison
   - Token premium analysis (CTC disparities)
   - Vocabulary utilization per language

2. **Performance Metrics**:
   - Task-specific accuracy (word-level vs sentence-level)
   - Cross-lingual transfer on XNLI, XQuAD
   - Compression rate per language
   - Inference compute requirements

3. **Coverage Metrics**:
   - Vocabulary allocation distribution
   - Token overlap patterns (total and semantic)
   - Script coverage balance
   - Morphological complexity handling

### Open Research Questions for Ensemble BPE

1. **Optimal Component Architecture**:
   - How many ensemble components? (2-3 specialized vs 10+ language-specific)
   - Should components be trained jointly or independently then aligned?
   - How to balance monolingual quality vs cross-lingual transfer?

2. **Combination Strategies**:
   - Weighted averaging vs routing vs voting?
   - Fixed weights vs context-dependent selection?
   - How to combine when components produce different segmentations?

3. **Fairness-Efficiency Trade-offs**:
   - Can ensemble simultaneously optimize fairness AND global efficiency?
   - Is there a Pareto frontier, or can both improve together?
   - How much performance loss is acceptable for fairness gains?

4. **Semantic Overlap Management**:
   - How to enforce semantic consistency across ensemble components?
   - Should semantically similar tokens across components share embeddings?
   - Can ensemble learn optimal overlap automatically?

## Open Questions & Future Directions

### Remaining Research Gaps

1. **Downstream Performance Understanding**
   - Fertility and STRR correlations with task performance need more investigation
   - How tokenization metrics translate to generation quality vs comprehension
   - Long-context scenarios with varying token premiums

2. **Dynamic Vocabulary Allocation**
   - Most research assumes static vocabulary; could it adapt during training?
   - How to handle language shift in deployment (new domains, code-switching)
   - Online learning approaches for vocabulary rebalancing

3. **Interaction with Model Architecture**
   - How do positional encodings interact with variable fertility?
   - Do attention mechanisms compensate for tokenization inequities?
   - Optimal model size adjustments for fair tokenization

4. **Beyond Indo-European Languages**
   - Most research focuses on well-studied language families
   - Need more work on truly low-resource languages
   - Indigenous and endangered languages with unique characteristics

5. **Script and Orthography**
   - How to handle languages with multiple scripts?
   - Optimal treatment of logographic systems (Chinese, Japanese)
   - Romanization vs native script trade-offs

### Promising Future Directions

1. **Learned Tokenization**
   - Neural tokenizers that learn optimal segmentation end-to-end
   - Differentiable tokenization for gradient-based optimization
   - Multi-task learning incorporating fairness objectives

2. **Hybrid Approaches**
   - Combining BPE, Unigram, WordPiece strengths
   - Character + subword + word ensemble
   - Context-dependent tokenization strategies

3. **Fairness-First Design**
   - Starting from fairness constraints rather than retrofitting
   - Multi-objective optimization frameworks
   - Explicit equity metrics in tokenizer training

4. **Cross-Lingual Semantic Grounding**
   - Tighter integration with semantic representations
   - Multimodal grounding for vocabulary alignment
   - Conceptual rather than lexical tokenization

5. **Adaptive and Personalized Tokenization**
   - User-specific or domain-specific vocabulary adaptation
   - Streaming vocabulary updates
   - Continual learning frameworks

6. **Evaluation Standardization**
   - Comprehensive benchmark suites across diverse languages
   - Standardized fairness metrics and reporting
   - Open datasets for tokenization research

### Immediate Next Steps for Research Community

1. Develop comprehensive multilingual tokenization benchmark covering 100+ languages with diverse scripts
2. Create standardized fairness evaluation toolkit incorporating Gini, STRR, and semantic overlap metrics
3. Conduct systematic ablation studies on component interactions in ensemble tokenizers
4. Build public repository of language-specific vocabulary size recommendations (VOCAP-style)
5. Investigate tokenization impact on specific downstream tasks beyond current benchmarks
6. Explore integration of semantic similarity constraints during vocabulary construction
7. Develop efficient inference algorithms for ensemble tokenization systems

## References

### Primary Research Papers

1. Limisiewicz, T., Balhar, J., & Mareček, D. (2023). Tokenization Impacts Multilingual Language Modeling: Assessing Vocabulary Allocation and Overlap Across Languages. *Findings of ACL 2023*. arXiv:2305.17179. https://arxiv.org/abs/2305.17179

2. Foroutan, N., Meister, C., Paul, D., Niklaus, J., Ahmadi, S., Bosselut, A., & Sennrich, R. (2024). Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization. arXiv:2508.04796. https://arxiv.org/html/2508.04796

3. Arnett, C., Chang, T. A., Biderman, S., & Bergen, B. K. (2025). Explaining and Mitigating Crosslingual Tokenizer Inequities. arXiv:2510.21909. https://arxiv.org/html/2510.21909

4. Kallini, J., Jurafsky, D., Potts, C., & Bartelds, M. (2025). False Friends Are Not Foes: Investigating Vocabulary Overlap in Multilingual Language Models. arXiv:2509.18750. https://arxiv.org/abs/2509.18750

5. Nayeem, M. T., Alqahtani, S., Laskar, M. T. R., Mohiuddin, T., & Bari, M. S. (2025). Beyond Fertility: Analyzing STRR as a Metric for Multilingual Tokenization Evaluation. arXiv:2510.09947. https://arxiv.org/abs/2510.09947

6. Zheng, B., Dong, L., Huang, S., Singhal, S., Che, W., Liu, T., Song, X., & Wei, F. (2021). Allocating Large Vocabulary Capacity for Cross-Lingual Language Model Pre-Training. *EMNLP 2021*. https://aclanthology.org/2021.emnlp-main.257.pdf

7. Patil, V., Talukdar, P., & Sarawagi, S. (2022). Overlap-based Vocabulary Generation Improves Cross-lingual Transfer Among Related Languages. *ACL 2022*. arXiv:2203.01976. https://arxiv.org/abs/2203.01976 | https://aclanthology.org/2022.acl-long.18/

8. SuperBPE Team (2025). SuperBPE: Space Travel for Language Models. arXiv:2503.13423. https://arxiv.org/abs/2503.13423 | https://superbpe.github.io/

9. Multiple Authors (2024). Parallel Tokenizers: Rethinking Vocabulary Design for Cross-Lingual Transfer. arXiv:2510.06128. https://arxiv.org/abs/2510.06128

10. Multiple Authors (2024). Trans-Tokenization and Cross-lingual Vocabulary Transfers: Language Adaptation of LLMs for Low-Resource NLP. arXiv:2408.04303. https://arxiv.org/html/2408.04303v1

### Additional Resources

11. ACL Anthology (2023). Tokenization Impacts Multilingual Language Modeling. https://aclanthology.org/2023.findings-acl.350/

12. ResearchGate. Tokenization Impacts Multilingual Language Modeling: Assessing Vocabulary Allocation and Overlap Across Languages. https://www.researchgate.net/publication/371136633_Tokenization_Impacts_Multilingual_Language_Modeling_Assessing_Vocabulary_Allocation_and_Overlap_Across_Languages

13. Papers with Code. Overlap-based Vocabulary Generation Improves Cross-lingual Transfer Among Related Languages. https://paperswithcode.com/paper/overlap-based-vocabulary-generation-improves

14. Vaidehi99. OBPE GitHub Repository. https://github.com/vaidehi99/obpe

15. MarkTechPost (2025). SuperBPE: Advancing Language Models with Cross-Word Tokenization. https://www.marktechpost.com/2025/03/23/superbpe-advancing-language-models-with-cross-word-tokenization/

16. Grigorev, G. Tokenization from First Principles. https://ggrigorev.me/posts/tokenizer-superbpe/

17. GetMaxim.ai. SuperBPE: Rethinking Tokenization for Language Models. https://www.getmaxim.ai/blog/superbpe-rethinking-tokenization-for-language-models/

18. Rohan Paul. Tutorial: Balancing Vocabulary Size in Modern LLMs (GPT-4, LLaMA, Mistral). https://www.rohan-paul.com/p/tutorial-balancing-vocabulary-size

19. Rohan Paul. Transfer Learning Across Languages: Building Truly Multilingual LLMs. https://www.rohan-paul.com/p/transfer-learning-across-languages

20. HuggingFace. Summary of the Tokenizers. https://huggingface.co/docs/transformers/tokenizer_summary

### Conference Proceedings & Technical Reports

21. RWS Language Weaver. Issue #121 - Finding the Optimal Vocabulary Size for Neural Machine Translation. https://www.rws.com/language-weaver/blog/issue-121-finding-the-optimal-vocabulary-size-for-neural-machine-translation/

22. IndiaAI. Cross-lingual Transfer, Vocabulary Generation, and Dialogue State Tracking. https://indiaai.gov.in/article/cross-lingual-transfer-vocabulary-generation-and-dialogue-state-tracking

---

*Report compiled: 2025-11-01*
*Total sources reviewed: 22+ papers and articles*
*Focus areas: Multilingual vocabulary allocation, cross-lingual token sharing, fairness metrics, tokenization algorithms*
