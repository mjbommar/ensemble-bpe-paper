# Code Tokenization: BPE and Subword Methods for Programming Languages

## Summary

Code tokenization represents a critical challenge in applying natural language processing techniques to programming languages. While Byte Pair Encoding (BPE) has become the dominant subword tokenization algorithm for large language models (used in GPT, Llama, Mistral, and most modern LLMs), its application to source code reveals fundamental mismatches between frequency-based statistical tokenization and grammar-based programming language structure.

Research demonstrates that BPE tokenizers fragment approximately 70% of source code (which consists primarily of identifiers) into non-meaningful subword tokens, leading to sequence lengths roughly 2x longer than grammar-based alternatives. This "TokDrift" phenomenon causes models to exhibit sensitivity to semantically-neutral formatting changes: minor spacing or naming convention variations alter token sequences, causing prediction changes in 6-60% of cases depending on the model. Despite these challenges, domain-specific tokenizers trained on code data can compress sequences by over 40% compared to general-purpose tokenizers, and recent work shows that models can successfully adapt to new tokenizers when fine-tuned on 50+ billion tokens.

The field is evolving toward identifier-aware tokenization strategies (CodeT5), structure-aware representations (GraphCodeBERT), and grammar-aligned tokenization approaches that better respect programming language syntax boundaries. These innovations address the fundamental tension between BPE's frequency-driven merging and code's structured, compositional nature, where identifiers like "getUserDetails" carry semantic meaning that arbitrary subword splitting destroys.

## Key Findings

- **BPE is ubiquitous but problematic for code**: All modern LLMs (GPT-2 through GPT-4, Llama 3, Mistral, etc.) use BPE for tokenization, but the algorithm's frequency-based approach creates fundamental misalignment with programming language grammar, splitting identifiers into non-meaningful fragments and causing 2x sequence inflation. [TokDrift paper, arXiv:2510.14972]

- **Domain-specific tokenizers provide massive compression gains**: Code-trained tokenizers (like InCoder) compress sequences by 26-40% compared to general-purpose tokenizers (like Llama), translating directly to faster inference and larger effective context windows. [arXiv:2402.01035v1]

- **Vocabulary size has negligible impact on code performance**: Counterintuitively, testing vocabulary sizes from 32k to 256k showed near-zero correlation (r = -0.13, p = 0.87) with downstream code generation metrics, challenging assumptions about optimal vocabulary sizing. [arXiv:2402.01035v1]

- **Training data composition matters moderately**: Tokenizers trained on 70% code and 30% English text achieve optimal compression for code tasks, though the effect is moderate (5-6 percentage point variation). [arXiv:2402.01035v1]

- **Tokenization drift causes significant behavioral instability**: Minor formatting changes (adding spaces, changing from camelCase to snake_case) alter token sequences sufficiently to change model predictions in 6-60% of cases, with identifiers showing double the sensitivity compared to keywords. [TokDrift paper, arXiv:2510.14972]

- **Frequency dominates over compositionality**: In natural language, 90-95% of BPE's effectiveness comes from frequency-based encoding rather than compositional benefits, though this finding hasn't been validated specifically for code. [arXiv:2306.01393v3]

- **Token merging offers post-hoc efficiency gains**: Merging fragmented identifier tokens reduces computational costs by 1-19% (FLOPs) with minimal performance degradation for classification tasks and sometimes improved performance for generation tasks. [arXiv:2507.14423]

- **Out-of-vocabulary remains a critical challenge**: Variable names and parameter names represent the most common OOV words in code, as developers freely create identifiers with diverse naming patterns that don't naturally segment into common subwords. [SpringerLink investigation]

- **Pre-tokenization strategy matters more than vocabulary**: GPT-4's pre-tokenization regular expression provides superior compression compared to GPT-2's approach, representing a more impactful optimization lever than vocabulary size adjustments. [arXiv:2402.01035v1]

- **Structure-aware approaches show promise**: GraphCodeBERT's data flow integration demonstrates that models "prefer structure-level attentions over token-level attentions" in code tasks, suggesting structural awareness can transcend pure token-based processing limitations. [arXiv:2009.08366]

## Relevant Research & Papers

### Getting the Most Out of Your Tokenizer for Pre-training and Domain Adaptation (2024)
- **Authors/Source**: arXiv:2402.01035v1
- **Year**: 2024
- **Key Contributions**:
  - Systematic analysis of three optimization levers: training data distribution, pre-tokenization regex, and vocabulary size
  - Demonstrated code-specific tokenizers compress 26-40% better than general tokenizers
  - Proved vocabulary size has negligible correlation with downstream performance (r = -0.13)
  - Showed models require 50+ billion tokens for successful tokenizer adaptation
  - Introduced Fast Vocabulary Transfer (FVT) technique for initializing embeddings
  - Provided formulas for balancing compression against inference costs
- **Relevance to Ensemble BPE**: This research directly informs optimal training strategies for ensemble tokenizers, particularly the finding that training data composition (70% code / 30% text) matters more than vocabulary size, and that pre-tokenization regular expressions represent a critical but often overlooked optimization dimension.

### TokDrift: When LLM Speaks in Subwords but Code Speaks in Grammar (2024)
- **Authors/Source**: arXiv:2510.14972
- **Year**: 2024
- **Key Contributions**:
  - Identified and quantified fundamental misalignment between BPE's statistical tokenization and grammar-based code structure
  - Demonstrated that minor formatting changes cause 6-60% prediction variation across models
  - Showed identifiers exhibit double the sensitivity (10.82% vs 6.61%) compared to keywords
  - Traced the problem to early embedding layers where subword segmentation fails to capture grammar boundaries
  - Evaluated 9 code models across 3 tasks using 24 semantic-preserving rewrite rules
  - Advocated for grammar-aware or domain-adaptive tokenizers as solution direction
- **Relevance to Ensemble BPE**: TokDrift highlights a fundamental problem that ensemble approaches might address: if different tokenizers in an ensemble use different segmentation strategies (some grammar-aware, some frequency-based), the ensemble could potentially reduce sensitivity to formatting variations while maintaining the compression benefits of BPE.

### On the Effect of Token Merging on Pre-trained Models for Code (2024)
- **Authors/Source**: arXiv:2507.14423
- **Year**: 2024
- **Key Contributions**:
  - Quantified that ~70% of source code consists of identifiers that BPE fragments
  - Demonstrated token merging reduces FLOPs by 1-19% with minimal performance loss
  - Showed generation tasks sometimes improve with merging (CodeBLEU +2.47 points)
  - Proved strategic layer placement matters: late merging (layers 10-12) for classification, flexible for generation
  - Compared simple averaging vs. learnable attention-based merging strategies
  - Offered compatibility path for existing pre-trained models without vocabulary changes
- **Relevance to Ensemble BPE**: Token merging represents a post-tokenization solution that could complement ensemble tokenization strategies, potentially allowing an ensemble to include both fine-grained and merged token representations, optimizing for different task requirements dynamically.

### GraphCodeBERT: Pre-training Code Representations with Data Flow (2020)
- **Authors/Source**: arXiv:2009.08366, Microsoft Research
- **Year**: 2020
- **Key Contributions**:
  - Introduced data flow as semantic-level structure capturing "where-the-value-comes-from" variable relationships
  - Implemented graph-guided masked attention function to incorporate structure into Transformers
  - Demonstrated models prefer structure-level over token-level attentions in code search
  - Pre-training tasks include edge prediction and representation alignment
  - Showed advantages over syntactic structures (ASTs) by avoiding unnecessarily deep hierarchies
- **Relevance to Ensemble BPE**: GraphCodeBERT demonstrates that pure token-based approaches have inherent limitations for code understanding. An ensemble tokenizer system could potentially incorporate both subword tokens and structural tokens (representing data flow edges or AST nodes) to capture multiple granularities of code representation simultaneously.

### CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models (2021)
- **Authors/Source**: ACL Anthology, EMNLP 2021
- **Year**: 2021
- **Key Contributions**:
  - Introduced identifier-aware pre-training that recognizes token type information
  - Treats identifiers as meaningful semantic units rather than arbitrary character sequences
  - Unified encoder-decoder architecture for bidirectional understanding and sequential generation
  - Maintains identifier cohesion during tokenization to preserve naming convention semantics
  - Demonstrated improvements on code summarization, variable renaming, and defect detection
- **Relevance to Ensemble BPE**: CodeT5's identifier-aware approach suggests a critical feature for ensemble tokenizers: the ability to recognize and preserve identifier boundaries. An ensemble could include both identifier-preserving tokenizers and standard BPE tokenizers, learning when to prioritize semantic coherence versus compression efficiency.

### SentencePiece: A Simple and Language Independent Subword Tokenizer (2018)
- **Authors/Source**: Kudo & Richardson, ACL Anthology D18-2012
- **Year**: 2018
- **Key Contributions**:
  - Trains subword models directly from raw sentences without pre-tokenization
  - Enables end-to-end language-independent tokenization
  - Treats whitespace as native character for flexible cross-language handling
  - Provides C++ and Python implementations under Apache 2 license
  - Achieves comparable accuracy to pre-tokenized approaches on English-Japanese translation
- **Relevance to Ensemble BPE**: SentencePiece's language-agnostic design and raw-text training make it particularly suitable for ensemble approaches handling multiple programming languages simultaneously. Its whitespace handling could complement BPE in an ensemble for better managing code indentation and formatting.

### Assessing Frequency versus Compositionality for Subword Tokenization in NMT (2023)
- **Authors/Source**: arXiv:2306.01393v3
- **Year**: 2023
- **Key Contributions**:
  - Demonstrated frequency accounts for 90-95% of BPE's effectiveness in natural language
  - Used Huffman coding to isolate frequency effects from compositional benefits
  - Showed Huffman reaches within 4-8% of BPE performance (COMET metric)
  - Quantified that compositionality contributes significantly less than assumed
  - Focused on translation tasks (Czech, German, English, French)
- **Relevance to Ensemble BPE**: While this research doesn't address code specifically, it raises important questions about whether compositionality matters more for programming languages (where identifiers like "getUserData" have compositional semantics) compared to natural language. An ensemble approach could test whether different tokenization strategies optimized for frequency versus compositionality show complementary strengths on code tasks.

### CodeBERT: A Pre-Trained Model for Programming and Natural Languages (2020)
- **Authors/Source**: Microsoft, GitHub repository
- **Year**: 2020
- **Key Contributions**:
  - Multi-programming-lingual model covering 6 languages (Python, Java, JavaScript, PHP, Ruby, Go)
  - Uses RoBERTa tokenizer with byte-level BPE
  - Pre-trained on natural language-programming language pairs
  - Handles diverse programming syntaxes with unified tokenization approach
  - Demonstrated transfer learning effectiveness across multiple code tasks
- **Relevance to Ensemble BPE**: CodeBERT's multi-lingual approach shows that single tokenizers can handle multiple programming languages, but the reliance on RoBERTa's NL-focused tokenizer highlights the compromise inherent in unified approaches. An ensemble could specialize different tokenizers for different languages or maintain both NL-optimized and code-optimized tokenizers simultaneously.

### Byte-Pair Encoding tokenization - Hugging Face
- **Authors/Source**: Hugging Face LLM Course
- **Year**: 2024-2025
- **Key Contributions**:
  - Comprehensive educational resource on BPE implementation
  - Explains byte-level BPE reducing base vocabulary to 256 Unicode bytes
  - Documents usage in GPT-2, RoBERTa, and modern transformers
  - Provides practical code examples and training procedures
  - Discusses handling of spaces, special characters, and programming operators
- **Relevance to Ensemble BPE**: This resource documents the standard BPE implementation that most code models inherit. Understanding these baseline approaches is essential for designing ensemble systems that can incorporate traditional BPE alongside novel tokenization strategies while maintaining compatibility with existing model architectures and training pipelines.

## Technical Details

### BPE Algorithm Core Steps

1. **Initialize vocabulary**: Start with 256 single-byte tokens (0-255) for byte-level BPE
2. **Identify frequent pairs**: Scan text to find most commonly occurring adjacent token pairs
3. **Merge and record**: Replace the most frequent pair with a new token ID, record in merge table
4. **Iterate**: Repeat merging until desired vocabulary size reached or no compression gain

The algorithm is greedy and deterministic, typically reducing sequence length by approximately 50% for natural language. For code, compression varies by programming language and tokenizer training data.

### Vocabulary Size Considerations

Modern LLMs use varying vocabulary sizes:
- GPT-2: 50,257 tokens
- GPT-4: ~100,256 tokens
- GPT-4o: 199,997 tokens
- LLaMA: ~128,000 tokens
- Mistral: ~32,000 tokens

Research shows that once vocabulary reaches ~24k, every common word becomes a single token. Beyond this point, additional vocabulary captures increasingly rare words and code-specific identifiers. The optimal size depends on model scale, domain (code vs. text), and computational constraints rather than downstream task performance.

### Pre-tokenization Regular Expressions

Pre-tokenization defines how raw text splits before BPE merging begins. GPT-4's regex offers superior compression compared to GPT-2's approach for code:

**GPT-2 approach**: Treats each space as separate token, problematic for Python's indentation-based syntax

**GPT-4 approach**: More sophisticated pattern matching that better preserves code structure while enabling compression

The "Punct" tokenizer (research variant) prevents sequences like ".append" from becoming single tokens, favoring interpretability over compression.

### Code-Specific Tokenization Challenges

1. **Identifier Fragmentation**: Variable names like "calculateUserScore" split into arbitrary fragments ["calculate", "User", "Score"] or ["calc", "ulate", "User", "Score"] depending on training data frequency statistics

2. **Operator Handling**: Punctuation-heavy code (operators, brackets, semicolons) can inflate token counts if not handled by pre-tokenization rules

3. **Indentation Sensitivity**: Whitespace carries syntactic meaning in Python but gets variably tokenized depending on frequency

4. **Multi-language Support**: Different programming languages have different keyword densities, comment styles, and naming conventions

5. **Case Sensitivity**: camelCase vs. snake_case vs. PascalCase produce different tokenizations, causing model sensitivity to semantically-neutral formatting choices

### Token Merging Strategies

Post-tokenization merging addresses BPE fragmentation:

**Simple Averaging**: Aggregate fragmented token embeddings by averaging, minimal computation overhead

**Learnable Attention**: Use attention mechanism to weight fragment contributions, achieves better performance but adds parameters

**Strategic Placement**:
- Classification tasks: Merge late (layers 10-12) to preserve fine-grained distinctions
- Generation tasks: Merge early with minimal performance impact

### Grammar-Aware Tokenization Approaches

Emerging alternatives to pure BPE:

1. **Identifier-aware tokenization** (CodeT5): Recognize and preserve identifier boundaries as semantic units
2. **Structure-aware tokenization** (GraphCodeBERT): Incorporate data flow or AST structure alongside token sequences
3. **Grammar-based splitting**: Use language parsers to determine token boundaries before applying compression
4. **Hybrid approaches**: Combine grammar rules with frequency statistics

### Training Data Composition

For code-specific tokenizers, optimal composition appears to be:
- 70% source code from target programming languages
- 30% natural language (documentation, comments, text)

Training on pure code risks losing natural language understanding needed for comments and documentation, while pure text training sacrifices code compression efficiency.

### Fast Vocabulary Transfer (FVT)

When adapting models to new tokenizers:

1. Initialize new embeddings from old vocabulary where tokens overlap
2. For novel tokens, use average or weighted combination of similar tokens
3. Fine-tune on 50+ billion tokens for full adaptation
4. Below 50B tokens, performance degradation remains substantial

This technique enables retrofitting pre-trained models with better tokenizers without training from scratch.

## Implications for Ensemble BPE Tokenization

### Addressing TokDrift Through Diversity

The TokDrift phenomenon—where semantically neutral formatting changes alter tokenization and model predictions—presents a compelling use case for ensemble tokenizers. By training multiple tokenizers with different:
- Pre-tokenization regular expressions (GPT-2 vs. GPT-4 style)
- Training data compositions (code-heavy vs. balanced)
- Merge strategies (frequency-only vs. grammar-informed)

An ensemble system could achieve more stable predictions by aggregating across tokenization variations that individually show high sensitivity.

### Multi-Granularity Representation

Evidence from GraphCodeBERT and CodeT5 suggests code benefits from multiple representation granularities simultaneously:
- Subword tokens for compression and unknown word handling
- Identifier-level tokens for semantic coherence
- Structural tokens representing data flow or syntactic relationships

An ensemble approach naturally accommodates this multi-granularity requirement by incorporating specialized tokenizers for each level.

### Language-Specific Specialization

CodeBERT's multi-lingual approach demonstrates unified tokenizers can handle diverse programming languages, but at efficiency costs. An ensemble system could:
- Train language-specific tokenizers optimized for Python, Java, JavaScript, etc.
- Maintain a general-purpose tokenizer for cross-language patterns
- Dynamically weight tokenizer contributions based on detected language

This specialization-with-fallback strategy could achieve better compression per language while maintaining broad coverage.

### Optimal Vocabulary Allocation

The finding that vocabulary size shows negligible correlation with downstream performance (r = -0.13) suggests a different optimization strategy for ensembles:

Rather than one large vocabulary, maintain multiple moderate vocabularies (32-64k each) with different optimization objectives:
- Frequency-optimized for compression
- Identifier-optimized for semantic preservation
- Grammar-aligned for structural consistency

Total parameter count remains similar, but representational diversity increases.

### Compositionality vs. Frequency Trade-offs

The frequency-dominates-compositionality finding from NMT research raises questions about code:

Programming identifiers often have compositional semantics ("getUserData", "maxRetryAttempts") where parts carry meaning. An ensemble could test whether:
- One tokenizer optimizes for frequency-based compression
- Another prioritizes compositional segmentation
- The combination captures both efficiency and semantic coherence

### Post-Training Adaptation Path

The requirement of 50+ billion tokens for successful tokenizer adaptation presents challenges for domain-specific applications. An ensemble approach offers an alternative:

Rather than fully adapting a model to a new tokenizer, add the new tokenizer to an ensemble with partially-trained integration. This could enable:
- Incremental vocabulary expansion without full retraining
- Domain adaptation with less data (by keeping general tokenizers active)
- Graceful degradation when encountering domain shifts

### Efficiency Optimization

Token merging research shows 1-19% computational savings through post-tokenization aggregation. An ensemble system could:
- Use fine-grained tokenization during training for maximum information
- Apply task-specific merging during inference for efficiency
- Maintain different granularities for different model layers

This dynamic granularity adjustment matches evidence that classification needs fine tokens while generation tolerates merging.

### Handling OOV and Rare Identifiers

Out-of-vocabulary identifiers remain problematic despite subword tokenization. An ensemble approach could:
- Maintain a character-level tokenizer as guaranteed fallback
- Use domain-specific tokenizers for common identifier patterns
- Employ learnable routing to select appropriate tokenizer per identifier

This multi-level coverage ensures no identifier becomes completely opaque to the model.

### Structural Integration

GraphCodeBERT's structure-aware attention suggests tokenization alone is insufficient. An ensemble framework could naturally integrate:
- Token-level representations from multiple BPE tokenizers
- Structure-level representations from AST or data flow graphs
- Hybrid representations combining both granularities

This aligns with evidence that models prefer structural attention for code tasks while maintaining token-level flexibility for generation.

### Evaluation and Routing Strategy

Key research questions for ensemble BPE systems:

1. **Static vs. Dynamic Weighting**: Should tokenizer weights be learned during pre-training or adjusted per input?
2. **Task-Specific Ensembles**: Should different tasks (code completion vs. bug detection) use different tokenizer combinations?
3. **Layer-Specific Selection**: Should different model layers receive different tokenization granularities?
4. **Compression-Quality Trade-offs**: How to balance ensemble complexity against inference efficiency?

The research reviewed suggests dynamic, task-aware ensembles would likely outperform static combinations, but at computational cost.

## Open Questions & Future Directions

### Theoretical Foundations

1. **Grammar-Statistics Unification**: Can we develop theoretical frameworks that unify frequency-based BPE with grammar-based tokenization, formalizing when each approach is optimal?

2. **Compositionality in Code**: Does compositionality matter more for programming languages than natural language? Existing research on frequency vs. compositionality examined only NMT tasks—replication for code is needed.

3. **Optimal Tokenizer Diversity**: What is the minimum number and maximum diversity of tokenizers needed in an ensemble before hitting diminishing returns?

4. **Tokenization Stability Metrics**: How do we quantify and optimize for tokenization stability beyond TokDrift's format variation tests?

### Practical Implementation

5. **Efficient Ensemble Architectures**: How can multiple tokenizers be integrated without proportionally increasing model size and inference cost?

6. **Training Dynamics**: How do gradient flows change when models receive multiple tokenization views simultaneously? Are there optimization challenges?

7. **Vocabulary Management**: With multiple tokenizers, how do we handle vocabulary explosion and ensure efficient embedding storage?

8. **Streaming and Incremental Processing**: Can ensemble tokenizers work effectively for streaming code completion where input arrives incrementally?

### Domain-Specific Applications

9. **Language-Specific Optimization**: Should ensemble systems include per-language tokenizers for major languages (Python, Java, C++) or rely on universal approaches?

10. **Domain Adaptation**: How much code data is needed to train effective domain-specific tokenizers for specialized areas (embedded systems, web development, data science)?

11. **Multi-Modal Code**: How should tokenizers handle code mixed with natural language (documentation, comments), LaTeX (in scientific code), or SQL (in application code)?

12. **Legacy Code**: Do older codebases with different naming conventions benefit from different tokenization strategies than modern code?

### Evaluation Methodologies

13. **Comprehensive Benchmarks**: Existing evaluations focus on HumanEval and MBPP—are these sufficient for assessing tokenization quality, or do we need compression-specific, robustness-specific, and compositionality-specific benchmarks?

14. **Ablation Studies**: Which ensemble components matter most? Is diversity in pre-tokenization regex more important than vocabulary size variation?

15. **Cross-Model Transfer**: Do optimal tokenization strategies transfer across model architectures (encoder-only, decoder-only, encoder-decoder)?

16. **Long-Context Effects**: How does tokenization choice interact with context length? Do different tokenizers show different degradation patterns at 32k+ token contexts?

### Integration with Broader Systems

17. **IDE Integration**: How can ensemble tokenizers improve real-time code completion in IDEs where latency constraints are strict?

18. **Code Search and Similarity**: Do ensemble approaches that prioritize different aspects (compression vs. semantics) improve code search and clone detection?

19. **Vulnerability Detection**: Does tokenization stability correlate with security—are models with unstable tokenization more susceptible to adversarial code formatting?

20. **Automated Repair**: When models generate code fixes, does tokenization strategy affect the syntactic correctness and semantic preservation of repairs?

### Connections to Broader ML

21. **Ensemble Learning Theory**: How do traditional ensemble learning principles (bagging, boosting, stacking) apply to tokenization ensembles?

22. **Meta-Learning**: Can models learn to learn tokenization strategies, adapting tokenization online based on input characteristics?

23. **Uncertainty Quantification**: Can disagreement among ensemble tokenizers provide useful uncertainty estimates about model predictions?

24. **Interpretability**: Do ensemble tokenizers improve model interpretability by providing multiple views of the same code?

## References

### Academic Papers

1. Getting the Most Out of Your Tokenizer for Pre-training and Domain Adaptation (2024)
   https://arxiv.org/html/2402.01035v1

2. TokDrift: When LLM Speaks in Subwords but Code Speaks in Grammar (2024)
   https://arxiv.org/html/2510.14972

3. On the Effect of Token Merging on Pre-trained Models for Code (2024)
   https://arxiv.org/html/2507.14423

4. GraphCodeBERT: Pre-training Code Representations with Data Flow (2020)
   https://arxiv.org/abs/2009.08366
   https://openreview.net/forum?id=jLoC4ez43PZ

5. CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models (2021)
   https://aclanthology.org/2021.emnlp-main.685.pdf

6. SentencePiece: A Simple and Language Independent Subword Tokenizer (2018)
   https://aclanthology.org/D18-2012/

7. Assessing Frequency versus Compositionality for Subword Tokenization in NMT (2023)
   https://arxiv.org/html/2306.01393v3

8. Problematic Tokens: Tokenizer Bias in Large Language Models (2024)
   https://arxiv.org/html/2406.11214v3

9. Investigating the OOV Problem and Its Impacts on Neural Program Repair
   https://link.springer.com/chapter/10.1007/978-3-032-00828-2_11

10. Augmenting the Interpretability of GraphCodeBERT for Code Similarity Tasks (2024)
    https://arxiv.org/html/2410.05275v1

### Technical Resources and Implementations

11. Byte-Pair Encoding tokenization - Hugging Face LLM Course
    https://huggingface.co/learn/llm-course/en/chapter6/5

12. Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch - Sebastian Raschka
    https://sebastianraschka.com/blog/2025/bpe-from-scratch.html

13. minbpe: Minimal, clean code for BPE algorithm - Andrej Karpathy
    https://github.com/karpathy/minbpe

14. CodeBERT: A Pre-Trained Model for Programming and Natural Languages - Microsoft
    https://github.com/microsoft/CodeBERT

15. Hugging Face Transformers - Tokenizer Summary
    https://huggingface.co/docs/transformers/tokenizer_summary

16. SentencePiece - Google Research
    https://github.com/google/sentencepiece

17. Papers with Code - CodeT5 Explained
    https://paperswithcode.com/method/codet5

18. CodeT5+ - Salesforce AI Research
    https://github.com/salesforce/CodeT5

### Educational Articles and Blog Posts

19. Let's Build the GPT Tokenizer: A Complete Guide - fast.ai
    https://www.fast.ai/posts/2025-10-16-karpathy-tokenizers

20. How BPE works - the tokenization algorithm used by LLMs
    https://sidsite.com/posts/bpe/

21. Byte-Pair Encoding: Subword-based tokenization algorithm - Towards Data Science
    https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0/

22. The BPE Tokenizer - MartinLwx's Blog
    https://martinlwx.github.io/en/the-bpe-tokenizer/

23. Complete Guide to Subword Tokenization Methods
    https://blog.octanove.org/guide-to-subword-tokenization/

24. Tokenization Optimization: Best Practices for LLMs
    https://www.prompts.ai/en/blog/tokenization-optimization-best-practices-for-llms

25. Byte-pair encoding - Wikipedia
    https://en.wikipedia.org/wiki/Byte-pair_encoding

### Related Tools and Libraries

26. TensorFlow Text - Subwords Tokenizer Guide
    https://www.tensorflow.org/text/guide/subwords_tokenizer

27. Papers with Code - Gradient-Based Subword Tokenization
    https://paperswithcode.com/method/gradient-based-subword-tokenization

28. Transformer-XL with BPE on large datasets
    https://github.com/chiayewken/transformer_xl

### Specialized Topics

29. Tokenization Impacts Multilingual Language Modeling
    https://www.researchgate.net/publication/372915763_Tokenization_Impacts_Multilingual_Language_Modeling_Assessing_Vocabulary_Allocation_and_Overlap_Across_Languages

30. Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training
    https://arxiv.org/html/2508.15390

31. How Much is Enough? The Diminishing Returns of Tokenization Training Data
    https://arxiv.org/html/2502.20273v1
