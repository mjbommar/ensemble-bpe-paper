# Byte-Level BPE and Character-Level vs Subword Tokenization: A Comprehensive Research Survey

## Summary

This research explores the evolution and technical foundations of tokenization strategies in natural language processing, with particular focus on byte-level Byte Pair Encoding (BPE), the comparison between character-level and subword approaches, and UTF-8 tokenization challenges in multilingual contexts. The findings reveal that tokenization remains a critical design decision with no universal solution—different applications benefit from different strategies based on factors including language diversity, computational constraints, and downstream task requirements.

Byte-level BPE, popularized by GPT-2 and subsequent large language models, represents a significant advancement over traditional character-level or word-level approaches. By operating on UTF-8 byte sequences rather than Unicode characters, byte-level BPE achieves a compact base vocabulary of 256 tokens while maintaining universal coverage across all possible text inputs. However, recent research has identified fundamental limitations in standard BPE implementations, including frequency imbalances, pre-tokenization constraints, and cross-lingual inequities, leading to innovative variants such as Boundless BPE, Scaffold-BPE, and Parity-aware BPE.

The landscape is further evolving toward fully byte-level architectures like the Byte Latent Transformer, which eliminates fixed tokenization entirely in favor of dynamic, entropy-based patch segmentation. These developments suggest the field is moving beyond asking "which tokenization method is best?" toward fundamentally rethinking how language models should segment and process text.

## Key Findings

- **No Universal Solution**: Research conclusively demonstrates "there is and likely will never be a silver bullet singular solution for all applications" (Mielke et al., 2021). Tokenization must be thoughtfully selected based on application-specific requirements rather than assumed as a solved problem.

- **Byte-Level BPE Achieves Universal Coverage**: GPT-2's byte-level approach uses 256 bytes as the base vocabulary, ensuring complete coverage of any UTF-8 encoded text without unknown tokens, while standard character-level approaches would require thousands of Unicode characters for comparable coverage (Radford et al., 2019).

- **Pre-tokenization Creates Distribution Skew**: Standard BPE implementations use pre-tokenization (splitting on whitespace and punctuation), which "causes the distribution of tokens in a corpus to heavily skew towards common, full-length words," limiting vocabulary scaling efficiency (Glocker et al., 2025).

- **Frequency Imbalance Degrades Training**: Original BPE inadvertently creates "Scaffold Tokens" that primarily serve as components of longer tokens and appear infrequently on their own, introducing training inefficiencies that can be mitigated through dynamic removal mechanisms (Wang et al., 2024).

- **Standard Tokenization Favors High-Resource Languages**: Frequency-based tokenization objectives systematically disadvantage low-resource languages, with costs to process text in languages like Dzongkha, Odia, Santali, or Shan being more than 12 times higher than English in GPT-4 (Petrov et al., 2023).

- **UTF-8 Encoding Length Creates "Byte Premium"**: The variable encoding length of UTF-8 implicitly penalizes non-Latin scripts—ASCII characters require one byte, Greek/Cyrillic require two bytes, and Chinese/Japanese/Korean require three bytes, creating inherent efficiency disparities (Petrov et al., 2023).

- **Boundless BPE Improves Compression by 15%**: Relaxing pre-tokenization constraints and merging complete pretokens into "superwords" achieves up to 15% improvement in bytes per token compression efficiency with substantially more uniform token distributions (Glocker et al., 2025).

- **WordPiece Differs from BPE in Selection Criteria**: While BPE selects the most frequent symbol pair for merging, WordPiece selects pairs that "maximize the likelihood of the training data once added to the vocabulary," providing a more statistically principled approach (Schuster & Nakajima, 2012).

- **Byte-Level Models Enable New Architectures**: The Byte Latent Transformer demonstrates that models can operate on raw bytes with dynamic, entropy-based patch segmentation, achieving better scaling efficiency than tokenization-based models at equivalent inference costs (Fickinger et al., 2024).

- **Character-Level Tokenization Trades Coverage for Sequence Length**: While character-level approaches eliminate vocabulary issues and handle out-of-vocabulary words seamlessly, they produce very long sequences that increase computational costs and may lose semantic understanding.

## Relevant Research & Papers

### Between words and characters: A Brief History of Open-Vocabulary Modeling and Tokenization in NLP
- **Authors/Source**: Mielke, Aicher, Salesky, et al.
- **Year**: 2021
- **arXiv ID**: 2112.10508
- **Key Contributions**:
  - Comprehensive survey connecting pre-neural and neural era tokenization research
  - Demonstrates the evolution from treating words as atomic units to subword-based methods
  - Explores the full spectrum from bytes to multi-word expressions
  - Establishes that no singular tokenization solution exists for all applications
- **Relevance to Ensemble BPE**: Provides historical context showing that ensemble approaches align with the field's recognition that different tokenization strategies have complementary strengths worth combining.

### Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier
- **Authors/Source**: Glocker, Lopez, and Leong
- **Year**: 2025
- **Conference**: COLM 2025
- **arXiv ID**: 2504.00178
- **Key Contributions**:
  - Identifies pre-tokenization as a fundamental limitation causing skewed token distributions
  - Introduces BoundlessBPE, which selectively merges complete pretokens into "superwords"
  - Achieves up to 15% improvement in compression efficiency (bytes per token)
  - Demonstrates substantially more uniform token distribution across corpora
- **Relevance to Ensemble BPE**: Suggests that combining tokenizers with different boundary constraints (standard BPE vs Boundless BPE) could provide complementary coverage of linguistic patterns.

### Scaffold-BPE: Enhancing Byte Pair Encoding with Simple and Effective Scaffold Token Removal
- **Authors/Source**: Wang et al.
- **Year**: 2024
- **arXiv ID**: 2404.17808
- **Key Contributions**:
  - Identifies inherent frequency imbalance flaw in original BPE algorithm
  - Introduces "Scaffold Tokens" concept—low-frequency tokens that primarily serve as components
  - Proposes dynamic scaffold token removal mechanism (parameter-free, computation-light)
  - Demonstrates consistent improvements over standard BPE in language modeling and machine translation
- **Relevance to Ensemble BPE**: Highlights that different BPE training strategies (with/without scaffold removal) produce qualitatively different tokenizations that might complement each other in ensemble settings.

### Parity-Aware Byte Pair Encoding: Improving Cross-lingual Fairness in Tokenization
- **Authors/Source**: Petrov et al.
- **Year**: 2025
- **arXiv ID**: 2508.04796
- **Key Contributions**:
  - Demonstrates systematic bias in frequency-based tokenization favoring dominant languages
  - Quantifies cost disparities (12x more expensive for some low-resource languages vs English)
  - Proposes parity-aware BPE that maximizes compression of worst-compressed language at each merge
  - Achieves cross-lingual equity with negligible impact on global compression or downstream performance
- **Relevance to Ensemble BPE**: Suggests that ensembles could include specialized tokenizers trained with parity-aware objectives to ensure fair representation across languages.

### Byte Latent Transformer: Patches Scale Better Than Tokens
- **Authors/Source**: Fickinger et al.
- **Year**: 2024
- **arXiv ID**: 2412.09871
- **Key Contributions**:
  - Introduces fully byte-level architecture without fixed tokenization
  - Encodes bytes into dynamically sized patches based on next-byte entropy
  - Demonstrates better scaling than tokenization-based models at equivalent inference costs
  - First comprehensive scaling study of byte-level models up to 8B parameters
  - Shows improvements in reasoning and long-tail generalization
- **Relevance to Ensemble BPE**: Represents potential future direction where ensemble methods might combine fixed tokenization with dynamic patching strategies.

### Language Model Tokenizers Introduce Unfairness Between Languages
- **Authors/Source**: Petrov et al.
- **Year**: 2023
- **arXiv ID**: 2305.15425
- **Key Contributions**:
  - Systematic analysis of UTF-8 encoding length disparities across scripts
  - Documents "byte premium" effects where non-Latin scripts are penalized
  - Quantifies computational cost and efficiency implications across languages
  - ASCII characters require 1 byte, Greek/Cyrillic 2 bytes, CJK 3 bytes
- **Relevance to Ensemble BPE**: Motivates the need for ensembles that include script-specific or language-specific tokenizers to balance UTF-8 encoding inequities.

### Summary of the tokenizers (Hugging Face Documentation)
- **Authors/Source**: Hugging Face Team
- **Year**: 2024
- **URL**: https://huggingface.co/docs/transformers/en/tokenizer_summary
- **Key Contributions**:
  - Comprehensive technical documentation of BPE, WordPiece, and byte-level BPE
  - Explains algorithmic differences: BPE uses frequency, WordPiece uses likelihood maximization
  - Documents which models use which approaches (GPT-2/RoBERTa use BPE, BERT uses WordPiece)
  - Details GPT-2's 50,257 vocabulary: 256 bytes + 1 end-of-text token + 50,000 merges
- **Relevance to Ensemble BPE**: Provides implementation details necessary for understanding how to combine different tokenizer types in ensemble frameworks.

### Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch
- **Authors/Source**: Sebastian Raschka
- **Year**: 2025
- **URL**: https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
- **Key Contributions**:
  - Educational implementation showing BPE algorithm steps clearly
  - Explains UTF-8 handling in byte-level BPE
  - Provides practical implementation details for GPT-2 style tokenization
- **Relevance to Ensemble BPE**: Offers implementation guidance for building custom tokenizers that could be combined in ensemble frameworks.

### MinBPE: Minimal, Clean Code for BPE Algorithm
- **Authors/Source**: Andrej Karpathy
- **Year**: 2024
- **URL**: https://github.com/karpathy/minbpe
- **Key Contributions**:
  - Pedagogical implementation with three variants: BasicTokenizer, RegexTokenizer, GPT4Tokenizer
  - Demonstrates essential merge mechanism and frequency analysis
  - Shows importance of regex pre-tokenization to prevent merges across semantic boundaries
  - Highlights security considerations for special token handling
  - Clean, commented code enabling direct comprehension
- **Relevance to Ensemble BPE**: Provides minimal reference implementation useful for experimenting with ensemble tokenizer architectures.

## Technical Details

### Byte-Level BPE Algorithm

The core byte-level BPE algorithm operates through the following steps:

1. **UTF-8 Encoding**: Input text is first encoded to UTF-8 byte sequences
2. **Base Vocabulary Initialization**: Start with 256 tokens representing all possible byte values (0-255)
3. **Byte-to-Unicode Mapping**: Create a mapping that avoids control characters:
   - Include printable ASCII (33-126) and extended Latin (161-172, 174-255)
   - Map remaining bytes to higher Unicode code points starting at 256
   - This prevents BPE from encountering problematic whitespace/control characters
4. **Frequency Analysis**: Count all adjacent byte pair occurrences in the corpus
5. **Iterative Merging**: Repeatedly merge the most frequent pair into a new token
6. **Vocabulary Growth**: Continue until reaching target vocabulary size (e.g., GPT-2 uses 50,000 merges)

### GPT-2 Regex Pre-tokenization Pattern

GPT-2 uses the following regex pattern for pre-tokenization:

```regex
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

This pattern:
- Captures common English contractions ('s, 't, 're, 've, 'm, 'll, 'd)
- Matches optional space followed by Unicode letter sequences (`\p{L}`)
- Matches optional space followed by Unicode number sequences (`\p{N}`)
- Captures punctuation and special characters
- Handles whitespace appropriately to prevent unwanted merges

### WordPiece Selection Criterion

While BPE uses a simple frequency-based heuristic:
```
merge = argmax_{(a,b)} count(a, b)
```

WordPiece uses a likelihood-based criterion:
```
merge = argmax_{(a,b)} P(ab) / (P(a) × P(b))
```

This measures how much merging the pair increases the likelihood of the training data, providing a more statistically principled selection mechanism.

### Boundless BPE Superword Formation

Boundless BPE extends standard BPE by:
1. Performing normal BPE merges within pretokens
2. Additionally merging complete pretokens into "superwords"
3. Superwords need not be semantically meaningful (e.g., " of" + " the" → " of the")
4. This relaxes the strict pretoken boundary constraint
5. Results in more uniform token distribution and improved compression

### Scaffold Token Removal Mechanism

Scaffold-BPE identifies and removes tokens that:
1. Were created during BPE merging process
2. Primarily appear as components of longer tokens
3. Have low independent frequency in the corpus
4. Contribute to training inefficiency through frequency imbalance

The dynamic removal is parameter-free and computation-light, applied during tokenization rather than vocabulary construction.

### Parity-Aware Merge Selection

Parity-aware BPE modifies the merge selection criterion:

Standard BPE:
```
merge = argmax_{(a,b)} count_total(a, b)
```

Parity-aware BPE:
```
worst_lang = argmin_{lang} compression_ratio(lang)
merge = argmax_{(a,b)} count_in_lang(a, b, worst_lang)
```

At each step, this maximizes compression for the currently worst-compressed language, trading some global efficiency for cross-lingual equity.

### Byte Latent Transformer Patch Segmentation

BLT replaces fixed tokenization with dynamic patching:
1. Encode bytes into variable-length patches
2. Segment based on entropy of next byte prediction
3. Use longer patches where data is predictable (low entropy)
4. Use shorter patches where complexity is high (high entropy)
5. This adapts computational allocation to data complexity dynamically

## Implications for Ensemble BPE Tokenization

The research findings have several important implications for ensemble tokenizer training:

### 1. Complementary Strengths of Different BPE Variants

Different BPE training strategies produce qualitatively different tokenizations:
- **Standard BPE**: Frequency-based merging, respects pretoken boundaries
- **Boundless BPE**: Crosses pretoken boundaries, more uniform distribution
- **Scaffold-BPE**: Removes low-frequency intermediate tokens, reduces training imbalance
- **Parity-aware BPE**: Prioritizes cross-lingual fairness over global compression

An ensemble could leverage these different optimization objectives to achieve more robust tokenization across diverse inputs.

### 2. Language-Specific Tokenization Needs

The UTF-8 "byte premium" effect creates systematic disadvantages for non-Latin scripts. An ensemble approach could include:
- General-purpose byte-level BPE for broad coverage
- Script-specific or language-specific tokenizers trained with parity-aware objectives
- Specialized tokenizers for morphologically rich languages
- Combined outputs to ensure fair representation across languages

### 3. Pre-tokenization Strategy Variation

The Boundless BPE results suggest that pre-tokenization constraints significantly impact tokenization quality. An ensemble could combine:
- Tokenizers with different regex patterns (GPT-2, GPT-4, custom patterns)
- Tokenizers with varying boundary constraints (strict vs relaxed)
- Tokenizers with different granularities (word-level, subword-level, byte-level)

### 4. Vocabulary Size Optimization

Different vocabulary sizes may be optimal for different aspects:
- Smaller vocabularies (1K-5K): Better for rare patterns, morphological variations
- Medium vocabularies (10K-30K): Balance between coverage and efficiency
- Larger vocabularies (50K-100K): Better for common patterns, whole words

An ensemble could include tokenizers at different vocabulary scales to capture patterns at multiple granularities.

### 5. Frequency vs Likelihood Trade-offs

BPE and WordPiece represent different optimization objectives. An ensemble could combine:
- BPE tokenizers (frequency-based merging) for capturing common patterns
- WordPiece tokenizers (likelihood-based merging) for statistically optimal decompositions
- Custom objectives that optimize for specific downstream tasks

### 6. Dynamic vs Static Segmentation

The Byte Latent Transformer demonstrates advantages of dynamic segmentation. Future ensemble approaches might combine:
- Fixed tokenization for consistency and interpretability
- Dynamic patching for adaptive computation allocation
- Hybrid approaches that use both strategies contextually

### 7. Handling Out-of-Distribution Text

Different tokenization strategies handle novel inputs differently:
- Byte-level approaches guarantee coverage but may produce long sequences
- Character-level approaches handle typos/noise well
- Subword approaches balance efficiency and coverage

Ensembles could provide more robust handling of out-of-distribution text by combining these complementary capabilities.

### 8. Training Data Distribution Sensitivity

Standard BPE is highly sensitive to training corpus distribution, systematically favoring dominant languages and patterns. An ensemble trained on:
- Different data distributions (balanced vs natural distribution)
- Different language mixtures (multilingual vs monolingual)
- Different domains (general vs specialized)

Could provide more balanced tokenization across diverse downstream applications.

### 9. Computational Efficiency Considerations

Different tokenization strategies have different computational profiles:
- Byte-level BPE produces longer sequences but has small vocabulary
- Character-level produces very long sequences with minimal vocabulary
- Subword balances sequence length and vocabulary size

Ensembles could optimize for specific computational constraints by selecting appropriate component tokenizers.

## Open Questions & Future Directions

### 1. Optimal Ensemble Composition
- What is the optimal number of tokenizers in an ensemble?
- How should tokenizers be selected to maximize diversity while maintaining quality?
- What metrics best measure tokenizer complementarity?

### 2. Ensemble Aggregation Strategies
- How should multiple tokenizations be combined (voting, averaging, stacking)?
- Should aggregation be learned or rule-based?
- Can attention mechanisms effectively weight different tokenizations contextually?

### 3. Dynamic vs Static Ensembles
- Should the ensemble composition change based on input characteristics (language, domain, script)?
- Can models learn to route different inputs to different tokenizers?
- What computational overhead is acceptable for dynamic selection?

### 4. Byte-Level vs Character-Level for Ensembles
- Should ensemble components operate at byte-level, character-level, or mixed?
- How do UTF-8 encoding artifacts affect ensemble performance?
- Can hybrid approaches mitigate the "byte premium" effect?

### 5. Pre-tokenization Diversity
- How much diversity in pre-tokenization patterns benefits ensemble performance?
- Should ensembles include both bounded and boundless BPE variants?
- What role does regex pattern design play in ensemble quality?

### 6. Cross-lingual Fairness in Ensembles
- Can ensembles naturally achieve better cross-lingual fairness than single tokenizers?
- Should some ensemble components be trained with parity-aware objectives?
- How do we measure and optimize for fairness in ensemble tokenization?

### 7. Training Efficiency
- What is the computational cost of training multiple tokenizers vs a single large vocabulary?
- Can tokenizers be trained in parallel or must they be sequential?
- Are there opportunities for transfer learning across ensemble components?

### 8. Vocabulary Size Trade-offs
- What is the total effective vocabulary size of an ensemble?
- How does ensemble vocabulary size compare to single tokenizer performance?
- Is there redundancy across ensemble components that could be eliminated?

### 9. Integration with Modern Architectures
- How do ensemble tokenizers interact with byte-level architectures like BLT?
- Can ensembles be integrated into end-to-end differentiable systems?
- What architectural modifications are needed to leverage ensemble tokenizations effectively?

### 10. Evaluation Metrics
- How should ensemble tokenization quality be measured?
- Are downstream task performance metrics sufficient or do we need tokenization-specific metrics?
- How do we balance compression efficiency, fairness, and model performance?

### 11. Scaffold Token Handling in Ensembles
- Should all ensemble components use scaffold token removal?
- Do scaffold tokens create useful diversity in ensembles?
- How do scaffold tokens affect ensemble aggregation?

### 12. Boundary Constraint Diversity
- What is the optimal mix of standard, boundless, and other boundary-constrained tokenizers?
- How do boundary constraints interact with different languages and scripts?
- Can adaptive boundary constraints improve ensemble performance?

### 13. Generalization to New Domains
- How well do ensembles generalize to domains not represented in training data?
- Do ensembles reduce the need for domain-specific tokenizer retraining?
- What diversity in training domains optimizes ensemble robustness?

### 14. Morphological Richness
- How do ensembles handle morphologically rich languages better than single tokenizers?
- Should ensemble components specialize in different morphological patterns?
- Can ensembles capture both stem and affix patterns effectively?

### 15. Future of Tokenization
- Will fixed tokenization remain relevant as byte-level models advance?
- How should ensemble approaches adapt to incorporate dynamic patching?
- What is the long-term role of tokenization in language model architectures?

## References

### Academic Papers

1. Mielke, S. J., Aicher, Z., Salesky, E., et al. (2021). Between words and characters: A Brief History of Open-Vocabulary Modeling and Tokenization in NLP. arXiv:2112.10508. https://arxiv.org/abs/2112.10508

2. Glocker, S., Lopez, B., & Leong, W. (2025). Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier. COLM 2025. arXiv:2504.00178. https://arxiv.org/abs/2504.00178

3. Wang, Z., et al. (2024). Scaffold-BPE: Enhancing Byte Pair Encoding with Simple and Effective Scaffold Token Removal. arXiv:2404.17808. https://arxiv.org/abs/2404.17808

4. Petrov, S., et al. (2025). Parity-Aware Byte Pair Encoding: Improving Cross-lingual Fairness in Tokenization. arXiv:2508.04796. https://arxiv.org/abs/2508.04796

5. Fickinger, A., et al. (2024). Byte Latent Transformer: Patches Scale Better Than Tokens. arXiv:2412.09871. https://arxiv.org/abs/2412.09871

6. Petrov, S., et al. (2023). Language Model Tokenizers Introduce Unfairness Between Languages. arXiv:2305.15425. https://arxiv.org/abs/2305.15425

7. Schuster, M., & Nakajima, K. (2012). Japanese and Korean Voice Search. IEEE International Conference on Acoustics, Speech and Signal Processing.

8. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI Blog.

### Technical Documentation and Blog Posts

9. Hugging Face Team. (2024). Summary of the tokenizers. Transformers Documentation. https://huggingface.co/docs/transformers/en/tokenizer_summary

10. Raschka, S. (2025). Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch. https://sebastianraschka.com/blog/2025/bpe-from-scratch.html

11. Karpathy, A. (2024). MinBPE: Minimal, clean code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. GitHub Repository. https://github.com/karpathy/minbpe

12. Dabbura, I. (2024). Tokenization Uncovered: How BPE Shapes the Mind of a Language Model. https://imaddabbura.github.io/posts/nlp/BPE-Tokenizer.html

13. Towards Data Science. (2024). Word, Subword, and Character-Based Tokenization: Know the Difference. https://towardsdatascience.com/word-subword-and-character-based-tokenization-know-the-difference-ea0976b64e17/

14. Chiusano, F. (2024). Two minutes NLP — A Taxonomy of Tokenization Methods. Medium. https://medium.com/nlplanet/two-minutes-nlp-a-taxonomy-of-tokenization-methods-60e330aacad3

15. Complete Guide to Subword Tokenization Methods in the Neural Era. (2024). Octanove Blog. https://blog.octanove.org/guide-to-subword-tokenization/

16. Adli, D. (2025). BPE vs WordPiece vs SentencePiece: A Beginner-Friendly Guide to Subword Tokenization. Medium. https://medium.com/@dhiyaadli/bpe-vs-wordpiece-vs-sentencepiece-a-beginner-friendly-guide-to-subword-tokenization-8047b39d82e0

17. Macijauskas, A. (2024). Tokenizers deep dive. Personal Website. https://augustasmacijauskas.github.io/personal-website/posts/tokenizers-deep-dive/tokenizers-deep-dive.html

### Code Repositories

18. Hugging Face Transformers. (2024). GPT-2 Tokenization Implementation. https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/tokenization_gpt2.py

19. OpenAI. (2019). GPT-2 Encoder Implementation. https://github.com/openai/gpt-2/blob/master/src/encoder.py

20. Google SentencePiece. (2024). Unsupervised text tokenizer for Neural Network-based text generation. https://github.com/google/sentencepiece

### Additional Resources

21. Njoroge, K. (2024). Character vs. Word Tokenization in NLP: Unveiling the Trade-Offs in Model Size, Parameters, and Compute. https://njoroge.tomorrow.co.ke/blog/ai/word_vs_character_level_tokenization

22. Trott, S. (2024). Tokenization in large language models, explained. Substack. https://seantrott.substack.com/p/tokenization-in-large-language-models

23. Saturn Cloud Blog. (2024). How Does Byte-Level BPE Algorithm in GPT-2 and RoBERTa Work? https://saturncloud.io/blog/how-does-bytelevel-bpe-algorithm-in-gpt2-and-roberta-work/

24. Pochetti, F. (2024). Byte Pair Encoding: building the GPT tokenizer with Karpathy. https://francescopochetti.com/byte-pair-encoding-building-the-gpt-tokenizer-with-karpathy/

25. Papers with Code. (2024). BPE Explained. https://paperswithcode.com/method/bpe

26. Byte Pair Encoding, Wikipedia. https://en.wikipedia.org/wiki/Byte_pair_encoding
