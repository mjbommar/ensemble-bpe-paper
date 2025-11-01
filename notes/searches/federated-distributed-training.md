# Federated and Distributed Tokenizer Training: A Comprehensive Review

## Summary

Federated and distributed tokenizer training represents an emerging and critical area of research at the intersection of privacy-preserving machine learning and fundamental NLP infrastructure. While traditional tokenizer training requires centralized access to complete training corpora, modern applications increasingly demand approaches that can train tokenizers across distributed datasets without compromising privacy or requiring massive data centralization.

This research reveals three distinct but complementary threads: (1) **privacy-preserving federated tokenizer training** using differential privacy techniques to train tokenizers without direct access to private data, (2) **parallel and GPU-accelerated BPE implementations** that dramatically improve tokenization throughput for high-batch inference workloads, and (3) **distributed vocabulary building systems** that scale to vocabularies with hundreds of millions of terms. The most significant finding is that federated approaches can achieve performance within 1% of "oracle" tokenizers trained on centralized private data while consuming zero additional privacy budget through clever application of differential privacy's postprocessing guarantee.

The implications for ensemble BPE tokenization are substantial: these techniques enable training multiple diverse tokenizers across heterogeneous data sources without requiring data centralization, potentially improving tokenizer robustness and domain adaptation while preserving privacy. The combination of federated learning, batched merge operations, and GPU parallelization creates a pathway toward efficient, privacy-preserving ensemble tokenizer training at scale.

## Key Findings

- **Zero-budget federated tokenizer training**: Bagdasaryan et al. (2022) demonstrated that tokenizers can be trained indirectly through model sampling during federated learning, achieving performance within 1% of oracle tokenizers without consuming additional privacy budget via differential privacy's postprocessing guarantee.

- **Vocabulary mismatch penalty is severe**: Training tokenizers on mismatched public data instead of target private data results in approximately 20% perplexity degradation, making federated approaches essential when direct data access is restricted.

- **GPU parallelization achieves 2-2.5x speedup**: BlockBPE's GPU implementation demonstrates 2x faster throughput than tiktoken and 2.5x faster than HuggingFace Tokenizers on high-batch workloads by replacing CPU-bound regex pre-tokenization with byte-level pre-tokenization and parallelized GPU merge operations.

- **Batched merging enables laptop-scale training**: BatchBPE shows that hundreds of token pairs can be safely merged simultaneously during vocabulary building, making high-quality tokenizer training feasible on basic laptops through reduced computational overhead and memory footprint.

- **Column-wise partitioning enables massive vocabularies**: Distributed word2vec systems demonstrate that partitioning vectors by dimension rather than by word enables 1/20 to 1/100 relative bandwidth reduction, successfully scaling to 200 million vocabulary items in production at Yahoo Gemini.

- **Domain-specific federated tokenizers outperform general ones**: FedByteBPE research in finance demonstrates that federated learning approaches can create domain-specific tokenizers that preserve privacy while maintaining competitive performance for specialized applications.

- **Three approaches to differential privacy in NLP**: The field employs gradient perturbation, embedding vector perturbation, and ensemble model-based methods, each with distinct tradeoffs for privacy-utility balance.

## Relevant Research & Papers

### Training a Tokenizer for Free with Private Federated Learning
- **Authors**: Elena Bagdasaryan, Vitaly Shmatikov, and collaborators
- **Year**: 2022
- **Venue**: arXiv 2203.09943v1
- **Key Contributions**:
  - Proposed cyclical approach: sample from federated model → train tokenizer on samples → update embeddings
  - Leveraged differential privacy's postprocessing guarantee to avoid consuming privacy budget
  - Demonstrated subword tokenizers outperform word-level variants in federated contexts
  - Achieved within 1% of oracle tokenizer performance
- **Relevance to Ensemble BPE**: This approach could enable training multiple diverse BPE tokenizers across different federated datasets or organizations, each sampling from domain-specific models, creating a natural foundation for ensemble tokenization while preserving privacy.

### BlockBPE: Parallel BPE Tokenization
- **Authors**: Research team focused on GPU acceleration
- **Year**: 2024 (July)
- **Venue**: arXiv 2507.11941v1
- **Key Contributions**:
  - First GPU implementation of BPE tokenization
  - Replaced regex pre-tokenization with byte-level approach enabling parallelization
  - Achieved O(n·d) complexity vs O(n log n) for CPU implementations when d=1
  - Demonstrated optimal performance at batch sizes 256-1024 with longer sequences
  - Used cuCollections for concurrent GPU hashmaps and CCCL for block-wide operations
- **Relevance to Ensemble BPE**: GPU parallelization could enable efficient simultaneous training or application of multiple BPE tokenizers in an ensemble, particularly for inference workloads where multiple tokenization strategies need to process the same input in parallel.

### Batching BPE Tokenization Merges (BatchBPE)
- **Authors**: Alexander P. Morgan
- **Year**: 2024 (August)
- **Venue**: arXiv 2408.04653
- **Key Contributions**:
  - Proved hundreds of token pairs can be safely merged simultaneously
  - Pure Python open-source implementation for accessibility
  - Reduced memory footprint enabling training on consumer hardware
  - Validated through encoded length measurements across multiple vocabularies
- **Relevance to Ensemble BPE**: Batched merging could dramatically reduce the computational cost of training multiple BPE vocabularies for ensemble approaches, making it feasible to experiment with diverse tokenizer configurations even in resource-constrained environments.

### Federated Learning-Based Tokenizer for Domain-Specific Language Models in Finance
- **Authors**: Finance domain researchers
- **Year**: 2024
- **Venue**: Springer (Conference proceedings, 978-3-031-78538-2_1)
- **Key Contributions**:
  - FedByteBPE framework for privacy-preserving domain tokenizer training
  - Demonstrated federated learning viability for specialized domains
  - Local vocabulary building with centralized aggregation approach
  - Addressed financial industry's privacy and compliance requirements
- **Relevance to Ensemble BPE**: Domain-specific federated tokenizers could serve as specialized components in an ensemble, where different tokenizers handle different domains (finance, medical, legal) while trained on private data sources without centralization.

### Network-Efficient Distributed Word2vec Training System for Large Vocabularies
- **Authors**: Distributed systems researchers (Yahoo research team)
- **Year**: 2016
- **Venue**: arXiv 1606.08495
- **Key Contributions**:
  - Column-wise vector partitioning across parameter servers
  - Server-side computation of dot products and linear combinations
  - 1/20 to 1/100 bandwidth reduction vs conventional approaches
  - Production deployment with 200M vocabulary items
  - 1.6M input words/second processing rate (300-dim vectors, 15 shards)
  - 2.44% query coverage improvement and 9.39% revenue increase in A/B testing
- **Relevance to Ensemble BPE**: The architectural principles of distributing vocabulary components across servers could inform how ensemble tokenizers distribute and aggregate information from multiple BPE models, particularly when dealing with massive combined vocabularies.

### Differentially Private Natural Language Models: Recent Advances and Future Directions
- **Authors**: Lijie Hu, Ivan Habernal, Lei Shen, Di Wang
- **Year**: 2023
- **Venue**: arXiv 2301.09112
- **Key Contributions**:
  - First systematic review of differential privacy in NLP
  - Identified three main approaches: gradient perturbation, embedding perturbation, ensemble models
  - Highlighted unique challenges in DP-NLP vs standard DP deep learning
  - Survey of techniques to prevent reconstruction attacks and side-channel threats
- **Relevance to Ensemble BPE**: Ensemble model-based differential privacy approaches directly relate to combining multiple tokenizers, suggesting that ensemble tokenization may inherently provide privacy benefits beyond single-tokenizer approaches.

### Privacy-Preserving Federated Learning and NLP Applications
- **Various sources from 2024 research**
- **Key Topics Covered**:
  - Rolling-hash-based text representation for client-side encoding
  - Bitwise quantization and local differential privacy (LDP)
  - Three privacy preservation types: encryption-based, perturbation-based, masking-based
  - Cross-cloud data harmonization and dynamic model aggregation
  - PrivateNLP@ACL 2024 workshop discussions
- **Relevance to Ensemble BPE**: These privacy-preserving techniques could enable secure aggregation of tokenization statistics across multiple organizations or data sources, essential for training ensemble tokenizers on sensitive data.

## Technical Details

### Federated Tokenizer Training Architecture

The core innovation in federated tokenizer training involves a cyclical process that bypasses the traditional requirement for direct corpus access:

1. **Model Sampling Phase**: During private federated learning, generate synthetic sequences by sampling from the current language model
2. **Tokenizer Training Phase**: Train a new tokenizer (BPE, WordPiece, etc.) on the sampled synthetic sequences
3. **Embedding Update Phase**: Update the model's embedding layer to accommodate the new vocabulary
4. **Resume Training Phase**: Continue federated learning with the improved tokenizer

**Privacy Guarantee**: This approach leverages differential privacy's postprocessing property—since the sampled sequences are generated from a DP-protected model, any computation on those samples (including tokenizer training) requires no additional privacy budget.

**Performance Characteristics**:
- Within 1% of oracle tokenizer performance
- Eliminates the 20% perplexity penalty from vocabulary mismatch
- Subword tokenizers (BPE, WordPiece) outperform word-level approaches in federated settings

### GPU-Parallel BPE Implementation (BlockBPE)

**Core Algorithm**:
Each GPU thread block processes one string with the following per-thread operations:
1. Read tokens at positions `i` and `i+1`
2. Query merge table for token pair rank
3. Track minimum rank across threads (block-level synchronization)
4. Mark merge positions in binary vector
5. Apply exclusive prefix sum for compaction
6. Write merged/unmerged tokens to new positions

**Complexity Analysis**:
- Time complexity: O(n·d) where n = block size, d = number of strides
- When d=1 (sequence fits in block): approaches O(n)
- Traditional CPU implementation: O(n log n)
- Block size constraint: maximum 1024 threads per GPU block on current hardware

**Performance Optimization Points**:
- Small batches × long sequences (64 × 2048/4096): use larger block size (1024)
- Large batches × short sequences (1024 × 128/256): use smaller block size (256)
- Byte-level pre-tokenization avoids regex bottleneck (75% of CPU runtime)

**Trade-offs**:
- General tasks (MMLU, GPQA, AGIEval): 98.9-99.9% similarity with standard tokenization
- Math tasks (GSM8K): 56% accuracy drop due to different numeric tokenization

### Batched Merge Operations

**Safe Batching Theorem**: Multiple token pairs can be merged simultaneously if they don't create conflicts in the merge sequence. The algorithm identifies all pairs with rank below a threshold and merges them in a single pass.

**Implementation Approach**:
```
For each vocabulary building iteration:
1. Identify all candidate pairs for merging
2. Compute merge ranks for all candidates
3. Select non-conflicting subset below rank threshold
4. Apply all selected merges simultaneously
5. Update frequency statistics
```

**Benefits**:
- Hundreds of pairs merged per iteration vs single pair
- Reduces iteration count dramatically
- Lower memory footprint through optimized data structures
- Enables training on consumer hardware (laptops)

### Distributed Vocabulary Architecture

**Column-wise Partitioning Strategy**:
Traditional approach: Store complete vectors for vocabulary subset on each server
Novel approach: Store dimension subset for ALL vocabulary words on each server

**Advantages**:
- Linear scaling with number of shards (S)
- Bandwidth reduction factor: S/d (where d = vector dimension)
- Typical reduction: 1/20 to 1/100 vs conventional parameter servers
- No single-server memory bottleneck for massive vocabularies

**RPC Interface**:
- `dotprod()`: Compute partial dot products on each shard, aggregate results
- `adjust()`: Apply gradient updates to dimension partitions
- Server-side negative sampling using seeded RNGs (eliminates index transmission)

**Production Performance**:
- 1.6M words/second (300-dim vectors, 15 shards)
- 1.2M words/second (1000-dim vectors, 25 shards)
- Successfully handles 200M vocabulary items
- Weekly retraining cycle in production deployment

### Differential Privacy Integration

**Three Methodological Approaches**:

1. **Gradient Perturbation**: Add calibrated noise to model gradients during federated aggregation
   - Provides formal DP guarantees via privacy accounting
   - Requires careful noise calibration to balance privacy-utility tradeoff
   - Works with standard optimization algorithms

2. **Embedding Perturbation**: Apply privacy mechanisms to learned word/token representations
   - Protects against embedding inversion attacks
   - Can be applied post-training or during training
   - May impact downstream task performance

3. **Ensemble Models**: Combine multiple models trained on disjoint data partitions
   - PATE (Private Aggregation of Teacher Ensembles) framework
   - Privacy through aggregation and consensus
   - Natural fit for ensemble tokenization approaches

**Privacy Budget Management**:
- Standard approach: Allocate separate budgets for tokenizer and model training
- Postprocessing approach: Train tokenizer on model samples using zero additional budget
- Tradeoff: Privacy budget vs model utility vs training time

## Implications for Ensemble BPE Tokenization

### Direct Applications

1. **Privacy-Preserving Ensemble Training**: The federated tokenizer training approach enables creating ensemble BPE models where each component is trained on different private datasets without data centralization. Organizations could collaboratively build ensemble tokenizers while maintaining data sovereignty.

2. **Parallel Ensemble Inference**: BlockBPE's GPU parallelization techniques could simultaneously apply multiple BPE tokenizers to the same input, enabling efficient ensemble tokenization for inference workloads. The O(n·d) complexity with d=1 suggests near-linear scaling for parallel tokenizer application.

3. **Cost-Effective Ensemble Experimentation**: BatchBPE's laptop-scale training makes it feasible to train dozens or hundreds of diverse BPE vocabularies with different configurations, enabling comprehensive exploration of ensemble composition strategies without requiring HPC resources.

4. **Domain-Specialized Ensemble Components**: Following the finance tokenizer work, ensemble BPE could combine domain-specific tokenizers (medical, legal, financial, technical) where each component is trained via federated learning on domain-private data, creating robust general-purpose tokenization.

### Architectural Possibilities

**Federated Ensemble Architecture**:
```
Organization A → BPE_A (trained via federated sampling)
Organization B → BPE_B (trained via federated sampling)
Organization C → BPE_C (trained via federated sampling)
                 ↓
         Ensemble Aggregation Layer
                 ↓
         Combined Tokenization Output
```

**GPU-Parallel Ensemble Inference**:
```
Input Text → [BPE_1, BPE_2, ..., BPE_N] (parallel GPU blocks)
          ↓
    Consensus/Voting Mechanism
          ↓
    Final Token Sequence
```

**Distributed Vocabulary Management**:
For ensemble with N tokenizers, each with vocabulary V:
- Total vocabulary: potentially N×V terms
- Column-wise partitioning: distribute dimensions across shards
- Each shard handles all vocabularies for its dimension subset
- Bandwidth scales as S/(d×N) vs conventional approaches

### Privacy Benefits

1. **Multi-Level Privacy**: Each component tokenizer trained with differential privacy provides individual guarantees, and ensemble aggregation may provide additional privacy amplification
2. **Data Minimization**: Organizations contribute tokenizers rather than raw data, reducing privacy attack surface
3. **Robustness to Attacks**: Ensemble diversity may increase difficulty of vocabulary reconstruction attacks targeting any single tokenizer

### Performance Considerations

**Training Efficiency**:
- Batched merging: Train N tokenizers with ~constant factor overhead vs single tokenizer
- Federated sampling: Parallelize across organizations/data sources
- GPU acceleration: Train multiple vocabularies simultaneously

**Inference Efficiency**:
- GPU parallelization: Apply N tokenizers with O(N·n·d) complexity
- Optimal block sizing: Adjust per-tokenizer based on sequence characteristics
- Voting/consensus overhead: Additional O(N·L) where L = output sequence length

**Quality Metrics**:
- Perplexity: Ensemble may reduce perplexity through diverse subword coverage
- Vocabulary coverage: Union of N vocabularies handles more edge cases
- Domain adaptation: Specialized tokenizers improve domain-specific performance

### Open Research Questions

1. **Optimal Ensemble Composition**: How many BPE tokenizers should be combined? How should their vocabularies differ (size, training data, merge strategy)?

2. **Consensus Mechanisms**: What voting/aggregation strategy works best for conflicting tokenization decisions across ensemble members?

3. **Privacy Amplification**: Does ensemble aggregation provide measurable privacy amplification beyond individual tokenizer guarantees?

4. **Incremental Update**: Can ensemble tokenizers be updated incrementally as new federated data becomes available without full retraining?

5. **Cross-Lingual Ensembles**: How do federated ensemble tokenizers perform when components are trained on different languages?

## Open Questions & Future Directions

### Privacy and Security

1. **Formal Privacy Analysis of Ensemble Aggregation**: While individual tokenizers may have DP guarantees, the privacy properties of ensemble consensus mechanisms remain understudied. Does voting/averaging across tokenizers provide privacy amplification, or could it create new attack surfaces?

2. **Adversarial Robustness**: Can malicious participants in federated tokenizer training poison the vocabulary or inject backdoors? How can ensemble diversity mitigate such attacks?

3. **Membership Inference Resistance**: To what extent can attackers determine if specific documents were used in training by querying ensemble tokenizers vs single tokenizers?

4. **Cross-Organization Privacy Leakage**: In federated settings, does comparing tokenizers trained by different organizations reveal information about their respective datasets?

### Scalability and Performance

1. **Massive-Scale Ensemble Training**: Current work demonstrates 200M vocabulary items for single tokenizers. Can distributed architectures support ensembles with billions of combined vocabulary entries?

2. **Streaming/Online Ensemble Updates**: Can ensemble BPE tokenizers adapt to streaming data in federated settings without periodic full retraining?

3. **Heterogeneous Hardware Optimization**: How should ensemble tokenizer training and inference be optimized for mixed CPU/GPU/TPU environments in federated deployments?

4. **Communication-Efficient Aggregation**: What compression techniques minimize bandwidth for aggregating tokenizer statistics across federated participants?

### Algorithmic Innovations

1. **Beyond BPE**: Can federated and distributed approaches extend to other tokenization algorithms (Unigram, WordPiece, SentencePiece variants)? Would ensemble combinations of different algorithms outperform homogeneous BPE ensembles?

2. **Adaptive Ensemble Selection**: Can models learn to dynamically select which ensemble component(s) to use for different inputs, domains, or tasks?

3. **Hierarchical Ensembles**: Would multi-level ensembles (ensembles of ensembles) provide benefits for handling diverse data distributions?

4. **Joint Optimization**: Current approaches train tokenizers then models separately. Can end-to-end federated training jointly optimize ensemble tokenizers and language models?

### Evaluation and Benchmarking

1. **Standard Metrics for Ensemble Tokenizers**: What evaluation metrics best capture the benefits of ensemble tokenization beyond perplexity (diversity, coverage, robustness)?

2. **Cross-Domain Benchmarks**: Existing benchmarks focus on single-domain evaluation. New benchmarks are needed for heterogeneous multi-domain scenarios where ensemble tokenizers should excel.

3. **Privacy-Utility Tradeoff Quantification**: How can we systematically measure the privacy-utility frontier for ensemble tokenizers with varying privacy budgets and ensemble sizes?

4. **Fairness and Bias**: Do ensemble tokenizers reduce or amplify biases compared to single tokenizers? How does federated training across diverse organizations impact representation of underrepresented languages/dialects?

### Practical Deployment

1. **Tooling and Infrastructure**: What open-source frameworks are needed to make federated ensemble tokenizer training accessible to practitioners without distributed systems expertise?

2. **Cost-Benefit Analysis**: For what data scales and privacy requirements do federated ensemble approaches become cost-effective vs centralized training with privacy-preserving techniques?

3. **Regulatory Compliance**: How do ensemble tokenizers align with GDPR, CCPA, and other privacy regulations? Do federated approaches provide compliance advantages?

4. **Interoperability**: Can ensemble tokenizers trained in federated settings be shared, composed, or extended across organizational boundaries while maintaining privacy?

### Domain-Specific Challenges

1. **Low-Resource Languages**: Can federated ensemble approaches improve tokenization for low-resource languages by aggregating sparse data across institutions without centralization?

2. **Multilingual Ensembles**: How should ensemble tokenizers handle code-switching and multilingual text? Should ensemble components specialize by language?

3. **Scientific and Technical Domains**: Can domain-specific ensemble tokenizers (medical, legal, scientific) outperform general-purpose tokenizers while preserving proprietary corpus privacy?

4. **Time-Varying Vocabularies**: How can ensemble tokenizers track evolving language (new terminology, slang, technical terms) in federated settings where participants observe different temporal distributions?

### Theoretical Understanding

1. **Convergence Guarantees**: Under what conditions do federated ensemble tokenizer training algorithms converge? What convergence rates can be guaranteed?

2. **Approximation Bounds**: How closely can federated ensemble tokenizers approximate oracle tokenizers with full centralized data access?

3. **Sample Complexity**: What sample sizes are required at each federated participant to achieve target ensemble quality? How does this scale with number of participants and ensemble size?

4. **Information-Theoretic Limits**: What are the fundamental limits of tokenizer quality achievable under differential privacy constraints in federated settings?

## References

### Academic Papers

1. Bagdasaryan, E., Shmatikov, V., et al. (2022). "Training a Tokenizer for Free with Private Federated Learning." arXiv:2203.09943v1. https://arxiv.org/abs/2203.09943v1 | https://openreview.net/pdf?id=rhz7nqYfF-q

2. BlockBPE Research Team (2024). "BlockBPE: Parallel BPE Tokenization." arXiv:2507.11941v1. https://arxiv.org/html/2507.11941v1

3. Morgan, A. P. (2024). "Batching BPE Tokenization Merges." arXiv:2408.04653. https://arxiv.org/abs/2408.04653

4. Finance Domain Researchers (2024). "Federated Learning-Based Tokenizer for Domain-Specific Language Models in Finance." Springer Conference Proceedings. https://link.springer.com/chapter/10.1007/978-3-031-78538-2_1

5. Yahoo Research Team (2016). "Network-Efficient Distributed Word2vec Training System for Large Vocabularies." arXiv:1606.08495. https://ar5iv.labs.arxiv.org/html/1606.08495

6. Hu, L., Habernal, I., Shen, L., & Wang, D. (2023). "Differentially Private Natural Language Models: Recent Advances and Future Directions." arXiv:2301.09112. https://arxiv.org/abs/2301.09112

7. Kudo, T., & Richardson, J. (2018). "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing." ACL 2018. https://aclanthology.org/D18-2012/ | https://arxiv.org/abs/1808.06226

### Technical Resources

8. HuggingFace Tokenizers Library. "Fast State-of-the-Art Tokenizers optimized for Research and Production." https://github.com/huggingface/tokenizers | https://huggingface.co/docs/tokenizers/quicktour

9. Google SentencePiece. "Unsupervised text tokenizer for Neural Network-based text generation." https://github.com/google/sentencepiece

10. Gautier, D. "bpeasy: Fast bare-bones BPE for modern tokenizer training." https://github.com/gautierdag/bpeasy

### Survey and Review Articles

11. ScienceDirect (2023). "Privacy-preserving Federated Learning and its application to natural language processing." https://www.sciencedirect.com/science/article/pii/S0950705123002253

12. ScienceDirect (2021). "Privacy preservation in federated learning: An insightful survey from the GDPR perspective." https://www.sciencedirect.com/science/article/pii/S0167404821002261

13. arXiv (2025). "Federated Learning: A Survey on Privacy-Preserving Collaborative Intelligence." https://arxiv.org/html/2504.17703

### Workshop and Community Resources

14. PrivateNLP@ACL 2024 Workshop. https://sites.google.com/view/privatenlp/

15. NIST (2024). "The UK-US Blog Series on Privacy-Preserving Federated Learning: Introduction." https://www.nist.gov/blogs/cybersecurity-insights/uk-us-blog-series-privacy-preserving-federated-learning-introduction

16. ACM Queue. "Federated Learning and Privacy." https://queue.acm.org/detail.cfm?id=3501293

### Additional Technical Articles

17. OpenReview. "Training a Tokenizer for Free with Private Federated Learning." https://openreview.net/pdf?id=rhz7nqYfF-q

18. Towards Data Science. "SentencePiece Tokenizer Demystified." https://towardsdatascience.com/sentencepiece-tokenizer-demystified-d0a3aac19b15/

19. MartinLwx's Blog. "BPE Tokenization Demystified: Implementation and Examples." https://martinlwx.github.io/en/the-bpe-tokenizer/

20. FreeCodeCamp. "How to Train BPE, WordPiece, and Unigram Tokenizers from Scratch using Hugging Face." https://www.freecodecamp.org/news/train-algorithms-from-scratch-with-hugging-face/

21. Sebastian Raschka. "Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch." https://sebastianraschka.com/blog/2025/bpe-from-scratch.html

### Related Research Areas

22. arXiv (2024). "Data Mixture Inference: What do BPE Tokenizers Reveal about their Training Data?" https://arxiv.org/html/2407.16607v2

23. arXiv (2024). "Does Differential Privacy Impact Bias in Pretrained NLP Models?" https://arxiv.org/abs/2410.18749

24. arXiv (2025). "Research on Large Language Model Cross-Cloud Privacy Protection and Collaborative Training based on Federated Learning." https://arxiv.org/html/2503.12226v1

25. Nature Scientific Reports (2025). "Federated learning with differential privacy for breast cancer diagnosis enabling secure data sharing and model integrity." https://www.nature.com/articles/s41598-025-95858-2
