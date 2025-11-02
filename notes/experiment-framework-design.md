# Ensemble BPE Experimentation Framework: Comprehensive Design

## Executive Summary

This document outlines a complete experimentation framework for ensemble tokenization research, synthesizing findings from 100+ papers across 20 research areas. The framework is designed to be implemented in Rust or Python using HuggingFace's `tokenizers` library as foundation, with systematic exploration of ensemble construction, combination, and evaluation strategies.

**Framework Goals:**
1. Test whether ensemble BPE can exceed APX-completeness bounds (0.333-0.625)
2. Evaluate statistical consistency (κ∘τ∘p⋆ = p⋆) of ensemble approaches
3. Measure Zipf alignment (R² > 0.93) and compression efficiency
4. Compare ensemble vs. single tokenizer across diverse tasks/domains
5. Identify optimal ensemble configurations for different use cases

**Implementation Strategy:**
- Build on HuggingFace `tokenizers` (Rust core, Python bindings)
- Modular design enabling component swapping
- Comprehensive evaluation harness
- Reproducible experimentation with clear baselines

---

## Part I: Core Infrastructure

### 1.1 Base Tokenizer Abstraction

**Purpose**: Unified interface for all tokenizer types to enable mix-and-match ensemble construction.

**Interface Design**:
```rust
trait TokenizerComponent {
    // Training
    fn train(&mut self, corpus: &Corpus, config: &TrainConfig) -> Result<()>;
    fn train_incremental(&mut self, batch: &[String]) -> Result<()>;

    // Encoding/Decoding
    fn encode(&self, text: &str) -> Vec<TokenId>;
    fn decode(&self, tokens: &[TokenId]) -> String;
    fn encode_batch(&self, texts: &[String]) -> Vec<Vec<TokenId>>;

    // Vocabulary Access
    fn vocabulary(&self) -> &Vocabulary;
    fn merge_rules(&self) -> &MergeRules;
    fn vocab_size(&self) -> usize;

    // Statistics
    fn token_frequencies(&self, corpus: &Corpus) -> HashMap<TokenId, u64>;
    fn compression_ratio(&self, corpus: &Corpus) -> f64;

    // Serialization
    fn save(&self, path: &Path) -> Result<()>;
    fn load(path: &Path) -> Result<Self>;
}
```

**Implementations**:
- `StandardBPE`: HuggingFace BPE
- `Unigram`: HuggingFace Unigram LM
- `WordPiece`: HuggingFace WordPiece
- `MorphologicalBPE`: BPE with morphology constraints
- `PickyBPE`: BPE with IoS-based filtering (from vocabulary-pruning-filtering.md)
- `ScaffoldBPE`: BPE with scaffold token removal
- `ByteLevelBPE`: Byte-level encoding
- `BoundlessBPE`: Without pre-tokenization constraints (from byte-character-level.md)

**Reference from notes**:
> "BPE Gets Picky introduces Intersection over Self (IoS) metric to dynamically remove intermediate tokens during training, achieving 2-10% token removal while maintaining performance"
> — From vocabulary-pruning-filtering.md

---

### 1.2 Corpus Management

**Purpose**: Efficient data handling for training and evaluation.

**Data Structures**:
```rust
struct Corpus {
    samples: Vec<String>,
    metadata: CorpusMetadata,
    partitions: Option<Vec<Partition>>,
}

struct CorpusMetadata {
    domain: String,
    language: String,
    size_bytes: u64,
    encoding: String,
}

struct Partition {
    id: usize,
    sample_indices: Vec<usize>,
    strategy: PartitionStrategy,
}
```

**Partitioning Strategies** (from background-01.md and bootstrap-bagging.md):
1. **Random Bootstrap**: Sample with replacement (classic bagging)
2. **Disjoint Chunks**: Non-overlapping partitions
3. **Stratified Sampling**: Balance domains/languages
4. **Overlapping Windows**: Sliding windows with configurable overlap
5. **Domain-Specific**: One partition per domain
6. **Temporal Split**: By time period (for evolving corpora)
7. **Morphological Complexity**: Partition by morphological richness
8. **Script-Based**: Partition by writing system (for multilingual)

**Reference from notes**:
> "Bootstrap aggregating for text preprocessing: Traditional bagging hasn't been directly applied to tokenization, but subword regularization and byte-level ensemble frameworks provide parallel approaches"
> — From bootstrap-bagging.md

---

### 1.3 Vocabulary Representation

**Purpose**: Efficient storage and manipulation of vocabularies and merge rules.

**Core Types**:
```rust
struct Vocabulary {
    tokens: Vec<Token>,
    token_to_id: HashMap<String, TokenId>,
    id_to_token: HashMap<TokenId, String>,
}

struct Token {
    id: TokenId,
    surface_form: String,
    frequency: u64,
    source_components: Option<Vec<TokenId>>, // For BPE merges
    metadata: TokenMetadata,
}

struct TokenMetadata {
    ios_score: Option<f64>,        // Intersection over Self
    scaffold_score: Option<f64>,    // Scaffold likelihood
    fertility: Option<f64>,         // Tokens per word
    morpheme_aligned: Option<bool>, // Morphological boundary alignment
    zipf_rank: Option<usize>,       // Position in frequency distribution
}

struct MergeRules {
    rules: Vec<MergeRule>,
    rule_order: HashMap<(TokenId, TokenId), usize>,
}

struct MergeRule {
    left: TokenId,
    right: TokenId,
    result: TokenId,
    priority: usize,
    frequency: u64,
    source_tokenizer: Option<usize>, // Which ensemble component
}
```

**Reference from notes**:
> "Vocabulary pruning should filter intermediate tokens **during** component tokenizer training using IoS or scaffold detection, not as post-processing"
> — From vocabulary-pruning-filtering.md

---

## Part II: Component Tokenizer Variants

### 2.1 Standard Algorithms (Baselines)

**BPE Family**:
1. **Frequency-Optimized BPE**: Standard greedy frequency-based
   - Baseline for comparison
   - Reference: Sennrich et al. 2016

2. **Random BPE**: Stochastic merge selection
   - Random-softmax: Probabilistic sampling
   - Random-uniform: Uniform selection
   - Reference: Sälevä & Lignos 2023 (from background-01.md)
   - Purpose: Establish equivalence class baseline

3. **Byte-Level BPE**: UTF-8 byte encoding
   - No pre-tokenization
   - Handles any text universally
   - Reference: GPT-2 approach (from byte-character-level.md)

**Alternative Algorithms**:
4. **Unigram LM**: EM-based probabilistic tokenization
   - Likelihood maximization objective
   - Viterbi decoding
   - Reference: Kudo 2018 (from sentencepiece-unigram.md)

5. **WordPiece**: Likelihood-based merging
   - P(ab) / (P(a) × P(b)) criterion
   - Reference: Google BERT (from wordpiece-algorithm.md)

**Reference from notes**:
> "Unigram consistently beats BPE on morphological alignment, suggesting ensemble should include at least one Unigram-based tokenizer"
> — From morphological-linguistic.md

---

### 2.2 Specialized Variants for Ensemble Components

**Optimization-Focused**:
1. **Compression-Optimized BPE**
   - Maximize compression ratio on specific domain
   - Track CPT (Characters Per Token) during training
   - Reference: compression-ratio.md findings (r = -0.976 correlation)

2. **Zipf-Aligned BPE**
   - Train to maximize R² on log-log frequency plot
   - Early stopping when R² > 0.93
   - Reference: He et al. 2025 (from frequency-distribution.md)

**Linguistic-Focused**:
3. **Morphology-Aware BPE**
   - Prefer merges at morpheme boundaries
   - Use morphological analyzer for guidance
   - Reference: MorphPiece (from morphological-linguistic.md)
   - 10x training efficiency improvement reported

4. **Linguistically-Motivated Unigram**
   - Initialize with morphologically-segmented vocabulary
   - Reference: TreeTok approach (from vocabulary-pruning-filtering.md)

**Quality-Focused**:
5. **Picky BPE with IoS Filtering**
   - Intersection over Self metric during training
   - Remove intermediate/scaffold tokens dynamically
   - IoS(t) = |contexts where t appears alone| / |total contexts with t|
   - Reference: Chizhov et al. 2024 (from vocabulary-pruning-filtering.md)

6. **Scaffold-Free BPE**
   - Detect and remove scaffold tokens
   - +0.5-0.6 BLEU improvements reported
   - Reference: Wang et al. 2024 (from vocabulary-pruning-filtering.md)

**Domain-Specialized**:
7. **Code-Optimized BPE**
   - Preserve identifier boundaries
   - Handle indentation/formatting robustly
   - Reference: TokDrift findings (from code-tokenization.md)

8. **Multilingual-Fair BPE**
   - Parity-aware merge selection
   - Minimize Gini coefficient across languages
   - 83% Gini reduction reported
   - Reference: Petrov et al. 2025 (from multilingual-allocation.md)

**Reference from notes**:
> "MorphPiece achieved comparable performance with only 50,000 training steps vs. 400,000-500,000 for GPT-2 (10x improvement), suggesting linguistically informed tokenization significantly reduces learning burden"
> — From morphological-linguistic.md

---

### 2.3 Multi-Granularity Components

**Purpose**: Capture patterns at different scales simultaneously.

**Granularity Levels**:
1. **Character-Level Component**: 1-2 char tokens (fine-grained)
2. **Subword Component**: Standard BPE (medium)
3. **Word-Level Component**: Prefer complete words (coarse)
4. **Phrase-Level Component**: Multi-word expressions (very coarse)

**Implementation**:
```rust
struct MultiGranularityEnsemble {
    character_level: TokenizerComponent,  // Vocab ~256-1K
    subword_level: TokenizerComponent,    // Vocab ~8K-32K
    word_level: TokenizerComponent,       // Vocab ~50K-100K
    weights: [f64; 3],                    // Learned or fixed
}
```

**Reference from notes**:
> "Token frequency distribution follows Zipf's law with different optimal vocabulary sizes by domain: 30K for NLP, 4K for genomics, 3K for chemistry"
> — From frequency-distribution.md

---

## Part III: Ensemble Construction Strategies

### 3.1 Data-Driven Partitioning

**Strategy 1: Bootstrap Aggregating (Bagging)**
```rust
fn bootstrap_ensemble(corpus: &Corpus, n_components: usize, vocab_size: usize)
    -> Vec<TokenizerComponent> {
    (0..n_components).map(|_| {
        let sample = corpus.bootstrap_sample(); // Sample with replacement
        let mut tokenizer = StandardBPE::new(vocab_size);
        tokenizer.train(&sample);
        tokenizer
    }).collect()
}
```

**Strategy 2: Disjoint Domain Partitioning**
```rust
fn domain_ensemble(corpus: &Corpus, domains: &[String], vocab_size: usize)
    -> Vec<TokenizerComponent> {
    domains.iter().map(|domain| {
        let domain_corpus = corpus.filter_by_domain(domain);
        let mut tokenizer = StandardBPE::new(vocab_size);
        tokenizer.train(&domain_corpus);
        tokenizer
    }).collect()
}
```

**Strategy 3: Stratified Sampling by Language**
```rust
fn multilingual_ensemble(corpus: &Corpus, languages: &[String], vocab_size: usize)
    -> Vec<TokenizerComponent> {
    languages.iter().map(|lang| {
        let lang_corpus = corpus.filter_by_language(lang);
        let mut tokenizer = StandardBPE::new(vocab_size);
        tokenizer.train(&lang_corpus);
        tokenizer
    }).collect()
}
```

**Strategy 4: Temporal Partitioning**
- Split corpus by time periods
- Useful for evolving domains (news, social media)
- Tests temporal robustness

**Reference from notes**:
> "Federated tokenizer training enables privacy-preserving ensemble creation across organizations without data centralization, achieving within 1% of oracle performance"
> — From federated-distributed-training.md

---

### 3.2 Algorithm-Driven Diversity

**Strategy 5: Heterogeneous Algorithm Ensemble**
```rust
struct HeterogeneousEnsemble {
    bpe_component: StandardBPE,
    unigram_component: Unigram,
    wordpiece_component: WordPiece,
    morphological_component: MorphologicalBPE,
}
```

**Rationale**:
- BPE: Frequency-optimized compression
- Unigram: Likelihood-maximized morphology
- WordPiece: Likelihood-based pairs
- Morphological: Linguistically motivated

**Reference from notes**:
> "Heterogeneous ensembles (mixing BPE, Unigram, and morphological tokenizers) would be far more robust than homogeneous BPE-only ensembles"
> — From robustness-adversarial.md

**Strategy 6: Optimization-Objective Diversity**
```rust
struct OptimizationEnsemble {
    compression_optimizer: CompressionBPE,     // Maximize CPT
    zipf_optimizer: ZipfAlignedBPE,           // Maximize R²
    morphology_optimizer: MorphologicalBPE,    // Maximize boundary alignment
    performance_optimizer: AdaptiveBPE,        // Optimize via domain adaptation
}
```

---

### 3.3 Vocabulary-Size Diversity

**Strategy 7: Multi-Scale Vocabulary Ensemble**
```rust
fn multiscale_ensemble(corpus: &Corpus) -> Vec<TokenizerComponent> {
    vec![
        StandardBPE::new(2_000).train(&corpus),   // Small: 2K
        StandardBPE::new(8_000).train(&corpus),   // Sweet spot: 8K
        StandardBPE::new(32_000).train(&corpus),  // Standard: 32K
        StandardBPE::new(128_000).train(&corpus), // Large: 128K
    ]
}
```

**Rationale**:
- Small vocab (2-8K): Better morphological alignment, low-resource languages
- Medium vocab (32-50K): Industry standard, good balance
- Large vocab (100K+): Better compression, multilingual coverage

**Reference from notes**:
> "8K vocabulary optimal for small datasets (30K-1.3M examples); larger vocabularies (32K+) only benefit exceptionally large datasets (4.5M+ examples)"
> — From vocabulary-size.md

---

## Part IV: Ensemble Combination Methods

### 4.1 Vocabulary-Level Combination

**Method 1: Union with Conflict Resolution**
```rust
fn merge_vocabularies_union(components: &[TokenizerComponent],
                            final_size: usize) -> Vocabulary {
    let mut combined = Vocabulary::new();

    // Step 1: Collect all tokens from all components
    for component in components {
        for token in component.vocabulary().tokens() {
            combined.add_or_update(token);
        }
    }

    // Step 2: Resolve conflicts and rank
    let ranked_tokens = rank_tokens_by_strategy(
        &combined,
        RankingStrategy::WeightedVoting,
    );

    // Step 3: Select top final_size tokens
    Vocabulary::from_tokens(&ranked_tokens[..final_size])
}
```

**Ranking Strategies**:
1. **Frequency Sum**: ∑ᵢ frequency_i(token)
2. **Weighted Vote**: ∑ᵢ wᵢ · indicator_i(token)
3. **Average Rank**: mean(rank_i(token)) across components
4. **Consensus Threshold**: Include if present in ≥k components
5. **Quality-Weighted**: ∑ᵢ wᵢ · quality_i(token) where quality = IoS or compression

**Method 2: Intersection (Conservative)**
```rust
fn merge_vocabularies_intersection(components: &[TokenizerComponent]) -> Vocabulary {
    let mut result = components[0].vocabulary().clone();
    for component in &components[1..] {
        result = result.intersection(component.vocabulary());
    }
    result
}
```

**Method 3: K-of-N Voting**
```rust
fn merge_vocabularies_kofn(components: &[TokenizerComponent],
                           k: usize,
                           final_size: usize) -> Vocabulary {
    let mut token_votes: HashMap<String, usize> = HashMap::new();

    for component in components {
        for token in component.vocabulary().tokens() {
            *token_votes.entry(token.surface_form.clone()).or_insert(0) += 1;
        }
    }

    // Keep tokens with ≥k votes
    let selected: Vec<_> = token_votes.iter()
        .filter(|(_, &votes)| votes >= k)
        .collect();

    // Sort by votes, then frequency, select top final_size
    Vocabulary::from_weighted_tokens(selected, final_size)
}
```

**Reference from notes**:
> "Two-stage voting-boosting (2SVB) achieved 0.8942 F1 score by combining diverse models and learning from disagreements"
> — From voting-consensus.md

---

### 4.2 Merge-Rule Level Combination

**Method 4: Merge Rule Voting**
```rust
struct MergeRuleVoting {
    rules: HashMap<(TokenId, TokenId), MergeRuleVote>,
}

struct MergeRuleVote {
    pair: (TokenId, TokenId),
    votes: Vec<VoteInfo>,
    combined_frequency: u64,
    average_priority: f64,
}

struct VoteInfo {
    component_id: usize,
    priority: usize,           // When was it merged (iteration number)
    frequency: u64,            // How often did pair occur
    weight: f64,               // Component weight
}

fn merge_rules_by_voting(components: &[TokenizerComponent]) -> MergeRules {
    let mut voting = MergeRuleVoting::new();

    // Collect votes
    for (idx, component) in components.iter().enumerate() {
        for rule in component.merge_rules() {
            voting.add_vote(
                rule.pair(),
                VoteInfo {
                    component_id: idx,
                    priority: rule.priority,
                    frequency: rule.frequency,
                    weight: 1.0 / components.len() as f64, // Uniform initially
                }
            );
        }
    }

    // Rank by consensus
    let ranked = voting.rank_by_strategy(RankStrategy::AveragePriority);
    MergeRules::from_ranked(ranked)
}
```

**Ranking Strategies for Merge Rules**:
1. **Majority Voting**: Include if ≥50% of components merged this pair
2. **Average Priority**: Sort by mean(priority_i) across components
3. **Frequency-Weighted**: Sort by ∑ᵢ wᵢ · frequency_i
4. **Consensus Strength**: Weight by number of agreeing components
5. **Quality-Filtered**: Only include merges passing quality threshold (IoS, etc.)

**Reference from notes**:
> "Merge order by average rank: For each candidate merge (pair of symbols), record the iteration at which each model merged it (or ∞ if never). Then sort merges by average merge iteration."
> — From background-01.md

---

### 4.3 Inference-Time Combination

**Method 5: Multi-Tokenizer Parallel Inference**
```rust
struct ParallelEnsemble {
    components: Vec<TokenizerComponent>,
    voting_strategy: VotingStrategy,
}

impl ParallelEnsemble {
    fn encode(&self, text: &str) -> Vec<TokenId> {
        // Apply all tokenizers in parallel
        let tokenizations: Vec<Vec<TokenId>> = self.components
            .par_iter()
            .map(|c| c.encode(text))
            .collect();

        // Vote on best tokenization
        self.select_best_tokenization(tokenizations)
    }

    fn select_best_tokenization(&self, candidates: Vec<Vec<TokenId>>)
        -> Vec<TokenId> {
        match self.voting_strategy {
            VotingStrategy::ShortestSequence => {
                candidates.into_iter().min_by_key(|t| t.len()).unwrap()
            },
            VotingStrategy::HighestLikelihood => {
                candidates.into_iter()
                    .max_by_key(|t| self.compute_likelihood(t))
                    .unwrap()
            },
            VotingStrategy::ConsensusAlignment => {
                self.align_and_merge(candidates)
            },
        }
    }
}
```

**Method 6: Byte-Level Ensemble Framework**
```rust
// From bootstrap-bagging.md: Byte-level probability framework
// enables combining models with incompatible vocabularies

struct ByteLevelEnsemble {
    components: Vec<TokenizerComponent>,
    weights: Vec<f64>,
}

impl ByteLevelEnsemble {
    fn encode_probabilistic(&self, text: &str) -> Vec<TokenId> {
        // Convert to bytes
        let bytes = text.as_bytes();

        // Get probability distributions from each component
        let byte_probs: Vec<Vec<f64>> = self.components.iter().map(|c| {
            c.encode_to_byte_probabilities(bytes)
        }).collect();

        // Weighted ensemble at byte level
        let ensemble_probs = self.combine_byte_probabilities(byte_probs);

        // Decode from ensemble byte probabilities
        self.decode_from_byte_probs(&ensemble_probs)
    }
}
```

**Reference from notes**:
> "Byte-level ensemble framework (2024) enables ensembling models with incompatible vocabularies by converting to byte-space, yielding 3.7% improvements"
> — From bootstrap-bagging.md

---

### 4.4 Adaptive/Dynamic Combination

**Method 7: Context-Aware Routing**
```rust
struct AdaptiveEnsemble {
    components: Vec<TokenizerComponent>,
    router: ComponentRouter,
}

struct ComponentRouter {
    domain_classifier: DomainClassifier,
    routing_strategy: RoutingStrategy,
}

impl AdaptiveEnsemble {
    fn encode_adaptive(&self, text: &str) -> Vec<TokenId> {
        // Classify input
        let domain = self.router.classify(text);

        // Select best component(s) for domain
        let selected_components = self.router.route(domain);

        // Apply selected components
        if selected_components.len() == 1 {
            selected_components[0].encode(text)
        } else {
            self.ensemble_encode(&selected_components, text)
        }
    }
}
```

**Routing Strategies**:
1. **Domain-based**: Route to domain-specific component
2. **Confidence-based**: Use component most confident on input
3. **Quality-based**: Select by expected quality (compression, etc.)
4. **Ensemble always**: Use all components but weight by confidence

**Reference from notes**:
> "Adaptive tokenization for domain adaptation achieves 72x speedup over full retraining while maintaining 97% of performance"
> — From domain-adaptation.md

---

## Part V: Evaluation Framework

### 5.1 Intrinsic Metrics (Tokenizer Quality)

**Compression Metrics**:
```rust
struct CompressionMetrics {
    characters_per_token: f64,        // Higher is better
    bytes_per_token: f64,             // Higher is better
    fertility: f64,                   // Lower is better (tokens per word)
    normalized_sequence_length: f64,  // Lower is better
}

fn evaluate_compression(tokenizer: &impl TokenizerComponent,
                       corpus: &Corpus) -> CompressionMetrics {
    let tokenizations = corpus.samples.iter()
        .map(|text| tokenizer.encode(text))
        .collect::<Vec<_>>();

    let total_tokens: usize = tokenizations.iter().map(|t| t.len()).sum();
    let total_chars: usize = corpus.samples.iter().map(|s| s.chars().count()).sum();
    let total_bytes: usize = corpus.samples.iter().map(|s| s.len()).sum();

    CompressionMetrics {
        characters_per_token: total_chars as f64 / total_tokens as f64,
        bytes_per_token: total_bytes as f64 / total_tokens as f64,
        fertility: total_tokens as f64 / corpus.word_count() as f64,
        normalized_sequence_length:
            (total_tokens as f64 / total_chars as f64) * tokenizer.vocab_size() as f64,
    }
}
```

**Distribution Quality Metrics**:
```rust
struct DistributionMetrics {
    zipf_r_squared: f64,              // >0.93 is optimal
    frequency_correlation: f64,        // With ideal power law
    heaps_law_fit: f64,               // Vocabulary growth rate
    entropy: f64,                      // Token distribution entropy
}

fn evaluate_zipf_alignment(tokenizer: &impl TokenizerComponent,
                          corpus: &Corpus) -> f64 {
    let frequencies = tokenizer.token_frequencies(corpus);
    let mut ranked: Vec<_> = frequencies.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by frequency descending

    // Fit log(frequency) ~ log(rank) linear model
    let log_ranks: Vec<f64> = (1..=ranked.len()).map(|r| (r as f64).ln()).collect();
    let log_freqs: Vec<f64> = ranked.iter().map(|(_, f)| (*f as f64).ln()).collect();

    compute_r_squared(&log_ranks, &log_freqs)
}
```

**Vocabulary Quality Metrics**:
```rust
struct VocabularyMetrics {
    ios_distribution: Vec<f64>,           // IoS scores for all tokens
    scaffold_token_ratio: f64,            // % scaffold tokens
    under_trained_token_ratio: f64,       // % tokens with freq < threshold
    morphological_alignment: f64,         // % morpheme-aligned splits
    unique_token_coverage: f64,           // % corpus covered by unique tokens
}

fn evaluate_vocabulary_quality(tokenizer: &impl TokenizerComponent,
                               corpus: &Corpus,
                               morphology_gold: Option<&MorphologyGold>)
    -> VocabularyMetrics {
    // IoS scores (Intersection over Self)
    let ios_scores = compute_ios_scores(tokenizer, corpus);

    // Scaffold detection
    let scaffold_ratio = detect_scaffold_tokens(tokenizer, corpus);

    // Under-trained tokens
    let freq_threshold = 10; // Tokens appearing <10 times
    let under_trained = count_rare_tokens(tokenizer, corpus, freq_threshold);

    // Morphological alignment
    let morph_alignment = if let Some(gold) = morphology_gold {
        evaluate_morphological_alignment(tokenizer, corpus, gold)
    } else {
        0.0
    };

    VocabularyMetrics {
        ios_distribution: ios_scores,
        scaffold_token_ratio: scaffold_ratio,
        under_trained_token_ratio: under_trained,
        morphological_alignment: morph_alignment,
        unique_token_coverage: compute_coverage(tokenizer, corpus),
    }
}
```

**Reference from notes**:
> "Zipf alignment (R² > 0.93) provides quantitative vocabulary size selection criterion; compression correlates with performance at r = -0.976 to -0.996"
> — From frequency-distribution.md and compression-ratio.md

---

### 5.2 Consistency and Robustness Metrics

**Statistical Consistency**:
```rust
struct ConsistencyMetrics {
    round_trip_accuracy: f64,          // encode(decode(tokens)) == tokens
    distribution_preservation: f64,     // KL-divergence κ∘τ∘p vs. p
    determinism_score: f64,             // Consistency across runs
    adversarial_robustness: f64,        // Resistance to perturbations
}

fn evaluate_consistency(tokenizer: &impl TokenizerComponent,
                       corpus: &Corpus) -> ConsistencyMetrics {
    // Test κ∘τ∘p⋆ = p⋆ condition
    let original_dist = compute_distribution(corpus);

    let tokenized = corpus.samples.iter()
        .map(|text| tokenizer.encode(text))
        .collect::<Vec<_>>();

    let reconstructed = tokenized.iter()
        .map(|tokens| tokenizer.decode(tokens))
        .collect::<Vec<_>>();

    let reconstructed_dist = compute_distribution_from_texts(&reconstructed);

    ConsistencyMetrics {
        round_trip_accuracy: compute_round_trip_accuracy(&corpus.samples, &reconstructed),
        distribution_preservation: compute_kl_divergence(&original_dist, &reconstructed_dist),
        determinism_score: test_determinism(tokenizer, corpus),
        adversarial_robustness: test_adversarial_robustness(tokenizer, corpus),
    }
}
```

**Reference from notes**:
> "Statistical consistency condition κ∘τ∘p⋆=p⋆ rarely satisfied; ensemble could satisfy consistency by marginalizing over multiple tokenizations"
> — From robustness-adversarial.md and theoretical-vs-empirical-analysis.md

---

### 5.3 Cross-Lingual Fairness Metrics

**Equity Measures**:
```rust
struct FairnessMetrics {
    gini_coefficient: f64,                    // Lower is more fair (0 = perfect)
    max_min_fertility_ratio: f64,             // Worst/best language ratio
    parity_score: f64,                        // Deviation from equal CPT
    strr_by_language: HashMap<String, f64>,   // Single Token Retention Rate
}

fn evaluate_multilingual_fairness(tokenizer: &impl TokenizerComponent,
                                  multilingual_corpus: &MultilingualCorpus)
    -> FairnessMetrics {
    let mut fertility_by_lang = HashMap::new();

    for (lang, corpus) in multilingual_corpus.by_language() {
        let metrics = evaluate_compression(tokenizer, corpus);
        fertility_by_lang.insert(lang.clone(), metrics.fertility);
    }

    FairnessMetrics {
        gini_coefficient: compute_gini_coefficient(&fertility_by_lang),
        max_min_fertility_ratio:
            fertility_by_lang.values().max() / fertility_by_lang.values().min(),
        parity_score: compute_parity_deviation(&fertility_by_lang),
        strr_by_language: compute_strr_by_language(tokenizer, multilingual_corpus),
    }
}
```

**Reference from notes**:
> "Parity-aware BPE reduced Gini coefficient by 83% with negligible accuracy impact; Latin script achieves 2.61 CPT while Devanagari gets only 0.99 CPT (62% efficiency gap)"
> — From multilingual-allocation.md and compression-ratio.md

---

### 5.4 Extrinsic Metrics (Downstream Performance)

**Language Modeling**:
```rust
struct LanguageModelingMetrics {
    perplexity: f64,
    bits_per_character: f64,
    training_time: Duration,
    convergence_rate: f64,
}

fn evaluate_on_language_modeling(tokenizer: &impl TokenizerComponent,
                                 train_corpus: &Corpus,
                                 eval_corpus: &Corpus,
                                 model_config: &LMConfig)
    -> LanguageModelingMetrics {
    // Train language model with this tokenizer
    let tokenized_train = tokenizer.encode_batch(&train_corpus.samples);
    let model = train_language_model(&tokenized_train, model_config);

    // Evaluate on held-out data
    let tokenized_eval = tokenizer.encode_batch(&eval_corpus.samples);
    let perplexity = model.evaluate_perplexity(&tokenized_eval);

    LanguageModelingMetrics {
        perplexity,
        bits_per_character: compute_bpc(perplexity, tokenizer, eval_corpus),
        training_time: model.training_duration(),
        convergence_rate: model.convergence_speed(),
    }
}
```

**Machine Translation**:
```rust
struct TranslationMetrics {
    bleu: f64,
    chrf: f64,
    ter: f64,
    training_efficiency: f64, // Performance per compute hour
}
```

**Classification Tasks**:
```rust
struct ClassificationMetrics {
    accuracy: f64,
    f1_score: f64,
    inference_speed: f64, // Examples per second
}
```

**Reference from notes**:
> "Compression correlates with performance: Pearson r = -0.976 (compression vs. BLEU), r = -0.994 (compression vs. perplexity)"
> — From compression-ratio.md

---

### 5.5 Efficiency Metrics

**Computational Cost**:
```rust
struct EfficiencyMetrics {
    training_time: Duration,
    training_memory: usize,
    inference_latency: Duration,      // Per sample
    inference_throughput: f64,        // Samples per second
    vocabulary_memory: usize,
    merge_rules_memory: usize,
}

fn benchmark_efficiency(tokenizer: &impl TokenizerComponent,
                       corpus: &Corpus) -> EfficiencyMetrics {
    // Training time
    let train_start = Instant::now();
    let mut test_tokenizer = tokenizer.clone();
    test_tokenizer.train(corpus);
    let training_time = train_start.elapsed();

    // Inference speed
    let inference_start = Instant::now();
    let n_samples = 1000;
    for sample in corpus.samples.iter().take(n_samples) {
        tokenizer.encode(sample);
    }
    let inference_time = inference_start.elapsed();

    EfficiencyMetrics {
        training_time,
        training_memory: estimate_training_memory(tokenizer),
        inference_latency: inference_time / n_samples,
        inference_throughput: n_samples as f64 / inference_time.as_secs_f64(),
        vocabulary_memory: size_of_vocabulary(tokenizer.vocabulary()),
        merge_rules_memory: size_of_merge_rules(tokenizer.merge_rules()),
    }
}
```

---

## Part VI: Experimental Design

### 6.1 Baseline Comparisons

**Single-Tokenizer Baselines**:
1. **Standard BPE** (HuggingFace implementation)
   - Vocabulary sizes: 8K, 16K, 32K, 50K, 100K
   - Gold standard for comparison

2. **Unigram LM** (SentencePiece)
   - Best morphological baseline
   - Same vocabulary sizes as BPE

3. **WordPiece** (BERT-style)
   - Alternative merging criterion

4. **Random BPE**
   - Establish equivalence class baseline
   - Multiple random seeds

5. **Picky BPE** (State-of-art refinement)
   - Current best single-tokenizer approach

**Reference from notes**:
> "Random BPE performs comparably to greedy in 3/4 language pairs, establishing that exact merge order isn't critical"
> — From background-01.md

---

### 6.2 Ensemble Configurations to Test

**Size Ablation**:
```
M = {2, 3, 5, 10, 20, 50} ensemble components
```

**Combination Strategy Ablation**:
1. Union with frequency voting
2. Intersection (conservative)
3. K-of-N voting (k = 2, 3, 5)
4. Weighted voting (uniform vs. performance-based)
5. Byte-level ensemble
6. Parallel inference with consensus

**Component Diversity Ablation**:
1. Homogeneous BPE ensemble (all same algorithm, different data)
2. Heterogeneous ensemble (BPE + Unigram + WordPiece + Morphological)
3. Multi-scale ensemble (different vocab sizes)
4. Multi-objective ensemble (compression + morphology + Zipf)

**Data Partitioning Ablation**:
1. Random bootstrap (classic bagging)
2. Disjoint chunks (0% overlap)
3. Overlapping windows (25%, 50%, 75% overlap)
4. Domain-specific partitions
5. Language-specific partitions
6. Stratified by morphological complexity

---

### 6.3 Datasets and Domains

**Natural Language**:
1. **Wikipedia** (multilingual, general)
2. **Common Crawl** (web-scale)
3. **Books Corpus** (long-form, literary)
4. **News** (temporal evolution)
5. **Social Media** (Twitter/Reddit, informal)

**Specialized Domains**:
6. **Legal Text** (domain-specific terminology)
7. **Medical/Scientific** (technical vocabulary)
8. **Code** (Python, JavaScript, multi-language)
9. **Math** (equations, LaTeX)
10. **Multilingual Parallel** (WMT, Europarl)

**Low-Resource**:
11. **FLORES-200** (multilingual evaluation)
12. **AmericasNLP** (indigenous languages)
13. **MasakhaNER** (African languages)

**Reference from notes**:
> "Low-resource languages face 2-10x tokenization inefficiency, making AI services more expensive and less accessible"
> — From low-resource-languages.md

---

### 6.4 Specific Experiments

**Experiment 1: APX-Completeness Bound Testing**
```
Hypothesis: Ensemble BPE can exceed α=0.625 approximation bound

Setup:
- Train single BPE (baseline)
- Train ensemble with M={5, 10, 20} components
- Measure compression ratio vs. theoretical optimal (via exhaustive search on small corpus)

Metrics:
- Approximation factor α = achieved_compression / optimal_compression
- Test if α_ensemble > α_single
```

**Experiment 2: Statistical Consistency Verification**
```
Hypothesis: Ensemble satisfies κ∘τ∘p⋆ = p⋆ better than single tokenizers

Setup:
- Measure KL-divergence(p_original, κ∘τ∘p_tokenized)
- Compare single vs. ensemble

Metrics:
- KL divergence (lower is better)
- Round-trip reconstruction error
- Cross-distribution consistency
```

**Experiment 3: Zipf Alignment Optimization**
```
Hypothesis: Ensemble achieves higher R² than individual components

Setup:
- Train components with various objectives
- Measure R² for each component and ensemble

Metrics:
- R² on log-log frequency plot
- Compare to optimal threshold (R² > 0.93)
```

**Experiment 4: Domain Generalization**
```
Hypothesis: Ensemble generalizes better across domains

Setup:
- Train on domain A
- Test on domains B, C, D
- Measure performance degradation

Metrics:
- Cross-domain perplexity
- Compression ratio stability
- Out-of-domain sequence length increase
```

**Experiment 5: Multilingual Fairness**
```
Hypothesis: Ensemble reduces cross-lingual tokenization disparity

Setup:
- Train on multilingual corpus
- Measure per-language fertility/CPT
- Compare Gini coefficient

Metrics:
- Gini coefficient (target: <0.3)
- Max/min fertility ratio
- Per-language STRR
```

**Reference from notes**:
> "Ensemble approaches can balance multiple competing objectives: compression + morphology + equity + performance"
> — From low-resource-languages.md

**Experiment 6: Computational Cost-Benefit**
```
Question: Is ensemble worth the computational overhead?

Setup:
- Measure training time: single vs. M-component ensemble
- Measure inference latency
- Measure downstream task improvement

Metrics:
- Training time ratio
- Inference latency ratio
- Performance improvement per unit compute
- Pareto frontier analysis
```

**Experiment 7: Component Diversity Impact**
```
Question: How does component diversity affect ensemble quality?

Setup:
- Vary diversity: homogeneous BPE → heterogeneous algorithms
- Measure vocabulary overlap and performance

Metrics:
- Vocabulary Jaccard similarity (diversity measure)
- Ensemble performance vs. diversity correlation
- Disagreement rate on tokenization decisions
```

**Experiment 8: Ablation on Ensemble Size**
```
Question: What is optimal M (number of components)?

Setup:
- Vary M from 2 to 50
- Measure performance and cost scaling

Metrics:
- Performance vs. M curve
- Diminishing returns point
- Training time vs. M
- Memory usage vs. M
```

---

## Part VII: Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)

**Deliverables**:
- [ ] Base `TokenizerComponent` trait
- [ ] Corpus management with partitioning
- [ ] Vocabulary and MergeRules data structures
- [ ] Wrappers for HuggingFace tokenizers
- [ ] Basic evaluation metrics (compression, Zipf R²)

**Tech Stack**:
```
Language: Rust (for performance) with Python bindings (for ML integration)
Dependencies:
- tokenizers (HuggingFace)
- rayon (parallel processing)
- serde (serialization)
- ndarray (numerical computing)
- pyo3 (Python bindings)
```

---

### Phase 2: Component Implementations (Weeks 3-4)

**Deliverables**:
- [ ] Standard BPE, Unigram, WordPiece wrappers
- [ ] Picky BPE with IoS filtering
- [ ] Scaffold-BPE implementation
- [ ] Morphological-aware BPE (with optional morphology analyzer)
- [ ] Random BPE variants
- [ ] Byte-level BPE

---

### Phase 3: Ensemble Construction (Weeks 5-6)

**Deliverables**:
- [ ] Bootstrap ensemble builder
- [ ] Domain/language-specific ensemble builder
- [ ] Heterogeneous ensemble builder
- [ ] Multi-scale ensemble builder
- [ ] Data partitioning strategies

---

### Phase 4: Ensemble Combination (Weeks 7-8)

**Deliverables**:
- [ ] Vocabulary merging (union, intersection, k-of-n)
- [ ] Merge rule voting
- [ ] Byte-level ensemble framework
- [ ] Parallel inference with voting
- [ ] Adaptive routing

---

### Phase 5: Evaluation Harness (Weeks 9-10)

**Deliverables**:
- [ ] All intrinsic metrics
- [ ] Consistency/robustness testing
- [ ] Cross-lingual fairness metrics
- [ ] Efficiency benchmarking
- [ ] Visualization tools (R² plots, fertility distributions)

---

### Phase 6: Integration & Experiments (Weeks 11-14)

**Deliverables**:
- [ ] Language modeling integration
- [ ] Machine translation integration
- [ ] Classification task integration
- [ ] Run all experiments (§6.4)
- [ ] Generate plots and tables for paper

---

### Phase 7: Paper Writing (Weeks 15-16)

**Deliverables**:
- [ ] Results analysis
- [ ] Paper draft
- [ ] Ablation studies
- [ ] Supplementary materials

---

## Part VIII: Expected Contributions

### 8.1 Empirical Contributions

1. **First systematic study of ensemble tokenization**
   - Comprehensive comparison of combination strategies
   - Ablation across design dimensions

2. **APX-completeness bound empirical testing**
   - Can ensemble exceed α=0.625?
   - Characterization of when/why

3. **Statistical consistency verification**
   - Empirical test of κ∘τ∘p⋆ = p⋆
   - Conditions for ensemble consistency

4. **Cross-domain generalization analysis**
   - Quantify ensemble robustness advantage
   - Domain-specific vs. general ensemble trade-offs

5. **Multilingual fairness improvement**
   - Reduce Gini coefficient via ensemble
   - Achieve parity across scripts/languages

### 8.2 Theoretical Contributions (if results support)

1. **Ensemble approximation bound analysis**
   - Proof or bound on α_ensemble
   - Relationship to component diversity

2. **Consistency conditions for ensemble**
   - Sufficient conditions for κ∘τ_ensemble∘p⋆ = p⋆
   - Marginal probability framework

3. **PAC learning bounds**
   - Sample complexity for ensemble vocabularies
   - VC dimension analysis

### 8.3 Practical Contributions

1. **Open-source implementation**
   - Reusable ensemble tokenization library
   - Integration with HuggingFace ecosystem

2. **Pre-trained ensemble tokenizers**
   - Release best configurations
   - Multi-domain, multilingual variants

3. **Best practices guide**
   - When to use ensemble vs. single
   - Configuration recommendations by use case

---

## Part IX: Success Criteria

### Minimum Viable Contribution
- Demonstrate ensemble BPE outperforms single BPE on at least one metric across multiple datasets
- Show when/why ensemble helps (ablation studies)
- Open-source implementation

### Strong Contribution
- Ensemble exceeds single BPE across multiple metrics (compression, Zipf R², downstream tasks)
- Evidence of improved consistency or approximation bounds
- Characterization of optimal ensemble configurations
- Multilingual fairness improvement demonstrated

### Exceptional Contribution
- Theoretical proof or bound on ensemble approximation factor
- Proof of statistical consistency under specific conditions
- Significant downstream task improvements (>5% on major benchmarks)
- New state-of-art tokenization approach

---

## Conclusion

This framework provides a comprehensive roadmap for systematic ensemble tokenization research. The modular design enables:

1. **Flexibility**: Easy to add new components, combination strategies, or metrics
2. **Reproducibility**: Clear experimental protocols and baselines
3. **Extensibility**: Foundation for follow-up research
4. **Practicality**: Implementation plan with realistic timeline

The research is grounded in 100+ papers across 20 areas, with specific attention to:
- Theoretical motivations (APX-completeness, consistency)
- Empirical best practices (Zipf alignment, compression metrics)
- Practical considerations (computational cost, inference speed)

**Key Insight**: We're ~85% in empirical territory, so systematic experimentation with strong evaluation is critical. The framework balances theoretical grounding (evaluation metrics, research questions) with pragmatic implementation (use existing libraries, start simple, expand systematically).

**Next Step**: Implement Phase 1 (core infrastructure) and validate with simple 2-component ensemble baseline before expanding to full framework.
