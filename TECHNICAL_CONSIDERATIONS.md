**Bottom Line Up Front:** COMET's recomposition objective—which discovers compositional energy functions through contrastive divergence training with Langevin dynamics—can be adapted to ARC's discrete algebraic domain by replacing continuous MCMC sampling with discrete methods (Gibbs-with-Gradients or D3PM), substituting pixel reconstruction with cross-entropy over grid tokens, and representing energy functions over program tokens/ASTs rather than raw embeddings. The most promising technical path combines **GFlowNet-EM for learning compositional program structure** with **nonparametric priors (Indian Buffet Process) for automatic component discovery**, while incorporating **object-centric representations** that preserve the topological structure the user correctly identifies as critical. Cognitive science research—particularly Hofstadter's conceptual slippage mechanism and Tenenbaum's probabilistic programs framework—provides essential design principles: concepts should be represented as generative programs, not feature vectors, and compositional discovery requires parallel probabilistic exploration with context-sensitive flexibility.

---

## COMET's mathematical objective and what must change

COMET discovers K compositional energy functions {E₁(x), E₂(x), ..., Eₖ(x)} that compose additively: **E_total(x) = Σᵢ Eᵢ(x)**. Each energy function maps images to scalars via neural networks, with lower energy indicating higher probability under p(x) ∝ exp(-E(x)). The training uses contrastive divergence:

**L = E_{x⁺~p_data}[E_θ(x⁺)] - E_{x⁻~p_θ}[E_θ(x⁻)]**

Negative samples x⁻ are generated via Langevin dynamics: **x_{t+1} = x_t - λ∇_x E(x_t) + ε** where ε ~ N(0, σ²). This iterative gradient descent on the energy landscape requires differentiable inputs—the fundamental obstacle for discrete domains. The "recomposition" insight is that by training separate energy functions to compose into coherent samples, the system discovers factors of variation without supervision. COMET uses architectural inductive biases (low latent dimensionality, positional embeddings) to encourage discovery of local versus global factors.

**Five critical assumptions require modification for ARC:**

| COMET Assumption | Discrete Requirement |
| --- | --- |
| Continuous input space ℝ^{H×W×C} | Discrete tokens {0,...,9}^{H×W} |
| Gradient-based Langevin sampling | Gibbs-with-Gradients or D3PM |
| MSE reconstruction loss | Cross-entropy over cell categories |
| Fixed K components | Nonparametric (IBP) component discovery |
| Pixel-level representation | Object-centric or program-level representation |

The compositional energy framework itself—sum of local energies defines global distribution—transfers directly to discrete domains. The challenge is entirely in the sampling and training infrastructure.

---

## Discrete energy-based models enable the adaptation

**Gibbs-with-Gradients (GWG)** provides the most direct path to discrete COMET. The key insight from Grathwohl et al. (ICML 2021) is that many discrete distributions arise from continuous functions restricted to discrete inputs. GWG computes gradient-informed proposals for Gibbs sampling:

For binary variables: **d̃(x)_i = (2x_i - 1) · ∂f/∂x_i** approximates the log-probability difference from flipping bit i. Proposals sample dimensions proportionally to |d̃(x)|, achieving near-optimal mixing in the class of locally-informed proposals. For categorical variables (like ARC's 10 colors): **d̃(x)*{ij} = ∂f/∂x*{ij} - x_i^T · ∂f/∂x_i** where the subtraction ensures valid probability updates.

**D3PM (Discrete Denoising Diffusion)** offers an alternative that avoids explicit MCMC. The forward process corrupts discrete tokens via transition matrices Q_t (uniform noise, absorbing masks, or embedding-space neighbors), while a neural network learns the reverse: p_θ(x_{t-1}|x_t). Training uses a variational bound plus auxiliary cross-entropy. D3PM's x₀-parameterization—predicting the clean data directly rather than noise—sidesteps non-differentiability entirely.

**Fisher Flow Matching** (Davis et al., NeurIPS 2024) treats categorical distributions as points on a statistical manifold with Fisher-Rao geometry. By mapping the simplex to the positive orthant of a hypersphere via φ(p) = 2√p, continuous flow matching applies. This provides closed-form geodesics and optimal forward KL minimization—potentially the most elegant mathematical framework for discrete compositional generation.

Recent work directly validates compositional energy for reasoning: **"Generalizable Reasoning through Compositional Energy Minimization"** (Oarga et al., October 2025) learns energy functions over subproblem solutions and composes them at test time via E_global = Σᵢ E_i. On N-Queens, 3-SAT, and graph coloring, this enables generalization to problems larger than those seen during training—precisely the out-of-distribution generalization ARC demands.

---

## Program synthesis provides the right representational substrate

The user's intuition that learning over DSL program tokens/ASTs preserves structural information better than grid embeddings aligns with both empirical findings and theoretical arguments. **DreamCoder's wake-sleep algorithm** demonstrates how compositional program structure emerges from a compression objective:

**Wake phase:** Find programs ρ_x that solve tasks x given current library L and neural guide Q:
ρ_x = argmax_{ρ: Q(ρ|x) large} P[x|ρ]P[ρ|L]

**Abstraction sleep:** Refactor discovered programs to expose reusable patterns and add them to library:
L* = argmax_L P[L] ∏*{x} max*{ρ: refactoring of ρ_x} P[x|ρ]P[ρ|L]

**Dream sleep:** Sample programs from L, execute them to generate synthetic tasks, and train Q to recognize these programs. This bootstrapping creates self-improving feedback.

The critical insight is that DreamCoder's compression objective—minimize description length of programs—is formally equivalent to energy minimization. Abstractions that "compress well" are precisely those that capture generalizable compositional structure. The **E-graph-based refactoring** is essential: it discovers compositional patterns not syntactically apparent in original programs by searching over semantically equivalent rewrites.

**Stitch** (Bowers et al., POPL 2023) achieves 3-4 orders of magnitude speedup over DreamCoder's abstraction algorithm through top-down corpus-guided synthesis, making library learning practical for larger corpora. **LILO** (Grand et al., 2024) adds LLM guidance and auto-documentation, enabling human-interpretable primitive names.

For ARC specifically, **Michael Hodel's ARC-DSL** provides 165 primitive operations with hand-written solvers for all 400 training tasks, proving completeness. However, search efficiency remains the bottleneck: Alford (2021) found that DreamCoder's main limitation on ARC was combinatorial explosion, not expressiveness. **GridCoder** (Ouellette, 2024) addresses this through execution-guided neural program synthesis, conditioning on intermediate grid states to prune invalid branches.

The **Latent Program Network** (Bonnet & Macfarlane, 2024) offers a differentiable alternative: encode programs in a 256-dimensional continuous latent space where gradient-based search refines solutions at test time. This achieves 46% on ARC training tasks and doubles performance on out-of-distribution tasks—suggesting that continuous optimization in a properly structured latent space can capture compositional program structure.

---

## Object-centric representations for ARC's discrete objects

**Slot Attention's iterative competitive attention** provides a differentiable mechanism for decomposing scenes into object representations without supervision. The mathematical core:

1. Initialize K slots from learned Gaussian: slots ~ N(μ, diag(σ))
2. Compute attention normalized over slots (competition): attn_{i,j} = softmax_slots(k(inputs)·q(slots)^T)
3. Weighted mean aggregation: updates = W^T · v(inputs) where W normalizes attention
4. GRU update with residual MLP: slots = GRU(slots, updates) + MLP(slots)

After T iterations, slots specialize to different objects through competition. The reconstruction objective—each slot decodes to RGB + alpha mask, combined via softmax mixture—provides unsupervised learning signal.

**Critical empirical finding for ARC:** Xu et al. (2023) showed that LLM performance on ARC **nearly doubled** when provided graph-based object representations (ARGA) instead of raw grids. ViTARC (Li et al., 2024) demonstrated that standard Vision Transformers fail catastrophically on ARC even with 1M training examples, but succeed (100% on >50% of tasks) when given **object-based positional encoding** and pixel-level tokenization. This validates the user's prior that embeddings destroy critical structural information.

**Adapting Slot Attention for ARC requires:**

- Discrete color embeddings (one-hot or learned) instead of RGB
- Connectivity-based slot initialization (flood-fill to find candidate objects)
- Cross-entropy reconstruction per cell instead of MSE
- Discrete output via argmax or Gumbel-softmax

**Grouped Discrete Representation (GDR)** (Zhao et al., 2024) shows that decomposing slot features into combinatorial attributes via channel grouping—indexed by tuples rather than scalars—preserves attribute-level similarities crucial for generalization. This bridges continuous slots and symbolic descriptions.

**EGO (Energy-based Object-centric learning)** (Zhang et al., ICLR 2023) demonstrates that Slot Attention can be reframed as energy minimization: the competitive softmax attention implements soft assignment minimizing energy over slot-input bindings, with iterative refinement as gradient descent. This provides a direct bridge between object-centric and energy-based approaches—compose object-level energies for compositional scene understanding.

---

## Cognitive science principles that must guide the design

**Hofstadter's Copycat architecture** reveals mechanisms essential for ARC-style abstract reasoning:

The **Slipnet** maintains a network of ~60 concepts with dynamic activation levels and adjustable conceptual distances. When solving "abc : abd :: xyz : ?", the system discovers that "rightmost" and "successor" (concepts activated by the first analogy) can slip to "leftmost" and "predecessor" in the new context—yielding "wyz" rather than the invalid "xya". This **conceptual slippage** is the computational mechanism for analogical transfer.

Key Copycat principles for ARC:

- **Parallel terraced scan**: Explore multiple hypotheses simultaneously at different commitment levels
- **Context-sensitive concept proximity**: The "distance" between concepts changes based on problem context
- **Emergent solutions**: No central executive; coherent answers emerge from collective agent activity
- **Temperature-regulated search**: High temperature enables exploration; low temperature (when structures cohere) enables exploitation

**Tenenbaum's probabilistic programs framework** provides the theoretical foundation: concepts are generative programs, not feature vectors. **Bayesian Program Learning** (Lake et al., Science 2015) demonstrated human-level one-shot learning of handwritten characters by representing characters as motor programs (sequences of strokes with relations). The three key ingredients—**compositionality**, **causality**, and **learning-to-learn**—map directly to ARC:

1. **Compositionality**: ARC transformations compose primitives (rotate, reflect, color-map)
2. **Causality**: Transformations are generative (input → output via program execution)
3. **Learning-to-learn**: Hierarchical priors over program structure enable few-shot transfer

**Human ARC solving data (H-ARC, LéGris et al., 2025)** from 1,729 humans across all 800 tasks shows:

- **76.2% accuracy on training, 64.2% on evaluation** with ~25 minutes per task
- 98.7% of tasks solved by at least one person
- Humans achieve **partially correct responses** systematically—errors reveal reasoning strategies
- People resort to surface-level statistics when deeper rules unclear

The **ARC-AGI-2 gap** is stark: tasks solvable by 2+ humans in ≤2 attempts yet top AI systems score <10%. This suggests AI systems lack the compositional mechanisms humans bring naturally.

**Spelke's Core Knowledge** identifies the cognitive primitives ARC was explicitly designed to require:

- **Object cognition**: Cohesion (parts move together), continuity (smooth paths), persistence
- **Number**: Approximate magnitude and small-set subitizing
- **Geometry**: Symmetry, rotation, reflection, connectivity
- **Goal-directedness**: Interpreting transformations as purposeful

These are not learned from ARC examples—they're the "start-up software" humans bring. A COMET-for-ARC system should hardcode or strongly bias toward these primitives rather than attempting to discover them from scratch.

---

## GFlowNets enable joint structure and parameter learning

**GFlowNets** learn to sample compositional objects (sequences, graphs, programs) proportionally to a reward function. Unlike RL which finds single high-reward solutions, GFlowNets maintain diversity—crucial for discovering multiple valid decompositions of ARC transformations.

**Critical capability: GFlowNets can learn reward decompositions**, not just sample from given rewards. The GFlowNet Foundations paper (Bengio et al., JMLR 2023) shows how to jointly train energy functions and GFlowNets: "the energy function is trained with samples from a GFlowNet, which, in turn, uses the energy function to form its reward." **LED-GFN** (Jang et al., ICLR 2024 oral) learns potential functions (energy decompositions) online within GFlowNet training to address sparse rewards.

**GFlowNet-EM** (Hu et al., ICML 2023) solves the core technical problem: learning latent variable models with discrete compositional latents where the E-step is intractable. GFlowNets approximate the posterior over latent configurations, enabling EM for models like non-context-free grammar induction—directly relevant to discovering ARC transformation structure.

The connection to variational inference is formal (Malkin et al., ICLR 2023): in certain cases, VI objectives equal GFlowNet objectives. But GFlowNets excel at **off-policy training without high-variance importance sampling** and **capturing multimodal posteriors**—both critical for compositional discovery where multiple valid decompositions exist.

**Direct ARC application exists:** GFN_to_ARC (GIST-DSLab) applies GFlowNets to sample program synthesis trajectories for ARC-style environments.

---

## The emergent-structured tradeoff admits a principled resolution

**The Indian Buffet Process (IBP)** provides the nonparametric framework for discovering the number of components. Unlike Dirichlet Processes (which assign objects to single clusters), IBP defines priors over **binary feature matrices**—each object possesses multiple features simultaneously. This matches ARC's compositional structure: transformations involve multiple primitive operations.

IBP's generative story: customers arrive at an Indian buffet and sample dishes (features) proportional to popularity, plus new dishes with probability α. The expected number of features grows as O(α log N), automatically adapting complexity to data.

**When structural assumptions help (the ARC regime):**

- Limited training data (~3 examples per task)
- Out-of-distribution generalization required (evaluation set deliberately differs)
- Task requires systematic compositional generalization
- Prior knowledge about domain structure exists (Core Knowledge primitives)

**When structural assumptions hurt:**

- Misalignment with true data structure
- Overly restrictive (prevents representing necessary patterns)
- Domain requires extremely flexible representations

**The deep learning vs. cognitive science divide** manifests clearly in ARC approaches:

- Pure neural (GPT-4 few-shot): ~20% accuracy, poor compositional generalization
- Program synthesis (DSL search): ~50% with ensembling, but combinatorial explosion
- Hybrid (LLM-guided DSL search + test-time training): 55-75% on recent competitions
- Human: 76% with zero task-specific training

**The minimum necessary structure for ARC** appears to be:

1. **Grid representation** (essential): ARC operates on 2D discrete grids
2. **Compositional assumption** (strongly helpful): Transformations compose
3. **Object-centric perception** (empirically critical): ~2x performance gain
4. **Number of components** (should be learned): Use IBP, not fixed K
5. **Specific decomposition** (should be learned): GFlowNet exploration

This suggests a spectrum: hardcode representation format and compositional assumption, use strong priors for perception and primitive operations, but allow the model to discover specific decompositions.

---

## A technically grounded proposal for COMET-style discovery on ARC

**Architecture: Hierarchical Compositional Energy with GFlowNet Inference**

```
Input: ARC task (input grids, output grids)
                    ↓
┌──────────────────────────────────────────────────────────┐
│  Layer 1: Object-Centric Perception                      │
│  - Discrete Slot Attention extracts objects              │
│  - Each slot: (color, position, shape descriptor)        │
│  - Connectivity prior initializes from flood-fill        │
│  - Energy: E_object(slots | grid) = -log p(grid | slots) │
└──────────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│  Layer 2: Transformation Energy Functions                │
│  - IBP prior determines number of active components      │
│  - Each E_k : (input_objects, output_objects, z_k) → ℝ  │
│  - Components correspond to primitive operations         │
│  - Composition: E_transform = Σ_k E_k (active components)│
└──────────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│  Layer 3: GFlowNet-EM for Joint Learning                │
│  - E-step: GFlowNet samples transformation decompositions│
│  - M-step: Update energy parameters via gradient descent │
│  - Uses discrete sampling (GWG) within GFlowNet rollouts │
└──────────────────────────────────────────────────────────┘
                    ↓
Output: Predicted output grid + transformation program

```

**Training objective (discrete recomposition):**

L = E_{(in,out)~data}[-log p_θ(out | in)]

where p_θ(out | in) = Σ_z p_θ(z | in) · p_θ(out | in, z) is marginalized over transformation decompositions z, approximated via GFlowNet samples.

**Specific technical choices:**

1. **Perception**: Discrete Slot Attention with one-hot color embeddings, 2D sinusoidal positional encoding, cross-entropy reconstruction. Initialize slots from connected components (flood-fill preprocessing).
2. **Energy parameterization**: Transformer encoder over (input_slots, output_slots) pairs, with separate heads for each candidate transformation primitive. Energy for primitive k: E_k = -TransformerHead_k(slots).
3. **Sampling**: Gibbs-with-Gradients within GFlowNet rollouts. The GFlowNet policy π(a|s) proposes primitive operations; GWG samples specific parameter values for each operation.
4. **Component discovery**: Truncated IBP with K_max = 20 components. The GFlowNet learns which components to activate per task. Regularize toward sparse activation (few components per transformation).
5. **Program output**: Post-hoc program extraction from the GFlowNet's sampled transformation sequence. Verify by execution; reject invalid programs.

**Baselines and improvement targets:**

| Approach | ARC-AGI-1 Accuracy | Key Limitation |
| --- | --- | --- |
| Brute-force DSL | ~40% | Combinatorial explosion |
| LLM program generation | ~42% | Poor compositional generalization |
| Test-time training (TRM) | ~45% | Computational cost, overfitting |
| SOAR (self-improving) | ~52% | Still limited by fixed primitives |
| This proposal (target) | 55-65% | Novel: learned decomposition + object-centric |

A meaningful improvement would be **>55% accuracy with fewer primitives than Hodel's 165**, demonstrating that compositional energy discovery can learn efficient abstractions rather than requiring human-designed DSLs.

---

## Key open problems and decision points

**Decision Point 1: Representation level for energy functions**

- Option A: Energy over raw grids (simplest, but destroys structure)
- Option B: Energy over object-centric slots (preserves topology, requires perception module)
- Option C: Energy over program tokens/ASTs (maximum structure preservation, hardest to train)
- **Recommendation**: Start with B, extend to C.

**Decision Point 2: Fixed vs. learned primitives**

- Option A: Use Hodel's 165-primitive DSL as fixed vocabulary
- Option B: Learn primitives via DreamCoder-style compression
- Option C: Hybrid—start with small core, expand via abstraction sleep
- OR USE A CODING LLM AND STEER PRIMITIVE DISCOVERY DIRECTION WITHIN THE WEIGHTS THROUGH SOME OBJECTIVE STEERING BASED ON PRIMITIVE USAGE.
- **Recommendation**: C. Fixed DSL limits generalization; pure learning is data-hungry.

**Decision Point 3: Sampling method for discrete energy functions**

- Option A: Gibbs-with-Gradients (connects directly to COMET's gradient-based sampling)
- Option B: D3PM (avoids explicit MCMC, modern diffusion framework)
- Option C: GFlowNet (learns to sample diverse solutions, amortized)
- **Recommendation**: GFlowNet as outer loop, GWG for local refinement within rollouts.

**Open Problem 1: Compositional generalization to novel structures**
How do we ensure that energy functions compose correctly for transformation types never seen during training? This is ARC's core challenge. Potential approaches: meta-learning over composition patterns, explicit relational structure in energy functions, test-time adaptation.

**Open Problem 2: Grounding abstract primitives**
How do primitives like "symmetry" or "continuation" get grounded in perception? Hofstadter's Slipnet had hand-designed conceptual structure. Can we learn this from ARC examples alone, or must some structure be built in (per Core Knowledge)?

**Open Problem 3: Efficient credit assignment for compositional programs**
When a composed transformation fails, which component is at fault? GFlowNet-EM addresses this partially, but credit assignment in long programs remains challenging. Potential approaches:

- execution-guided feedback (GridCoder)
- hierarchical GFlowNets
- explicit decomposition objectives.

**Open Problem 4: Bridging perception and symbolic reasoning**
The gap between continuous slot representations and discrete program symbols requires careful bridging. Vector quantization (VQ-VAE style) is one approach; explicit symbolic grounding (DeepObjectLog) is another. The right choice may be task-dependent.

---

## Prioritized reading list of essential papers

**Core Methods (read first):**

1. Du et al. (2021) — **"Unsupervised Learning of Compositional Energy Concepts" (COMET)**, NeurIPS 2021. The foundation; understand the recomposition objective thoroughly.
2. Hu et al. (2023) — **"GFlowNet-EM for Learning Compositional Latent Variable Models"**, ICML 2023. Key technique for joint structure-parameter learning.
3. Grathwohl et al. (2021) — **"Oops I Took A Gradient: Scalable Sampling for Discrete Distributions" (GWG)**, ICML 2021. Enables discrete MCMC with gradient information.

**Program Synthesis:**
4. Ellis et al. (2021) — **"DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning"**, PLDI 2021. Wake-sleep for compositional abstraction.
5. Bowers et al. (2023) — **"Top-Down Synthesis for Library Learning" (Stitch)**, POPL 2023. Scalable alternative to DreamCoder's abstraction.
6. Hodel (2023) — **"ARC-DSL"**, GitHub. Complete DSL for ARC with 165 primitives.

**Object-Centric Learning:**
7. Locatello et al. (2020) — **"Object-Centric Learning with Slot Attention"**, NeurIPS 2020. Foundation for unsupervised object discovery.
8. Zhang et al. (2023) — **"Robust and Controllable Object-Centric Learning through Energy-based Models" (EGO)**, ICLR 2023. Bridges object-centric and energy-based approaches.

**Cognitive Science:**
9. Hofstadter & FARG (1995) — **"Fluid Concepts and Creative Analogies"**, Basic Books. Conceptual slippage mechanism essential for analogy.
10. Lake et al. (2015) — **"Human-level concept learning through probabilistic program induction"**, Science. Bayesian program learning framework.
11. Lake et al. (2017) — **"Building machines that learn and think like people"**, BBS. Design principles for human-like AI.

**ARC-Specific:**
12. Chollet (2019) — **"On the Measure of Intelligence"**, arXiv. Defines ARC and fluid intelligence framing.
13. LéGris et al. (2025) — **"H-ARC: Comprehensive Behavioral Dataset"**, Scientific Data. Human solving patterns and strategies.
14. Xu et al. (2023) — **"Importance of Object-based Representations for LLMs on ARC"**, arXiv. Empirical evidence for object-centric approaches.

**Nonparametric Methods:**
15. Griffiths & Ghahramani (2011) — **"The Indian Buffet Process"**, JMLR. Nonparametric prior for discovering number of features.

---

## Conclusion

Adapting COMET to ARC requires three coordinated modifications:

1. replacing Langevin dynamics with discrete sampling methods (GWG or GFlowNets)
2. shifting the energy representation from pixel-level to object-centric or program-level substrates
3. relaxing the fixed-K assumption via nonparametric priors like the Indian Buffet Process. 

The most promising path combines **GFlowNet-EM**—which learns compositional latent structure through tractable amortized inference—with **object-centric perception** that preserves the topological structure ARC tasks demand.

Cognitive science provides non-negotiable design constraints: Hofstadter's work shows that abstract reasoning requires **context-sensitive conceptual slippage**, not rigid pattern matching; Tenenbaum's framework establishes that concepts must be **generative programs** supporting explanation and simulation, not just discriminative classifiers; and Core Knowledge research identifies the **cognitive primitives** (objects, number, space, agency) that should be built in rather than learned. The human ARC solving data confirms these insights—76% accuracy with zero training reflects the power of bringing the right compositional priors.

The key open question is not whether compositional energy methods can work for discrete domains—recent work on compositional energy minimization for constraint satisfaction demonstrates they can—but whether **discovery of the right decomposition** can be learned efficiently from ARC's limited training signal. GFlowNet-EM with IBP priors offers the most principled framework for this discovery, while object-centric representations and DSL-based program search provide the structural scaffolding to make the search tractable. The path forward requires tight integration of neural perception, energy-based composition, and symbolic program structure—a synthesis that no existing system fully achieves.