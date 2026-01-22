# MASTER TODO LIST: Compositional EBMs for ARC-AGI

## Overview
2-week sprint to implement compositional energy-based models for discrete symbolic reasoning on ARC-AGI tasks.

---

## Phase 1: Environment Setup (Day 1-2)
- [x] **1.1** Set up Python environment with PyTorch, JAX dependencies
- [ ] **1.2** Fork/clone GFN_to_ARC repository (github.com/GIST-DSLab/GFN_to_ARC)
- [ ] **1.3** Download ARC-AGI dataset (training + evaluation sets)
- [ ] **1.4** Set up experiment tracking (Weights & Biases or TensorBoard)
- [x] **1.5** Create project directory structure
- [x] **1.6** Verify ARC environment can load and visualize tasks

---

## Phase 2: Core Infrastructure (Day 2-3)
- [x] **2.1** Implement ARC grid representation utilities
  - [x] Grid loading/parsing
  - [x] Grid visualization (matplotlib)
  - [x] Grid transformations (numpy operations)
- [x] **2.2** Implement transformation primitive library
  - [x] Rotation (90°, 180°, 270°)
  - [x] Reflection (horizontal, vertical, diagonal)
  - [x] Color mapping/remapping
  - [x] Translation
  - [x] Crop/tile operations
- [ ] **2.3** Create task curation module
  - [x] Filter tasks by transformation type
  - [ ] Select 10-20 curated single-step tasks
  - [ ] Select 5-10 curated multi-step tasks
- [x] **2.4** Implement evaluation metrics
  - [x] Exact match accuracy
  - [x] Per-cell accuracy
  - [x] Transformation identification accuracy

---

## Phase 3: Subproblem 1 - Compositional Energy Functions (Day 3-7)

### 3A: Single-Transformation Energy Functions
- [x] **3A.1** Design energy function interface: `E(output | transformation, input) -> scalar`
- [x] **3A.2** Implement rotation energy function
  - [x] `E_rot(out | in) = ||rotate(in) - out||² + regularization`
- [x] **3A.3** Implement reflection energy function
  - [x] `E_ref(out | in) = ||reflect(in) - out||²`
- [x] **3A.4** Implement color mapping energy function
  - [x] `E_color(out | in) = cross_entropy(color_map(in), out)`
- [x] **3A.5** Implement translation energy function
- [ ] **3A.6** Implement crop/tile energy function
- [ ] **3A.7** Test single-transformation discrimination on 10 curated tasks
  - [x] Verify low energy for correct transformations (demo script)
  - [x] Verify high energy for incorrect transformations (demo script)

### 3B: Energy Composition
- [x] **3B.1** Implement additive energy composition: `E_total = Σ E_i`
- [x] **3B.2** Implement product-of-experts composition: `p(out) ∝ exp(-Σ E_i)`
- [ ] **3B.3** Implement Gibbs-with-Gradients (GWG) sampler for discrete grids
  - [ ] Gradient-informed proposal distribution
  - [ ] Categorical variable handling (10 ARC colors)
- [ ] **3B.4** Test composition on 5 two-step transformation tasks
- [ ] **3B.5** Measure composition accuracy (target: >70%)

### 3C: Day 5 Checkpoint Evaluation
- [ ] **3C.1** Run systematic evaluation on single-transformation tasks
- [ ] **3C.2** Document discrimination accuracy per transformation type
- [ ] **3C.3** Decision: Continue Subproblem 1 or pivot to Subproblem 2

---

## Phase 4: Subproblem 2 - GFlowNet Integration (Day 6-10, if needed)

### 4A: GFlowNet Setup
- [ ] **4A.1** Integrate with GFN_to_ARC codebase
- [ ] **4A.2** Reproduce baseline GFlowNet results on ARC tasks
- [ ] **4A.3** Define compositional reward: `R(s) = exp(-E_compositional(s))`

### 4B: Factored Energy Rewards
- [ ] **4B.1** Implement factored energy components
  - [ ] `E_shape`: Shape similarity energy
  - [ ] `E_color`: Color distribution energy
  - [ ] `E_structure`: Structural/topological energy
- [ ] **4B.2** Replace baseline reward with: `R = exp(-(E_shape + E_color + E_structure))`
- [ ] **4B.3** Train GFlowNet with compositional reward
- [ ] **4B.4** Compare sample diversity vs baseline
- [ ] **4B.5** Compare accuracy vs baseline on 50+ tasks

---

## Phase 5: Object-Centric Representations (Optional Enhancement)
- [ ] **5.1** Implement flood-fill object detection
- [ ] **5.2** Implement object-slot representation
  - [ ] Object: (color, position, shape_descriptor)
- [ ] **5.3** Define energy over object representations
- [ ] **5.4** Test object-centric vs grid-level energy

---

## Phase 6: Experiments & Ablations (Day 8-10)
- [ ] **6.1** Main experiment: Accuracy on 50+ ARC tasks
- [ ] **6.2** Ablation: Individual energy components
- [ ] **6.3** Ablation: Composition methods (additive vs product-of-experts)
- [ ] **6.4** Ablation: With/without GWG sampling
- [ ] **6.5** Zero-shot composition test (if Subproblem 1 succeeds)
  - [ ] Train on single transformations
  - [ ] Test on unseen compositions
- [ ] **6.6** Generate figures and visualizations
  - [ ] Energy landscape visualizations
  - [ ] Sample trajectories
  - [ ] Accuracy plots

---

## Phase 7: Writing & Documentation (Day 11-14)
- [ ] **7.1** Write paper outline (4-6 pages workshop format)
- [ ] **7.2** Write introduction and related work
- [ ] **7.3** Write methods section
- [ ] **7.4** Write experiments section
- [ ] **7.5** Write discussion and conclusion
- [ ] **7.6** Generate all figures for paper
- [ ] **7.7** Internal review and revisions
- [ ] **7.8** Submit to arXiv
- [ ] **7.9** Submit to ICLR 2026 Workshop (deadline ~Jan 30)

---

## Decision Points (Critical Checkpoints)

### Day 5 Checkpoint
**Question**: Do single-transformation energies discriminate correctly?
- **YES** → Continue Subproblem 1 through Day 10
- **NO** → Pivot to Subproblem 2 (GFlowNet fork)

### Day 8 Checkpoint (if pivoted)
**Question**: Can you reproduce baseline + improve with factored reward?
- **YES** → Complete experiments, write workshop paper
- **NO** → Execute Subproblem 3 (analysis paper)

### Day 10 Checkpoint
**Question**: Do compositions work?
- **YES** → Write up results, target ICLR workshops
- **NO** → Add Subproblem 2 as augmentation, write combined paper

---

## Success Criteria
- [ ] **Minimum Viable**: Single-transformation energy discrimination on 10-20 tasks
- [ ] **Target**: Composition accuracy >70% on 2-step transformations
- [ ] **Stretch**: Zero-shot composition generalization
- [ ] **Output**: ArXiv preprint + workshop submission by Day 14

---

## File Structure (Proposed)
```
EPR/
├── src/
│   ├── data/
│   │   ├── arc_loader.py          # ARC dataset loading
│   │   ├── task_curator.py        # Task selection/filtering
│   │   └── visualization.py       # Grid visualization
│   ├── energy/
│   │   ├── base.py                # Energy function interface
│   │   ├── transformations.py     # Transformation primitives
│   │   ├── rotation.py            # Rotation energy
│   │   ├── reflection.py          # Reflection energy
│   │   ├── color_map.py           # Color mapping energy
│   │   ├── translation.py         # Translation energy
│   │   ├── composition.py         # Energy composition
│   │   └── sampling.py            # GWG sampler
│   ├── gflownet/
│   │   ├── model.py               # GFlowNet architecture
│   │   ├── reward.py              # Compositional rewards
│   │   └── training.py            # Training loop
│   ├── objects/
│   │   ├── detection.py           # Object detection
│   │   └── representation.py      # Slot representations
│   └── evaluation/
│       ├── metrics.py             # Evaluation metrics
│       └── experiments.py         # Experiment runners
├── experiments/
│   ├── exp001_single_energy/      # Single transformation experiments
│   ├── exp002_composition/        # Composition experiments
│   └── exp003_gflownet/           # GFlowNet experiments
├── notebooks/
│   ├── exploration.ipynb          # Data exploration
│   └── visualization.ipynb        # Results visualization
├── configs/
│   └── default.yaml               # Configuration
├── tests/
│   └── test_energy.py             # Unit tests
└── paper/
    └── main.tex                   # Workshop paper
```
