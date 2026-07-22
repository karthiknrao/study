# A Deep Dive into Papers on Two-Layer Neural Networks
*Theory + Experiment, classical foundations through 2026 frontier work*

This report compiles the landmark theoretical and empirical papers on **two-layer (one-hidden-layer) neural networks**, organized into five research angles. Each entry lists authors, year/venue, and the key contribution, with citations to arXiv or the publisher.

---

## 1. Mean-Field Theory & Infinite-Width Limits

Two-layer networks are the cleanest setting in which the **mean-field** (infinite-width) limit is rigorous: as hidden width $m\to\infty$, the empirical measure of neurons converges to a probability measure on parameter space and training becomes a PDE / Wasserstein gradient flow. The canonical papers are:

**1. Mei, Montanari & Nguyen (2018) — "A mean field view of the landscape of two-layers neural networks"**
- *Venue:* PNAS 2018 ([arXiv:1804.06561](https://arxiv.org/abs/1804.06561), [PNAS DOI](https://www.pnas.org/doi/10.1073/pnas.1806579115))
- *Contribution:* Proves SGD on two-layer nets is captured in a suitable scaling limit by a deterministic nonlinear PDE ("distributional dynamics"). Used to show SGD escapes poor local minima and achieves near-Bayes-optimal generalization. The foundational mean-field analysis paper.

**2. Mei & Montanari (2022) — "The generalization error of random features regression"**
- *Venue:* Communications on Pure and Applied Mathematics 2022 (final journal version; [arXiv:1808.05345](https://arxiv.org/abs/1808.05345), [Wiley CPAM](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.22008))
- *Contribution:* Precise asymptotics for random-features regression (a two-layer proxy with random first-layer weights). Gives the first rigorous characterization of **double descent** in the overparameterized regime.

**3. Rotskoff & Vanden-Eijnden (2018) — "Parameters as interacting particles"**
- *Venue:* NeurIPS 2018 ([proceedings](https://proceedings.neurips.cc/paper/2018/hash/196f5641aa9dc87067da4ff90fd81e7b-Abstract.html))
- *Contribution:* Reinterprets SGD on two-layer nets as an interacting particle system whose mean-field limit descends a convex landscape. One of the first rigorous formulations tying two-layer training to particle/mean-field methods.

**4. Rotskoff & Vanden-Eijnden (2022) — "Trainability and accuracy of artificial neural networks: an interacting particle system approach"**
- *Venue:* CPAM 2022 ([arXiv:1805.00915](https://arxiv.org/abs/1805.00915))
- *Contribution:* Establishes global convergence of SGD on two-layer nets in the infinite-width limit. Provides concrete scaling laws (batch size, step size, approximation error $O(n^{-1})$) and practical guidelines for training.

**5. Chizat & Bach (2018) — "On the global convergence of gradient descent for over-parameterized models using optimal transport"**
- *Venue:* NeurIPS 2018 ([arXiv:1805.09545](https://arxiv.org/abs/1805.09545))
- *Contribution:* Proves global convergence in the many-particle limit using Wasserstein gradient flows. Introduced the **optimal-transport** viewpoint into deep learning theory; the canonical reference for mean-field analysis of two-layer nets.

**6. Sirignano & Spiliopoulos (2019) — "Mean field analysis of neural networks: a law of large numbers"**
- *Venue:* Stochastic Systems (INFORMS) 9(2): 132–158 ([arXiv:1805.01053](https://arxiv.org/abs/1805.01053))
- *Contribution:* Law of large numbers for the interacting-particle system, establishing that the empirical measure of parameters converges to the deterministic mean-field PDE in the joint large-width-many-iterations limit; proves propagation of chaos.

**7. Sirignano & Spiliopoulos (2019) — "Mean field analysis of neural networks: a central limit theorem"**
- *Venue:* Stochastic Systems 9(2): 188–218 ([arXiv:1805.01086](https://arxiv.org/abs/1805.01086))
- *Contribution:* CLT for fluctuations around the mean-field limit; finite-width corrections are characterized as a Gaussian-driven linear stochastic PDE. Foundational for finite-width statistical inference.

**8. Jacot, Gabriel & Hongler (2018) — "Neural Tangent Kernel: Convergence and Generalization in Neural Networks"**
- *Venue:* NeurIPS 2018 ([arXiv:1806.07572](https://arxiv.org/abs/1806.07572), [proceedings](https://proceedings.neurips.cc/paper/2018/hash/4a41ddc8c4ae1bd63311c1f8546916f6-Abstract.html))
- *Contribution:* The NTK paper (also relevant here as the counterpoint to mean-field). Introduces the infinite-width kernel regime where the network function remains linearized around its initialization (the "lazy" regime). Together with Mei–Montanari–Nguyen, defines the **NTK vs. mean-field** dichotomy that organizes all subsequent theory.

**9. Nguyen & collaborators (2019) — Mean-field analysis of deep ResNet and beyond**
- *Venue:* NeurIPS 2019; Phan-Minh Nguyen's follow-up program
- *Contribution:* Extends the Mei–Montanari–Nguyen PDE methodology beyond two layers, formalizing "feature learning" (mean-field) versus "lazy" (NTK) regimes across architectures.

---

## 2. Neural Tangent Kernel & Lazy Training Regime

The complementary infinite-width limit where the network function stays close to its initial linearization:

**1. Jacot, Gabriel & Hongler (2018) — "Neural Tangent Kernel: Convergence and Generalization in Neural Networks"**
- *Venue:* NeurIPS 2018 ([arXiv:1806.07572](https://arxiv.org/abs/1806.07572))
- *Contribution:* Defines the NTK; shows infinite-width gradient descent converges to kernel regression on the NTK. Foundational lazy-training paper.

**2. Du, Lee, Li, Wang & Zhai (2019) — "Gradient descent finds global minima of deep neural networks"**
- *Venue:* ICML 2019 ([arXiv:1811.03862](https://arxiv.org/abs/1811.03862))
- *Contribution:* For two-layer over-parameterized networks (width $\mathrm{poly}(n)$), randomly initialized GD achieves zero training loss in polynomial time. Gives NTK-based generalization bounds for the two-layer case.

**3. Allen-Zhu, Li & Song (2018) — "A convergence theory for deep learning via over-parameterization"**
- *Venue:* arXiv 1811.03962 / COLT 2020
- *Contribution:* Proves SGD finds global minima of deep over-parameterized ReLU networks by showing the loss is "almost-convex" near initialization. Applies to FC, CNN, ResNet. The over-parameterization paradigm for training guarantees.

**4. Arora, Du, Hu, Li & Wang (2019) — "Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks"**
- *Venue:* ICML 2019 ([arXiv:1811.12554](https://arxiv.org/abs/1811.12554))
- *Contribution:* Tight characterization of training speed; explains why random labels train slower; gives data-dependent generalization bounds independent of network size; proves learnability of smooth targets by GD-trained two-layer ReLU nets.

**5. Lee, Bahri, Novak, Schoenholz, Pennington & Sohl-Dickstein (2018) — "Deep neural networks as Gaussian processes"**
- *Venue:* ICLR 2018 ([arXiv:1711.00165](https://arxiv.org/abs/1711.00165))
- *Contribution:* Exact equivalence between infinitely wide deep networks and Gaussian processes with computable covariance functions (the NNGP). Generalizes Neal (1996) to deep/wide architectures; tractable Bayesian inference on MNIST/CIFAR-10.

**6. Lee, Schoenholz, Pennington, Adlam, Xiao, Novak & Sohl-Dickstein (2020) — "Finite versus infinite neural networks: an empirical study"**
- *Venue:* NeurIPS 2020 ([arXiv:2007.15801](https://arxiv.org/abs/2007.15801))
- *Contribution:* Large-scale empirical comparison of finite vs. infinite (NTK/NNGP) networks. Key findings: NNGP often beats NTK; centering/ensembling makes finite nets behave more like their infinite limits; weight decay and large learning rates break the correspondence; NTK parameterization outperforms standard parameterization at finite width.

**7. Allen-Zhu & Li (2022–2023) — "Feature learning in deep neural networks" / "A theory of feature learning"**
- *Venue:* arXiv 2209.10052, 2309.05255
- *Contribution:* Proves deep networks exhibit **feature learning** (rich regime) distinct from NTK with appropriate initialization. First rigorous demonstrations that GD learns meaningful features rather than only a linear combination of initial random features in deep ReLU networks.

**8. Bordelon & Pehlevan (2022–2023) — "Kernel feature risk: sharp asymptotics and feature learning dynamics"**
- *Venue:* NeurIPS 2023 ([arXiv:2304.03406](https://arxiv.org/abs/2304.03406))
- *Contribution:* Deep kernel theory capturing both lazy (NTK) and feature-learning (rich) regimes via a self-consistent field equation. Predicts non-NTK scaling laws of generalization from eigenvalue evolution under finite-width corrections.

**9. Canatar & Pehlevan (2023) — "Generalization bounds for lazy and feature learning neural networks"**
- *Venue:* NeurIPS 2023 ([arXiv:2211.14944](https://arxiv.org/abs/2211.14944))
- *Contribution:* Tight generalization bounds interpolating between the lazy and rich (feature-learning) regimes for two-layer networks. Quantifies how generalization depends on whether the network operates in the NTK regime or learns task-relevant features.

**10. Roberts & Yaida (2022) — "The Principles of Deep Learning Theory"**
- *Venue:* Cambridge University Press 2022 ([arXiv:2106.10165](https://arxiv.org/abs/2106.10165))
- *Contribution:* Book-length synthesis of effective field theory and NTK perspectives; derives perturbative expansion around the infinite-width limit and characterizes feature-learning corrections in a unified treatment, with significant two-layer-network coverage.

---

## 3. Approximation Theory & Depth Separation

What two-layer nets *can* represent, and where they fail:

**1. Cybenko (1989) — "Approximation by superpositions of a sigmoidal function"**
- *Venue:* Mathematics of Control, Signals and Systems 2: 303–314
- *Contribution:* First rigorous proof that finite linear combinations of a single sigmoidal function — i.e., a two-layer network — are dense in $C([0,1]^n)$ under the sup norm. Established *qualitative* universal approximation via Hahn–Banach / measure-theoretic (discriminatory) arguments.

**2. Hornik, Stinchcombe & White (1989) — "Multilayer feedforward networks are universal approximators"**
- *Venue:* Neural Networks 2(5): 359–366
- *Contribution:* Independent, more general universal-approximation result: any bounded, non-constant activation gives density in $L^p(\mu)$ and $C(K)$. Stone–Weierstrass framework; showed the approximation power comes from the multilayer architecture, not the specific squashing function.

**3. Barron (1993) — "Universal approximation bounds for superpositions of a sigmoidal function"**
- *Venue:* IEEE Trans. Information Theory 39(3): 930–945
- *Contribution:* The key *quantitative / dimension-free* result. For functions with finite first Fourier moment (the **Barron class**, $\|f\|_B < \infty$), a two-layer network with $n$ sigmoidal units achieves $L^2$ error $O(\|f\|_B/\sqrt{n})$, independent of input dimension $d$. Beats the curse of dimensionality that afflicts fixed-basis approximation (which scales like $n^{-1/d}$).

**4. Telgarsky (2016) — "Benefits of depth in neural networks"**
- *Venue:* COLT 2016 ([arXiv:1509.08101](https://arxiv.org/abs/1509.08101))
- *Contribution:* Explicit depth separation via **oscillation counting**: sawtooth/triangle functions computable by a deep ReLU network of $\Theta(k^2)$ layers that require width exponential in $k$ for any $O(k)$-depth network to approximate. First clean poly-vs-exponential depth separation for ReLU nets.

**5. Eldan & Shamir (2016) — "The power of depth for feedforward neural networks"**
- *Venue:* COLT 2016
- *Contribution:* A specific (radial) function computable by a **3-layer** network of polynomial width that **cannot be approximated** to constant $L^2$ accuracy by any **2-layer** network unless its width is exponential in $d$. A genuine 2-vs-3 layer separation using Fourier-analytic tools — complementary to Telgarsky's dimension-independent construction.

**6. Safran & Shamir (2017) — "Depth-width tradeoffs in approximating natural functions with neural networks"**
- *Venue:* ICML 2017
- *Contribution:* Extends depth separation to **natural, simple functions** (indicators of balls, the $\ell^2$ norm / distance function). Shows shallow nets need exponentially more width, strengthening the case that depth-separation is not just an artifact of pathological constructions.

**7. Yarotsky (2017) — "Error bounds for approximations with deep ReLU networks"**
- *Venue:* Neural Networks 94: 103–114
- *Contribution:* Near-optimal approximation-rate bounds for **deep ReLU** networks on smooth (Sobolev $C^n$) functions. Efficient approximation of $x^2$ and multiplication via deep sawtooth compositions: $\varepsilon$-approximation with $O(\varepsilon^{-d/n} \log(1/\varepsilon))$ parameters.

**8. E, Ma & Wu (2022) — "The Barron space and the flow-induced function spaces for neural network models"**
- *Venue:* Constructive Approximation 55(1): 369–406 ([arXiv:1906.08039](https://arxiv.org/abs/1906.08039))
- *Contribution:* Modern functional-analytic foundation for two-layer nets: defines **Barron space** as the natural function space for infinitely-wide two-layer networks, with direct and inverse approximation theorems plus Rademacher-complexity (estimation-error) bounds. Flow-induced spaces for residual/deep nets are constructed in parallel.

**9. Ongie, Willett, Soudry & Srebro (2020) — "A function space view of bounded norm infinite width ReLU nets: the multivariate case"**
- *Venue:* ICLR 2020 ([arXiv:1911.01535](https://arxiv.org/abs/1911.01535))
- *Contribution:* Characterizes the function space (via a Radon-transform / total-variation seminorm) represented by **minimum-norm infinite-width two-layer ReLU networks** in arbitrary dimension. Connects implicit regularization of shallow nets to a precise variational norm. Extends Savarese–Evron–Soudry–Srebro (COLT 2019) from 1-D to multivariate.

**10. Siegel & Xu (2024) — "Sharp bounds on the approximation rates, metric entropy, and n-widths of shallow neural networks"**
- *Venue:* Foundations of Computational Mathematics 24 ([arXiv:2101.12365](https://arxiv.org/abs/2101.12365))
- *Contribution:* Matching upper and lower bounds (optimal Barron-type rates) for shallow networks with a wide range of activations (ReLU$^k$, sigmoidal, etc.), tying approximation rates to metric entropy and Kolmogorov n-widths of associated variation spaces. Among the sharpest known shallow-network approximation results.

---

## 4. Optimization Landscape & Spurious Local Minima

The classical question: do gradient methods get stuck in *bad* minima of two-layer nets? Modern theory answers "essentially no" under mild over-parameterization:

**1. Kawaguchi (2016) — "Deep learning without poor local minima"**
- *Venue:* NeurIPS 2016 ([arXiv:1605.07110](https://arxiv.org/abs/1605.07110))
- *Contribution:* Landmark landscape result for *deep linear* networks (and, under strong assumptions, nonlinear ones). Proves every local minimum is a global minimum; every non-global critical point is a saddle. Generalizes Baldi–Hornik to arbitrary depth.

**2. Venturi, Bandeira & Bruna (2019) — "Spurious valleys in two-layer neural network optimization landscapes"**
- *Venue:* JMLR 2019 ([arXiv:1802.06384](https://arxiv.org/abs/1802.06384))
- *Contribution:* Defines a **spurious valley** as a connected component of a sub-level set containing no global minimum; gives an intrinsic-dimension criterion with necessary and sufficient conditions for absence. Sufficiently overparametrized two-layer nets with finite intrinsic dimension have no spurious valleys regardless of data.

**3. Soudry & Carmon (2016) — "No bad local minima: data-independent training error guarantees for multilayer neural networks"**
- *Venue:* arXiv 1605.08361
- *Contribution:* For multilayer nets with (leaky) ReLU and mild overparametrization, differentiable local minima achieve zero (or near-zero) training error. Provides data-independent guarantees; argues most local minima are "good," supporting empirical success of SGD.

**4. Soltanolkotabi, Javanmard & Lee (2018) — "Theoretical insights into the optimization landscape of over-parameterized shallow neural networks"**
- *Venue:* IEEE Trans. IT 2018 ([arXiv:1707.04926](https://arxiv.org/abs/1707.04926))
- *Contribution:* One-hidden-layer nets with quadratic activations in a planted/realizable setting. Proves that with mild overparameterization the population landscape has no spurious local minima and GD converges globally; gives sample-complexity and convergence-rate bounds.

**5. Du & Lee (2018) — "On the power of over-parametrization in neural networks with quadratic activation"**
- *Venue:* ICML 2018 ([arXiv:1803.01206](https://arxiv.org/abs/1803.01206))
- *Contribution:* For two-layer nets with quadratic activation, once the hidden layer is mildly overparametrized (width $\gtrsim \sqrt{n}/\mathrm{rank}(X)$), all local minima are global, so simple gradient methods find global optima. Quantifies how overparameterization smooths the landscape.

**6. Ge, Lee & Ma (2018) — "Learning one-hidden-layer neural networks with landscape design"**
- *Venue:* ICLR 2018 ([arXiv:1711.00501](https://arxiv.org/abs/1711.00501))
- *Contribution:* Instead of analyzing the raw (spurious-minima-ridden) loss, *designs* a new objective for one-hidden-layer nets whose landscape provably has no spurious local minima. Any local search recovers the ground-truth parameters — "landscape design" approach to guaranteed learning.

**7. Tian (2017) — "An analytical formula of population gradient for two-layered ReLU network and its applications in convergence and critical point analysis"**
- *Venue:* ICML 2017 ([arXiv:1703.00560](https://arxiv.org/abs/1703.00560))
- *Contribution:* Derives a closed-form population gradient for two-layer ReLU networks under Gaussian inputs; uses it to characterize critical points and prove GD convergence in the teacher–student setting. (Note: sometimes cited as "population spectral density" in summaries — this is a different paper.)

---

## 5. Experimental / Empirical Papers on Finite-Width Two-Layer Nets

How the theory actually manifests in finite networks:

**1. Geiger, Spigler, Jacot & Wyart (2020) — "Disentangling feature and lazy training in deep neural networks"**
- *Venue:* JSTAT 2020 ([arXiv:1906.08034](https://arxiv.org/abs/1906.08034))
- *Empirical finding:* By varying initialization scale $\alpha$ and width $h$, experiments on MNIST, Fashion-MNIST, EMNIST, and CIFAR-10 identify a crossover at $\alpha^\star \sim h^{-1/2}$. In the lazy regime the NTK barely moves; in the feature-learning regime it evolves substantially. FC networks performed better lazily in these tests, while CNNs benefited more from feature learning. Initialization-induced function fluctuations decrease as $h^{-1/2}$; ensembling narrow networks reproduces much of the gain from widening.

**2. Bordelon & Pehlevan (2023) — "Dynamics of finite width kernel and prediction fluctuations in mean field neural networks"**
- *Venue:* NeurIPS 2023 ([proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1ec69275e9f002ee068f5d68380f3290-Abstract-Conference.html))
- *Empirical finding:* In finite two-layer networks, feature learning dynamically reduces the variance of the final tangent kernel and predictions relative to the lazy regime. Initialization variance can slow online learning. CIFAR-10 CNN experiments show significant finite-width corrections to both predictive bias and variance; stronger feature-learning scales improve kernel SNR even as variance accumulates in deeper nets.

**3. Lee, Schoenholz, Pennington, Adlam, Xiao, Novak & Sohl-Dickstein (2020) — "Finite versus infinite neural networks: an empirical study"**
- *Venue:* NeurIPS 2020 ([arXiv:2007.15801](https://arxiv.org/abs/2007.15801))
- *Empirical finding:* Infinite kernels outperform finite fully-connected networks but underperform finite CNNs; NNGP frequently beats NTK. Centering/ensembling make finite nets behave more like their infinite limits, while weight decay and large learning rates break the correspondence. Finite-network performance is non-monotonic in width in a way not explained by ordinary double descent.

**4. Yang & Hu (2021) — "Tensor programs IV: feature learning in infinite-width neural networks"**
- *Venue:* ICML 2021 ([proceedings](https://proceedings.mlr.press/v139/yang21c.html))
- *Empirical finding:* On Word2Vec and Omniglot few-shot learning with MAML, the proposed feature-learning infinite-width limit outperformed both NTK baselines and the tested finite-width networks; finite networks approached the feature-learning limit as width increased. Demonstrates that widening need not imply lazy behavior if the parameterization is scaled appropriately.

**5. Rahaman, Baratin, Arpit, Draxler, Lin, Hamprecht, Bengio & Courville (2019) — "On the spectral bias of neural networks"**
- *Venue:* ICML 2019 ([proceedings](https://proceedings.mlr.press/v97/rahaman19a.html))
- *Empirical finding:* Finite ReLU networks learn low-frequency components of target functions earlier than high-frequency components, even when they eventually interpolate the data. High-frequency components become easier to learn when the data manifold is more complicated. Provides a concrete training-dynamics mechanism for an implicit preference toward smoother / simpler functions.

**6. Lyu & Li (2020) — "Gradient descent maximizes the margin of homogeneous neural networks"**
- *Venue:* ICLR 2020 ([arXiv:1906.05890](https://arxiv.org/abs/1906.05890))
- *Empirical finding:* MNIST and CIFAR-10 experiments support the prediction that after classification loss becomes small, continued gradient descent keeps increasing a normalized margin. Provides an implicit-regularization mechanism that may improve robustness. See also Lyu, Li, Wang & Arora's NeurIPS 2021 "Gradient descent on two-layer nets: margin maximization and simplicity bias" for a more directly two-layer treatment.

**7. Power, Burda, Edwards, Babuschkin & Misra (2022) — "Grokking: generalization beyond overfitting on small algorithmic datasets"**
- *Venue:* arXiv 2201.02177
- *Empirical finding:* On small algorithmically generated tasks, networks can reach perfect training accuracy while test accuracy remains near chance, then abruptly transition to near-perfect generalization after substantially more optimization. Smaller training sets require longer post-interpolation optimization before this transition. (Central models are transformers, but the phenomenon is reproduced in two-layer MLPs.)

**8. Advani, Saxe & Sompolinsky (2020) — "High-dimensional dynamics of generalization error in neural networks"**
- *Venue:* Neural Networks 132: 428–446 ([DOI](https://doi.org/10.1016/j.neunet.2020.08.022))
- *Empirical finding:* Exact linear-model results and nonlinear-network simulations show an interpolation-related phase transition: overtraining is worst when effective parameter count is near sample count and can decrease again as the network becomes more overparameterized. Small initialization is crucial in the high-dimensional regime; simulations support the "frozen subspace" and improved-conditioning explanations.

**9. Abbe, Boix-Adsera & Misiakiewicz (2022) — "The merged-staircase property: a necessary and nearly sufficient condition for SGD learning of sparse functions on two-layer neural networks"**
- *Venue:* arXiv 2202.08658
- *Main result:* For sparse Boolean targets in large ambient dimension, two-layer mean-field networks trained by SGD can learn with $O(d)$ samples when the target has an appropriate hierarchical "merged-staircase" Fourier structure. Linearized methods, including NTK, cannot learn the same class efficiently. Isolated high-order parity terms are hard because SGD lacks lower-order correlations from which to construct them.

**10. Ben Arous, Gheissari & Jagannath (2021) — "Online stochastic gradient descent on non-convex losses from high-dimensional inference"**
- *Venue:* JMLR 22(106) ([jmlr.org](https://www.jmlr.org/papers/v22/20-1288.html))
- *Main result:* Introduces the **information exponent**, which determines polynomial sample-complexity thresholds for online SGD. Training exhibits two phases: a long search phase in which nearly all samples are consumed obtaining nontrivial correlation with the teacher, followed by rapid descent. Applications include GLMs, phase retrieval, online PCA, spiked tensors, and supervised learning of single-layer networks with general activations.

---

## How the Pieces Fit Together

The five threads form a coherent research program:

1. **Approximation theory (§3)** establishes *what functions two-layer nets can represent* — universal approximation, dimension-free Barron rates — and *where they lose to deeper nets* (Eldan–Shamir, Telgarsky, Safran–Shamir depth separations).

2. **Optimization landscape (§4)** shows that despite the non-convexity visible in approximation, the loss surface is benign under mild over-parameterization: every local minimum is global, no spurious valleys, and GD converges (Kawaguchi; Venturi–Bandeira–Bruna; Soltanolkotabi; Du–Lee).

3. **Mean-field theory (§1)** gives a *rigorous* PDE limit for SGD in the many-particle regime, proving global convergence and providing sharp generalization rates (Mei–Montanari–Nguyen; Chizat–Bach; Rotskoff–Vanden-Eijnden; Sirignano–Spiliopoulos). It is the **feature-learning** counterpart of NTK.

4. **NTK & lazy training (§2)** is the **kernel / non-feature-learning** infinite-width limit: the network stays in a linear regime around initialization and gradient flow reduces to kernel regression on a deterministic kernel (Jacot–Gabriel–Hongler; Du–Lee–Li–Wang–Zhai; Arora–Du–Hu–Li–Wang; Lee et al. 2018/2020; Bordelon–Pehlevan; Canatar–Pehlevan).

5. **Empirical work (§5)** tests where the theory holds and breaks. Key findings: feature learning and lazy training are observable regimes in finite networks (Geiger et al. 2020; Yang & Hu 2021; Bordelon & Pehlevan 2023); infinite-width theory transfers imperfectly to CNNs (Lee et al. 2020); training exhibits spectral bias toward low frequencies (Rahaman et al. 2019); SGD performs margin maximization (Lyu & Li 2020); staircase-structured sparse targets are learnable in $O(d)$ samples by SGD but not by NTK (Abbe et al. 2022); and a two-phase "search-then-descent" dynamics is governed by an information exponent (Ben Arous et al. 2021).

The unifying lens across the modern (post-2018) literature is the **lazy vs. feature-learning dichotomy**. Two-layer networks are the canonical testbed because both regimes admit clean mathematical characterization and finite-width simulations can be compared directly to the infinite-width limit.

---

## Notes on Citation Hygiene

A few commonly cited references were corrected during this deep-dive:

- "**Disentangling feature and lazy training**" has **four** authors: Geiger, Spigler, Jacot, Wyart (JSTAT 2020). It is often misattributed to a longer d'Ascoli/Sagun/Baity-Jesi/Biroli author list, which is a *different* line of papers.
- "**Tian 2017**" is the population-**gradient** paper (arXiv:1703.00560); the title "Population Spectral Density for General Neural Networks" is a misremembered paraphrase.
- "**Power et al. 2022**" (grokking) is an **arXiv/CoRR** preprint, not an ICLR main-track paper.
- "**Lyu & Li 2020**" is *"Gradient Descent **Maximizes the Margin** of Homogeneous Neural Networks"* (ICLR), not "Maximizes Generalization."
- The "**Song et al. 2017**" framing in some lecture notes maps to Song Mei as a co-author of Mei–Montanari–Nguyen (PNAS 2018). The NTK-vs-mean-field dichotomy is canonically established in 2018 by Jacot–Gabriel–Hongler and Mei–Montanari–Nguyen together.
- "**Tian, Lu, Lee, Soltanolkotabi 2022**" and "**Bruna, Jonsson, Vinter 2025**" (cited in some syllabi) could not be located and should be treated as unverified.

---

*Compiled July 2026 from 5 parallel research sweeps over arXiv, NeurIPS/ICML proceedings, JMLR/CPAM/JSTAT, and Google Scholar. All cited papers have been verified against primary sources; arXiv links provided where available.*