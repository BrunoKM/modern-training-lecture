#import "@preview/touying:0.6.1": *

= Scaling Laws \ #text(weight: "thin", size: 0.8em)[_c.f._ Scale Is All You Need]

==
// Describe the empirical phenomenon (Kaplan et al.), Chinchilla
(Kaplan 2020)#cite(<kaplan2020scaling>) observe that training loss decreases predictably with:
- Model size $N$ (number of parameters) as $L(N) prop N^(-alpha_N)$
- Dataset size $D$ (number of tokens) as $L(D) prop D^(-alpha_D)$


// The “foundation model” training recipe: data is abundant, compute is scarce. Increase compute to get better performance

// Don't need to worry about generalisation error. Generalisation error doesn't exist in this setting: Training loss is an unbiased estimate of the validation loss
// TODO: Runa has a figure for this

// Scaling laws as a _practical_ tool for training at scale. Allow for
// 1. Comparing Training Setups (Which algorithm is better, A or B? Well, this might depend on the scale. We need to see the scaling law to see the whole picture.)
// 2. Projecting – how much compute/data do I need to reach a certain level of performance? If I spend this much on training, what performance can I expect to get?
// 3. Compute-efficient training: where should I allocate compute? Should I train on more data? Or train a larger model? We can use the scaling laws to decide.
// 3.1 (maybe derive the Chinchilla scaling rule from scratch).


// Maybe add: active area of research
// - What happens if you have repeating data (e.g. high quality data)?
// - Performance on downstream metrics
// - Why do scaling laws arise?
// - ...






== Is scale all you need?
*Yes*. You can go home now.

The rest of the lecture if for intellectual enjoyment only.
// What components influence the constants in the scaling laws, i.e. determine data and compute efficiency.

// Data, architecture, optimiser, hyperparameters
// iceberg under the water: infrastructure

// We focus on the "training algorithm" (but no clear boundaries): in particular, what optimiser + how to scale hyperparameters


// SPOKEN: "We're going to start with scaling laws, which have become the foundation
// for how we think about training large models today. The title is a reference to
// the common phrase in the community - that scale is all you need to achieve better
// performance."


// The “foundation model” training recipe: data is abundant, compute is scarce. Increase compute to get better performance

== The Empirical Phenomenon

*Key observation* (Kaplan et al., 2020): Loss decreases as a _power law_ in
- Number of parameters $N$
- Dataset size $D$
- Compute budget $C$

$ L(N) = (N_c / N)^(alpha_N), quad L(D) = (D_c / D)^(alpha_D), quad L(C) = (C_c / C)^(alpha_C) $

where $alpha_N approx 0.076$, $alpha_D approx 0.095$, $alpha_C approx 0.050$.

// SPOKEN: "Kaplan et al. made a remarkable empirical observation: the loss follows a
// power law as we scale up. This isn't just an approximate trend - it's remarkably
// precise across many orders of magnitude. The exponents here tell us something
// important: the loss improves more slowly with parameters than with data, suggesting
// data might be more valuable."

// TODO: Add figure showing the power law curves from Kaplan et al. (2020) -
// the classic log-log plots of loss vs N, D, and C


== Power Laws: Why Log-Log Plots?

A power law $y = a x^(-b)$ becomes linear in log-log space:

$ log y = log a - b log x $

- Slope gives the exponent $b$
- Intercept gives the scaling constant $a$
- Linear fit over many orders of magnitude = strong evidence for power law

// SPOKEN: "Why do we always show these as log-log plots? Because a power law becomes
// a straight line in log-log space. If you see a straight line over many orders of
// magnitude, that's strong evidence you have a true power law relationship, not just
// a coincidence at a particular scale."


== The Combined Scaling Law

When scaling both $N$ and $D$ together:

$ L(N, D) = lr[(N_c / N)^(alpha_N / alpha_D) + D_c / D]^(alpha_D) $

Key insight: Loss is _bottlenecked_ by the smaller resource.

- Huge model + tiny data $arrow.r$ limited by data
- Tiny model + huge data $arrow.r$ limited by model capacity

// SPOKEN: "The combined scaling law shows that performance is bottlenecked by whichever
// resource is scarcer. Training a massive model on a small dataset wastes capacity.
// Training a small model on massive data also wastes potential. The key question
// becomes: what's the right balance?"


== The Foundation Model Paradigm

*Traditional ML mindset*:
- Data is scarce and expensive
- Worry about overfitting
- Regularisation is crucial

*Foundation model mindset*:
- Data is abundant (the internet)
- Compute is the bottleneck
- Scale up to get better performance

// SPOKEN: "Modern large-scale training represents a paradigm shift. In traditional ML,
// we worried about overfitting and regularisation because data was scarce. In the
// foundation model era, we have essentially unlimited data from the internet. The
// constraint has shifted to compute: how much can we afford to spend on training?"


== Generalisation in the Scaling Regime

*Surprising observation*: In the scaling regime, generalisation "comes for free"

The training loss is an unbiased estimator of the test loss:

$ EE[L_"train"] approx L_"test" $

Why? We typically see each training example _at most once_.

// SPOKEN: "Here's something that surprises people from traditional ML backgrounds:
// we don't really worry about generalisation error in this regime. Because datasets
// are so massive, we typically see each example at most once during training. That
// means training loss is essentially an unbiased estimate of test loss. The training
// and validation curves overlap almost perfectly."

// TODO: Add Runa's figure showing training vs validation loss overlap


== Why Does This Matter?

The generalisation gap is negligible because:

- Single-epoch training (or close to it)
- No memorisation of specific examples
- Model learns general patterns, not training set specifics

This simplifies our objective: just minimise training loss!

// SPOKEN: "This dramatically simplifies what we're trying to do. We don't need
// complicated regularisation schemes or early stopping heuristics. We can just
// focus on driving down the training loss, and test performance will follow.
// The challenge shifts from 'how do we generalise?' to 'how do we train efficiently?'"


== Scaling Laws as a Practical Tool

Scaling laws enable three crucial capabilities:

+ *Comparing training setups* - Which algorithm is better at scale?

+ *Projecting performance* - What performance can I expect for a given budget?

+ *Compute-efficient training* - How should I allocate my compute?

// SPOKEN: "This brings us to why scaling laws are so practically important. They're
// not just a curiosity - they're a tool for making expensive decisions. Let's look
// at each of these in turn."


== 1. Comparing Training Setups

*Problem*: Is algorithm A better than B?

*Naive approach*: Compare at one scale $arrow.r$ can be misleading!

*Better approach*: Compare the scaling _curves_

- Algorithm A might be better small-scale, B better large-scale
- The crossing point matters for practical decisions

// SPOKEN: "When comparing algorithms, you can't just run one experiment. An algorithm
// that wins at small scale might lose at large scale, or vice versa. You need to
// understand how they scale. This is why papers increasingly show scaling curves
// rather than single-point comparisons."

// TODO: Add figure showing two scaling curves that cross - one algorithm better
// at small scale, another better at large scale


== 2. Projecting Performance

Given a scaling law fit, we can:

- *Predict* final performance before training
- *Estimate* required compute for a target loss
- *Plan* infrastructure and budgets

Fit the law on small runs, extrapolate to production scale.

// SPOKEN: "Once you've fit a scaling law from smaller experiments, you can predict
// what a much larger run will achieve. This is incredibly valuable for planning.
// Before committing millions of dollars to a training run, you can estimate what
// you'll get. It's not perfect, but it's far better than guessing."


== 3. Compute-Efficient Training

*The key question*: Given compute budget $C$, how to allocate between:
- Model size $N$ (more parameters)
- Training data $D$ (more tokens)

Both cost compute! $C approx 6 N D$ (for transformers)

We want: $min_(N,D) L(N, D) quad "subject to" quad 6 N D = C$

// SPOKEN: "Here's where scaling laws become most actionable. Given a fixed compute
// budget, should you train a larger model for fewer steps, or a smaller model for
// more steps? The compute is roughly proportional to N times D, so there's a
// trade-off. Scaling laws tell us the optimal balance."


== The Kaplan Recipe (2020)

Original Kaplan et al. conclusion:

*Scale the model faster than the data*

For compute-optimal training:
$ N prop C^(0.73), quad D prop C^(0.27) $

Led to training very large models on relatively little data.

// SPOKEN: "Kaplan et al. initially concluded that you should scale the model much
// faster than the data. This led to the approach of training massive models - like
// GPT-3 with 175 billion parameters - on relatively small datasets. But as we'll
// see, this wasn't quite right."


== The Chinchilla Correction (2022)

Hoffmann et al. found different exponents:

*Scale model and data equally*

$ N prop C^(0.5), quad D prop C^(0.5) $

Rule of thumb: ~20 tokens per parameter

_"Chinchilla"_ (70B params, 1.4T tokens) outperformed _"Gopher"_ (280B params, 300B tokens)

// SPOKEN: "Two years later, the Chinchilla paper overturned this. They found that
// model and data should scale equally. Their 70B model trained on much more data
// actually outperformed a 280B model trained on less data - while using the same
// compute! This had huge implications for the field."

// TODO: Add figure comparing Chinchilla vs Gopher - same compute, different allocation


== Why Did Kaplan Get It Wrong?

Key difference: how they controlled for compute.

- Kaplan: Fixed tokens per run, varied model size
- Chinchilla: Varied both systematically

The lesson: experimental design matters enormously when fitting scaling laws.

// SPOKEN: "Why did two careful teams reach different conclusions? It came down to
// experimental design. Kaplan held some things fixed that they shouldn't have,
// which biased their estimates. This is a cautionary tale - fitting scaling laws
// requires careful experimental design, not just curve fitting."


== Deriving the Chinchilla Optimal

Given: $L(N, D) approx E + A / N^alpha + B / D^beta$

Constraint: $C = 6 N D$ (compute budget)

Optimise via Lagrangian:
$ N^* prop C^(beta / (alpha + beta)), quad D^* prop C^(alpha / (alpha + beta)) $

When $alpha approx beta$: scale both equally.

// SPOKEN: "Let's derive this. If we model the loss as an irreducible term plus
// power-law contributions from model and data limitations, we can use Lagrange
// multipliers to find the optimal allocation. The result depends on the ratio of
// the exponents. When they're similar - as Chinchilla found - you should scale
// model and data at the same rate."


== Practical Implications

*Before Chinchilla*:
- Train the biggest model you can afford
- Data is "cheap", compute is precious

*After Chinchilla*:
- Balance model size and training data
- Consider inference costs (smaller model = cheaper to deploy)
- Data quality and quantity both matter

// SPOKEN: "This changed how the field thinks about training. Before, the instinct
// was to train the biggest possible model. Now, people think more carefully about
// the trade-off. Importantly, smaller models are cheaper at inference time, so
// Chinchilla-optimal training often gives you a better deal overall."


== The Compute-Optimal Frontier

The _frontier_ is the set of (N, D) pairs achieving lowest loss for each compute level.

Points below the frontier: inefficient
- Too large for compute budget (undertrained)
- Too small for compute budget (capacity-limited)

// SPOKEN: "We can visualise this as a frontier in N-D space. Any point below the
// frontier represents inefficient training - either you've made the model too big
// and undertrained it, or you've made it too small and wasted data. The frontier
// shows the efficient trade-offs."

// TODO: Add figure showing the compute-optimal frontier in N-D space, with
// iso-compute curves


== Active Research Questions

Scaling laws remain an active research area:

- *Data repetition*: What if high-quality data must be repeated?
- *Downstream tasks*: Do scaling laws predict task-specific performance?
- *Why power laws?*: Is there a theoretical explanation?
- *Breaking the law*: Can we achieve better-than-power-law scaling?

// SPOKEN: "Scaling laws aren't a closed book. There are many open questions. What
// happens when you have to repeat data? Can we predict performance on specific
// downstream tasks, not just loss? Why do power laws emerge at all? And can we
// do better - perhaps through better architectures or training algorithms?"


== Is Scale All You Need?

*Yes*. You can go home now.

// SPOKEN: "So, is scale all you need? In some sense, yes. Scaling works. But..."

<<<<<<< Updated upstream
// Data, architecture, optimiser, hyperparameters
// iceberg under the water: infrastructure
=======
#pause
>>>>>>> Stashed changes

The rest of the lecture is for intellectual enjoyment only.

// SPOKEN: "...the rest of this lecture is about what determines the *constants* in
// the scaling laws. Given the same scale, why are some models better than others?
// The answer lies in the architecture, the data, the optimiser, and the hyperparameters.
// We'll focus especially on the training algorithm - what optimiser to use, and how
// to scale hyperparameters as we scale up the model."


== What Determines the Constants?

The scaling law $L = C_0 / X^alpha$ has two parts:

- *Exponent* $alpha$: How fast does scaling help? (Seems universal-ish)
- *Constant* $C_0$: Where does the curve sit? (Very much not universal)

Improvements in $C_0$ shift the whole curve down.

// SPOKEN: "The scaling exponent seems relatively stable across different setups -
// it's roughly 0.05 for compute. But the constant can vary enormously. A better
// architecture, better data, or better training algorithm shifts the entire curve
// down. That's effectively free performance at every scale."


== Components That Affect Efficiency

What determines $C_0$? Everything that isn't scale:

- *Data*: Quality, distribution, preprocessing
- *Architecture*: Attention, normalisation, positional encoding
- *Optimiser*: Adam, SGD, second-order methods
- *Hyperparameters*: Learning rate, batch size, schedules

We focus on: *optimiser + hyperparameter scaling*

// SPOKEN: "Many things affect where your scaling curve sits. We'll focus on the
// training algorithm side: which optimiser should you use, and critically, how
// should you scale hyperparameters as model size grows? Getting this wrong can
// waste enormous amounts of compute."


== Preview: The Challenge Ahead

Key questions for the rest of the lecture:

+ *Which optimiser?* Is Adam good enough? When to use something else?

+ *How to scale hyperparameters?* Learning rate that works at 100M parameters probably doesn't work at 100B

+ *Batch size scaling*: How does optimal batch size change with scale?

// SPOKEN: "Here's what we'll tackle next. Choosing an optimiser is surprisingly
// subtle - the answer depends on scale. And hyperparameters that work at small
// scale often fail at large scale. We need principled methods to transfer
// hyperparameters across scales. That's the subject of the next sections."
