#import "@preview/touying:0.6.1": *
#import "../setup.typ": *

= Scaling Laws \ #text(weight: "thin", size: 0.8em)[_c.f._ Scale Is All You Need]

== Scaling Laws: The Empirical Phenomenon
// Describe the empirical phenomenon (Kaplan et al.), Chinchilla
(Kaplan 2020)#cite(<kaplan2020scaling>) observe that training loss decreases predictably with:
- Model size $N$ (number of parameters) as $L(N) prop N^(-alpha_N)$
- Dataset size $D$ (number of tokens) as $L(D) prop D^(-alpha_D)$
#par(leading: 0.0em, spacing: 0.0em)[
  #text(size: 0.7em)[Exponentials become lines on a log-log plot:]
]

#figure(
  image("../figures/kaplan2020main.png"),
  caption: [
    *Left:* The best final (test) loss for a given compute budget (allocating compute either to dataset size or model size) also traces out a power law.\
    *Center:* Final (test) loss for a model of a fixed size trained on different number of tokens. \
    *Right:* Final (test) loss when training for a fixed number of tokens training models with different number of parameters.
  ],
)

== Scaling Law Forms
We can fit a joint scaling law (with parameters $A, B, alpha, gamma$) to characterise the behaviour:
*Kaplan 2020*: $L(N, D) = (A 1 / N^alpha) + B (1 / D))^gamma$
#figure(
  image("../figures/kaplan2020scaling1.png"),
)
Issue: The limiting loss as $N, D -> infinity$ won't necessarily be zero. It should converge to #text(size: 0.6em)[(at least)] some irreducible loss $L_0$ (entropy of data generating distribution).

== Scaling Law Forms: Hoffman et al.
#shortcite(<hoffmann2022training>) include an irreducible loss $L_0$
$
  L(N, D) = L_0 + A 1 / N^alpha + B 1 / D^beta
$
with constants $L_0, A, B, alpha, beta$.
== Scaling Law Forms
However, often in practice, the #shortcite(<kaplan2020scaling>) form with a shared exponent is a better fit. When combined with the irreducible loss $L_0$, this is a form often used in practice:
$
  L(N, D) = L_0 + (A 1 / N^alpha + B 1 / D )^gamma
$
with constants $L_0, A, B, alpha, gamma$.
== The Foundation Model Paradigm
*Traditional ML mindset*:
- Data is scarce and expensive
- Worry about overfitting
- Regularisation is crucial

*Foundation model mindset*:
- Data is abundant#footnote[or _can_ be made abundant: internet, synthetic data, verifiable problems (e.g. theorem proving).], compute is scarce.
- To get better performance $->$ just increase the compute
- When data is abundant, and we're not repeating examples (single epoch training), we don't have to worry about generalisation error. The training loss _is_ an unbiased estimate of the test loss.

#figure(
  image("../figures/from-generalization-to-scaling.png", width: 90%),
)
#text(size: 0.1em)[#cite(<lechau2024rethinking>)]


// The “foundation model” training recipe: data is abundant, compute is scarce. Increase compute to get better performance

// Don't need to worry about generalisation error. Generalisation error doesn't exist in this setting: Training loss is an unbiased estimate of the validation loss
// TODO: Runa has a figure for this


== Scaling laws as a _practical_ tool

1. *Comparing training setups* - Which algorithm is best _at scale_?


#import "../figures/scaling-plots.typ": method-comparison-scaling
#figure(
  method-comparison-scaling(),
  caption: [
    Comparing methods at different scales. Method A (blue) is better at Scale 1, but Method B (red) has a steeper slope and overtakes at Scale 2. The crossover point determines which method to use at production scale.
  ],
)

==
1. *Comparing training setups* - Which algorithm is best _at scale_?

#image("../figures/kaplan-lstm-vs-transformer.png")

==
2. *Projecting performance* - What performance can I expect if I invest $100times$ more into training?
#pause
+ *Compute-efficient training* - How should I allocate my compute?

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


== Scaling Laws as a Practical Tool


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
