#import "@preview/touying:0.6.1": *
#import "../setup.typ": *

= Scaling Laws \ #text(weight: "thin", size: 0.8em)[_c.f._ Scale Is All You Need]

== Scaling Laws: The Empirical Phenomenon
// Describe the empirical phenomenon (Kaplan et al.), Chinchilla
#shortcite(<kaplan2020scaling>) observe that training loss decreases predictably with:
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
$
L(N, D) = (A 1 / N^alpha + B 1 / D)^gamma   quad quad #text[(_Kaplan scaling law_)]
$
#figure(
  image("../figures/kaplan2020scaling1.png"),
)
Issue: The limiting loss as $N, D -> infinity$ won't necessarily be zero. It should converge to #text(size: 0.6em)[(at least)] some irreducible loss $L_0$ (entropy of data generating distribution).

== Scaling Law Forms: Hoffman et al.
#shortcite(<hoffmann2022training>) include an irreducible loss $L_0$:

$
  L(N, D) = L_0 + A 1 / N^alpha + B 1 / D^beta quad quad #text[(_Chinchilla scaling law_)]
$

with constants $L_0, A, B, alpha, beta$.
== Scaling Law Forms
However, often in practice, the #shortcite(<kaplan2020scaling>) form with a shared exponent is a better fit. When combined with the irreducible loss $L_0$, this is a form often used in practice:
$
  L(N, D) = L_0 + (A 1 / N^alpha + B 1 / D )^gamma
$
with constants $L_0, A, B, alpha, gamma$.
== The Foundation Model Paradigm
*Traditional ML approach*:
- Data is scarce and expensive
- Worry about overfitting
- Regularisation is crucial

*Foundation model approach*:
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
)
#pause
  To compare machine learning methods, it's not enough to compare how they perform on one fixed dataset. *We need to compare how they scale.*

==
1. *Comparing training setups* - Which algorithm is best _at scale_?
#image("../figures/kaplan-lstm-vs-transformer.png")
Example comparison of LSTMs against transformers#cite(<kaplan2020scaling>)

==
2. *Projecting performance* – If I invest $100times$ into compute for training a larger model, what performance can I expect?   
==
3. *Compute-efficient training* - How should I allocate my compute?
Take for instance the Chinchilla scaling law: $L(N, D) = L_0 + A 1 / N^alpha + B 1 / D^beta$

For a given compute cost estimate $C(N, D)$ (e.g., $C(N, D) = 6 N D$ used in #cite(<hoffmann2022training>)) we can use e.g. Lagrange multipliers to find optimal $N, D$ for a given compute budget $C_0$.
$
nabla_(N,D, lambda) [L(N,D) + lambda (6 N D - C_0)] = 0
$
Solving the above gives the (compute optimal) constraint: $(alpha A) / (N^alpha) = (beta B ) / (D^beta)$.

When plugging in the values of $alpha, beta, A, B$ from the Chinchilla paper, we get the infamous _Chinchilla scaling rule_:
$
D=20 N
$

#text(size: 0.9em, fill: red)[But this is highly setup dependent (will depend on values of $alpha, beta, A, B$). Will not hold for a modern traininig setup!]

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

Key questions for the rest of the lecture:

+ *Which optimiser?* Is Adam good enough? Can we do better?

+ *How to scale hyperparameters?* Learning rate that works at 100M parameters doesn't work at 100B parameters. How do you scale in a principled way without costly retuning?
// + *Batch size scaling*: How does optimal batch size change with scale?
