#import "@preview/touying:0.6.1": *
#import themes.metropolis: *
#import "../setup.typ": *

= II. How to Scale Training Hyperparameters

==
#empty-slide[
  Did #shortcite(<kaplan2020scaling>) actually show that all you need is scale?

  #image("../figures/kaplan2020main.png", width: 80%)
  The devil's in the details:

  #block(width: 100%)[
    #set text(size: 0.9em, style: "italic")
    #set align(center)
    #block(width: 90%)[
      #set align(left)
      #quote[*We tuned learning rates*, and we experimented with learning rate schedules. But we may have neglected to tune some hyperparameter (e.g. intialization scale or momentum) that have an importanteffect on scaling.]
    ]
    #set align(left)
  ]


]
#empty-slide[
  In fact, larger models can have _worse_ performance if you don't adjust your hyperparameters! @yang2022tensorprogramsvtuning

  #figure(
    image("../figures/larger-not-always-better.png", height: 70%),
  )


]

#empty-slide[
  == A few solutions:
  - *Always tune your hyperparameters at every scale*
    - #text(fill: red)[Intractable when training models at a 100B+ parameter scale]
  - *Train at small scale, extrapolate optimal setup to large scale:*
    1. Fit a simple regression model mapping model and data size $N, D$ to optimal hyperparameters $gamma^*(N, D)$ at that scale. Extrapolate to much larger scales.
      - Naïve, but actually used in practice (e.g. DeepSeek#footnote[“DeepSeek LLM: Scaling Open-Source Language Models with Longtermism”])
    2. Figure out why things are going wrong, and derive principled parameterisations that scale well.


]

#slide(title: "One solution: Maximal Update Parameterisation")[
  #shortcite(<yang2022tensorprogramsvtuning>) aim to find a *maximal update parameterisation* under which _scaling *in width* is stable_, _consistently gives improved performance_, and _optimal hyperparameters transfer across scale_.


  === Idea behind the *Maximal Update Parameterisation* ($mu$P)
  - Parameterise the range os possible rules ways to scale hyperparameters in width $n$. For parameters at each layer $ell$:
  $
    #text()[Learning rate:] eta^((ell)) = eta_0 / n^(b^((ell))) quad #text[Initialisation scale] sigma^(ell) = sigma_0 / n^(a^((ell)))
  $
  - Identify the scaling coefficients $a^((ell)),b^((ell))$ for each $ell$ so that all quantities in the neural network throughout training remain _stable_  and _“sensible”_ as $n -> infinity$.
  - Turns out this leads to a unique parameterisation!
]

== The $mu$P desiderata
=== Model Definition

Take a feedforward network with $L$ layers of width $n$, acting on an input $x in RR^d$:

- *Layer widths:* $n_0 = d, quad n_1, dots, n_(L-1)=n, quad n_L = 1$
- *Features:* $h^((0))(x) := x$
- *Pre-activations:* $f^((l))(x) := W^((l)) h^((l-1))(x)$ #h(3em) for $ell = 1, dots, L$
- *Post-activations:* $h^((l))(x) := phi(f^((l))(x))$#h(3em) for $ell = 1, dots, L$

Here $phi: RR -> RR$ is an element-wise non-linearity and $W^((l)) in RR^(n_l times n_(l-1))$.

We will denote quantities at training step $t$ (for $t in {0, 1, 2, ...}$) with subscript $t$ #text(size: 0.8em, fill: gray)[(e.g. $f_t^((ell))(x)$ denotes pre-activations computed on $x$ with weights $(W_t^((0)),W_t^((1)))$ at timestep $t$)].

Weights are initialised element-wise #iid as:
$
  [W^((l))_0]_(i j) tilde cal(N)(0, alpha_l \/ n^(a^((ell))))
$

== The $mu$P desiderata

#slide(title: "Preliminaries: Asymptotic Notation")[
  We analyse behaviour as width $n -> infinity$. Recall *Landau notation* for a sequence of real numbers $(X_1, X_2, ...)$:

  - $X_n = O(1)$ means _bounded above_: $thick exists b, N > 0$ such that $|X_n| <= b$ for sufficiently large $n >= N$
  - $X_n = Omega(1)$ means _bounded below_: $thick exists a, N > 0$ such that $|X_n| >= a$ for sufficiently large $n >= N$
  - $X_n = Theta(1)$ means _bounded both form below and above_: $thick$ both $O(1)$ and $Omega(1)$

  #text(
    size: 0.7em,
    fill: gray.darken(30%),
  )[Technicality: For *random* sequences, we'll require these bounds hold *almost surely*.]

  We'll be dealing with sequences of _vectors_ of different size (e.g. activations in a given layer as we take width $n -> infinity$). We need to define what it means for these to be $O(1), Omega(1), Theta(1)$:

  #block(
    fill: luma(245),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    *Coordinate size* of $bold(v) in RR^n$: $quad "coord-size"(bold(v)) := sqrt(1/n sum_(i=1)^n v_i^2) = norm(bold(v))_2 \/ sqrt(n)$

    #v(0.3em)

    We say $bold(v)_n$ has $Theta(1)$ coordinate size if $"coord-size"(bold(v)_n) = Theta(1)$ as $n -> infinity$.
  ]
]

#slide(title: [$mu$P _desiderata_: What should hold as width → ∞?])[
  We want to identify parameterisations where *sensible* behaviour persists as we scale width $n -> infinity$.

  #v(0.2em)

  *1. Stability Desiderata:* Nothing should blow up during training.
  - Pre-activations $f_t^((ell))(x)$ and activations $h_t^((ell))(x)$ have $O(1)$ coordinate size
  - Network output $f_t^((L))(x)$ remains $O(1)$
  - Changes during training don't explode: $h_t^((ell)) - h_0^((ell))$ has $O(1)$ coordinate size

  #text(fill: palette1.lighten(30%), size: 0.9em)[
    _Without stability_:
    - Activations/gradients explode.Training will diverge at large width
    Default PyTorch/TensorFlow parameterisations _are_ unstable!
  ]

]
==
*2. Non-triviality desideratum:* The network should actually learn.
- Output changes during training: $f_t^((L)) - f_0^((L))$ is $Omega(1)$

#text(fill: palette1.lighten(30%), size: 0.9em)[
  Without non-triviality:
  - Network output doesn't change (can be achieved with e.g. `learning rate` $= 0$)
]
#v(0.3em)

#slide(title: "The Maximal Update Criterion")[
  Stability + non-triviality still leave many possible parameterisations.
  #text(size: 0.8em)[
    - E.g. the *“NTK parameterisation”* (learning rate $prop 1\/n$) satisfies stability and non-triviality, but *features don't change* in the limit --- the network behaves like a linear model!#footnote[This has led many people to conclude infinite width limits are pathological. But NTK is only one possible limit!]
  ]


  Need one final desideratum:
  #block(
    stroke: 0.5pt + luma(150),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    *Maximal Feature Learning:* Weight _updates_ should maximally affect feature computation:
    $
      (W_t^((ell)) - W_0^((ell))) h_t^((ell-1)) quad "has" quad Theta(1) "coordinate size"
    $
    for all layers $ell$.
  ]

  This criterion requires that the *change in weights* at each layer meaningfully changes how that layer processes its inputs --- not just a vanishing perturbation.
]


#empty-slide()[
  #block(
    stroke: 0.5pt + luma(150),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    *Maximal Feature Learning:* Weight _updates_ should maximally affect feature computation:
    $
      (W_t^((ell)) - W_0^((ell))) h_t^((ell-1)) quad "has" quad Theta(1) "coordinate size"
    $
    for all layers $ell$.
  ]

  #block(
    fill: luma(245),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    #shortcite(<yang2022tensorprogramsvtuning>) show that there is a *unique* parameterisation for SGD satisfying all these criteria --- the $mu$-Parameterisation ($mu$P).
  ]


  #text(size: 0.8em)[
    #align(center)[
      #table(
        columns: (auto, auto, auto),
        inset: 8pt,
        align: center,
        stroke: 0.5pt + luma(150),
        [*Layer type*], [Init. scale $sigma^((ell))$], [Learning rate $eta^((ell))$],
        [Input], [$sigma_0 \/ sqrt(n)$], [$eta_0 dot n$],
        [Hidden], [$sigma_0 \/ sqrt(n)$], [$eta_0$],
        [Output], [$sigma_0 \/ n$], [$eta_0 \/ n$],
      )
    ]
  ]
]

// A "taster" for the style of derivation.
#empty-slide[
  == Warmup: Deriving initialisation scale

  Key insights:

  #block(
    fill: luma(245),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    *Layer 1:* Pre-activations are Gaussian (sums of independent Gaussians). \
    *Layer $l > 1$:* Pre-activations converge to Gaussian as `width` $-> infinity$ by the Central Limit Theorem.
  ]

]
#empty-slide[

  == Model Definition

  Take a feedforward network with $L$ layers of width $n$, acting on an input $x in RR^d$:

  - *Layer widths:* $n_0 = d, quad n_1, dots, n_(L-1)=n, quad n_L = 1$
  - *Features:* $h^((0))(x) := x$
  - *Pre-activations:* $f^((l))(x) := W^((l)) h^((l-1))(x)$ #h(3em) for $ell = 1, dots, L$
  - *Post-activations:* $h^((l))(x) := phi(f^((l))(x))$#h(3em) for $ell = 1, dots, L$

  Here $phi: RR -> RR$ is an element-wise nonlinearity and $W^((l)) in RR^(n_l times n_(l-1))$.

  *Initialization.* Weights are drawn #iid:
  $
    W_(i j)^((l)) tilde cal(N)(0, alpha_l \/ n^(a^((ell))))
  $
  By choosing the coeffient $a^((ell))$ for each layer, we can control how the variance of pre-activations scales with width.

]
#empty-slide[
  == Layer 1: Finding the constraint on $a^((1))$
  The $i$-th pre-activation in layer 1 is:
  $
    f_i^((1))(x) = sum_(j=1)^d W_(i j)^((1)) x_j
  $

  Since $W_(i j)^((1)) tilde cal(N)(0, alpha_1 slash n^(a^((1))))$ (#iid), this is a sum of independent Gaussians. Computing the variance:
  #v(-1.0em)
  $
    Var(f_i^((1))(x)) = sum_(j=1)^d x_j^2 dot alpha_1 / n^(a^((1))) = (alpha_1 norm(x)^2) / n^(a^((1)))
  $
  #v(-1.0em)
  For inputs with $norm(x)^2 = Theta(d) = Theta(1)$ (i.e., input dimension is fixed), the coordinate size of $f^((1))(x)$ is:
  #v(-1.0em)
  $
    "coord-size"(f^((1))(x)) = sqrt(Var(f_i^((1)))) = sqrt(alpha_1 norm(x)^2) / n^(a^((1)) slash 2)
  $

  #block(
    stroke: 1pt + palette1,
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    *Stability constraint:* For $f^((1))(x)$ to have $Theta(1)$ coordinate size, need: $a^((1)) = 0$.
  ]

]
#empty-slide[
  == Hidden Layers: Finding the constraint on $a^((ell))$ for $ell > 1$

  For hidden layers, $n_(ell-1) = n$. The $i$-th pre-activation is:
  $
    f_i^((ell))(x) = sum_(j=1)^n W_(i j)^((ell)) h_j^((ell-1))(x) = alpha_ell / (n^(1/2 a^((ell)))) sum_(j=1)^n epsilon_(i j)^((ell)) h_j^((ell-1))(x)
  $

  where $epsilon_(i j)$ are #iid $cal(N)(0, 1)$. Each term $W_(i j)^((ell)) h_j^((ell-1))$ is independent with variance:
  $
    Var(W_(i j)^((ell)) h_j^((ell-1))) = alpha_ell / n^(1/ 2 a^((ell))) dot EE[(h_j^((ell-1)))^2]
  $
  (Heuristically) letting the width of the first layer go to infinity, we have that $EE[(h_j^((ell-1)))^2] = c$ for some constant $c$.

  By the Central Limit Theorem, if we set $a^((ell))=1$, we get that:
  $
    f_i^((ell))(x) = alpha_ell / (sqrt(n)) sum_(j=1)^n epsilon_(i j)^((ell)) h_j^((ell-1))(x) -> N(0, tilde(sigma)_ell^2)
  $
  for some $tilde(sigma)_ell^2$. If we set $a^((ell)) < 1$, the variance of $f_i^((ell))(x)$ would diverge as $n -> infinity$. If we set $a^((ell)) > 1$, the variance would vanish.



  #block(
    stroke: 1pt + palette1,
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    *Stability constraint:* For $f^((ell))(x)$ to have $Theta(1)$ coordinate size, we need $a^((ell)) = 1$.
  ]

]

#slide(title: "Results")[
  With $mu$P, you will keep getting better performance as you scale:
]




#slide(title: "Beyond width")[
  Extensions to handle _both_ *width* and *depth*:
  #figure(
    image("../figures/width-and-depth-transfer-completedp.png"),
    caption: [
      Complete-P and Completed-P#cite(<mlodozeniec2025completedhyperparametertransfermodules>) extend $mu$P for width _and_ depth transfer in a modern transformer training setup.
    ],
  )

]


// // Go through all the practical considerations:
// #slide(title: "Practical considerations when implementing")[
//   ...
//
// ]
// == $mu$P
//
//
// == Batch-size reparameterisation
//
