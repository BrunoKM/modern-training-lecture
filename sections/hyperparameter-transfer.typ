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
    image("../figures/larger-not-always-better.png"),
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
- Identify the scaling coefficients $a^((ell)),b^((ell))$ for each $ell$ so that all quantities in the neural network throughout training remain _stable_  and _“sensible”_ as $n -> infinity$
- Turns out this leads to a unique parameterisation!
]

#slide(title: "Warmup: initialisation scale")[
  === Model definition
  // Define a simple feedforward neural network
  === “Stability” desiderata
  // 
]

// Derivation...


#slide(title: "Real desiderata")[
  // Give all the desiderata from Tensor Programs IV and V
  #cite(<yang2023tensorprogramsvifeature>)
  ...

]

#slide(title: "Results")[
  With $mu$P, you will keep getting better performance as you scale:

]


// Go through all the practical considerations:
#slide(title: "Practical considerations when implementing")[
  ...

]


#slide(title: "Beyond width")[
  Extensions to handle _both_ *width* and *depth*:
  #figure(
    image("../figures/width-and-depth-transfer-completedp.png"),
    caption: [
      Complete-P and Completed-P#cite(<mlodozeniec2025completedhyperparametertransfermodules>) extend $mu$P for width _and_ depth transfer in a modern transformer training setup.
    ]
  )
  
]


== $mu$P


== Batch-size reparameterisation

