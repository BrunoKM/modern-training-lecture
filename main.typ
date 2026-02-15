#import "@preview/touying:0.6.1": *

#import "setup.typ": *
#import "@preview/drafting:0.2.2": *

#set text(region: "GB")
#set text(font: "Charter")

#import themes.metropolis: *
#show: metropolis-theme.with(
  aspect-ratio: "4-3",
  config-info(
    title: [*Training Neural Networks at Scale*],
    author: "Bruna Mlodozeschenhagen",
    layout: "medium",
    toc: true,
    count: none,
  ),
  config-colors(
    primary: palette3,
    primary-light: rgb("#d6c6b7"),
    secondary: rgb("#23373b"),
    neutral-lightest: white,
    neutral-dark: rgb("#23373b"),
    neutral-darkest: rgb("#23373b").darken(30%),
  ),
)


// #import themes.simple: *
// #show: simple-theme.with(aspect-ratio: "4-3")


#title-slide()

#slide(title: "Title")[
  Example slide
]

#empty-slide()[
  Content...
]

= Scaling Laws \ #text(weight: "thin", size: 0.8em)[_c.f._ Scale Is All You Need]

// Describe the empirical phenomenon (Kaplan et al.), Chinchilla

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

== 

// 






== Is scale all you need?
*Yes*. You can go home now.

The rest of the lecture if for intellectual enjoyment only.
// What components influence the constants in the scaling laws, i.e. determine data and compute efficiency.

// Data, architecture, optimiser, hyperparameters

// We focus on the "training algorithm" (but no clear boundaries): in particular, what optimiser + how to scale hyperparameters

= I. How to Choose the Optimiser

== Traditional optimisation
.

== On the difficulty of optimiser comparisons
.

== Elements of modern optimisation algorithms

=== Gradient descent

// Euclidean and non-euclidean?

=== Momentum

=== Preconditioning

== Adam and SignGD
.

== Shampoo and SpectralGD
.

== Other components

=== Weight decay

=== Hyperparameter schedules


= II. How to Scale Training Hyperparameters
== 

== $mu$P


== Batch-size reparameterisation


