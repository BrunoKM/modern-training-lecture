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

// Example slide commands:

// #slide(title: "Title")[
//   Example slide@malladi2022sdes
// ]

// #empty-slide()[
//   Content...
// ]

#include("sections/scaling-laws.typ")
#include("sections/optimisers.typ")
#include("sections/hyperparameter-transfer.typ")

#bibliography("references.bib", style: "chicago-notes")