// Scaling law figures using CeTZ
#import "@preview/cetz:0.3.2"

// Figure 1: Power law scaling - Loss vs Compute (simplified with raw primitives)
#let scaling-power-law() = {
  cetz.canvas({
    import cetz.draw: *

    // Axes
    line((0, 0), (8, 0), stroke: black)
    line((0, 0), (0, 5), stroke: black)

    // Axis labels
    content((4, -1.2), [Compute (log scale)])
    content((-0.8, 2.5), angle: 90deg, [Test Loss])

    // Power law curve: starts high, decreases smoothly
    let points = ()
    for i in range(0, 80) {
      let x = i * 0.1
      let y = 4.5 * calc.pow(0.7, x * 0.3)
      points.push((x, y))
    }

    line(..points, stroke: rgb("#2c61c2") + 2pt)

    // Data points
    for i in range(1, 8) {
      let x = i
      let y = 4.5 * calc.pow(0.7, x * 0.3) + 0.15 * calc.sin(i * 2)
      circle((x, y), radius: 0.08, fill: rgb("#2c61c2"), stroke: none)
    }

    // Tick marks
    for i in range(1, 8) {
      line((i, 0), (i, -0.1), stroke: black)
      content((i, -0.3), text(size: 0.7em)[$10^#(17 + i)$])
    }
    for i in range(1, 5) {
      line((0, i), (-0.1, i), stroke: black)
      content((-0.3, i), text(size: 0.7em)[#(5 - i)])
    }
  })
}

// Figure 2: IsoFLOP curves (Chinchilla style)
#let isoflop-curves() = {
  cetz.canvas({
    import cetz.draw: *

    // Axes
    line((0, 0), (8, 0), stroke: black)
    line((0, 0), (0, 5), stroke: black)

    content((4, -0.6), [Model Size (B params)])
    content((-0.8, 2.5), angle: 90deg, [Test Loss])

    // IsoFLOP curve for C1 (low compute) - minimum at small model
    let c1-points = ()
    for i in range(1, 80) {
      let x = i * 0.1
      let y = 1.2 + 0.08 * calc.pow(x - 1.5, 2) + 1.5 / (x + 0.5)
      c1-points.push((x, y))
    }
    line(..c1-points, stroke: rgb("#cc392a") + 1.5pt)

    // IsoFLOP curve for C2 (medium compute) - minimum at medium model
    let c2-points = ()
    for i in range(1, 80) {
      let x = i * 0.1
      let y = 0.8 + 0.05 * calc.pow(x - 3, 2) + 1.2 / (x + 0.5)
      c2-points.push((x, y))
    }
    line(..c2-points, stroke: rgb("#d97459") + 1.5pt)

    // IsoFLOP curve for C3 (high compute) - minimum at larger model
    let c3-points = ()
    for i in range(1, 80) {
      let x = i * 0.1
      let y = 0.5 + 0.03 * calc.pow(x - 5, 2) + 0.9 / (x + 0.5)
      c3-points.push((x, y))
    }
    line(..c3-points, stroke: rgb("#2c61c2") + 1.5pt)

    // Legend
    line((5.5, 4.5), (6.5, 4.5), stroke: rgb("#cc392a") + 1.5pt)
    content((7.2, 4.5), text(size: 0.7em)[$C_1$])
    line((5.5, 4.0), (6.5, 4.0), stroke: rgb("#d97459") + 1.5pt)
    content((7.2, 4.0), text(size: 0.7em)[$C_2$])
    line((5.5, 3.5), (6.5, 3.5), stroke: rgb("#2c61c2") + 1.5pt)
    content((7.2, 3.5), text(size: 0.7em)[$C_3$])

    // Mark minima
    circle((1.5, 2.2), radius: 0.1, fill: rgb("#cc392a"), stroke: none)
    circle((3.0, 1.5), radius: 0.1, fill: rgb("#d97459"), stroke: none)
    circle((5.0, 1.1), radius: 0.1, fill: rgb("#2c61c2"), stroke: none)

    // Optimal frontier line
    line((1.5, 2.2), (3.0, 1.5), (5.0, 1.1), stroke: (dash: "dashed", paint: gray))
  })
}

// Figure 3: Training vs Validation loss convergence
#let train-val-convergence() = {
  cetz.canvas({
    import cetz.draw: *

    // Axes
    line((0, 0), (8, 0), stroke: black)
    line((0, 0), (0, 5), stroke: black)

    content((4, -0.6), [Training Steps])
    content((-0.8, 2.5), angle: 90deg, [Loss])

    // Training loss curve
    let train-points = ()
    for i in range(1, 80) {
      let x = i * 0.1
      let y = 1 + 3.5 * calc.exp(-x * 0.4)
      train-points.push((x, y))
    }
    line(..train-points, stroke: rgb("#2c61c2") + 1.5pt)

    // Validation loss curve (slightly higher, with small noise)
    let val-points = ()
    for i in range(1, 80) {
      let x = i * 0.1
      let y = 1.05 + 3.5 * calc.exp(-x * 0.38)
      val-points.push((x, y))
    }
    line(..val-points, stroke: rgb("#cc392a") + 1.5pt)

    // Legend
    line((5.5, 4.5), (6.5, 4.5), stroke: rgb("#2c61c2") + 1.5pt)
    content((7.3, 4.5), text(size: 0.7em)[Train])
    line((5.5, 4.0), (6.5, 4.0), stroke: rgb("#cc392a") + 1.5pt)
    content((7.3, 4.0), text(size: 0.7em)[Val])
  })
}

// Figure 4: Method comparison at different scales
// Shows two methods where Method A is better at small scale but Method B overtakes at large scale
#let method-comparison-scaling() = {
  cetz.canvas({
    import cetz.draw: *

    // Axes
    line((0, 0), (8, 0), stroke: black)
    line((0, 0), (0, 5), stroke: black)

    // Axis labels
    content((4, -1.1), [Compute (log scale)])
    content((-0.8, 2.5), angle: 90deg, [Loss (log scale)])

    // Method A: starts lower, shallower slope (less efficient scaling)
    // y = 4.0 * x^(-0.15) in log space becomes a line
    let method-a-points = ()
    for i in range(5, 80) {
      let x = i * 0.1
      let y = 4.2 - 0.35 * x  // shallower slope
      method-a-points.push((x, y))
    }
    line(..method-a-points, stroke: rgb("#2c61c2") + 2pt)

    // Method B: starts higher, steeper slope (more efficient scaling)
    let method-b-points = ()
    for i in range(5, 80) {
      let x = i * 0.1
      let y = 4.8 - 0.55 * x  // steeper slope
      method-b-points.push((x, y))
    }
    line(..method-b-points, stroke: rgb("#cc392a") + 2pt)

    // Find intersection: 4.2 - 0.35x = 4.8 - 0.55x => 0.2x = 0.6 => x = 3
    let intersection-x = 3.0
    let intersection-y = 4.2 - 0.35 * intersection-x  // = 3.15

    // Vertical dashed line at small scale (x=1.5) - Method A is better
    line((1.5, 0), (1.5, 4.5), stroke: (dash: "dashed", paint: gray.darken(20%)))

    // Vertical dashed line at large scale (x=5.5) - Method B is better
    line((5.5, 0), (5.5, 4.5), stroke: (dash: "dashed", paint: gray.darken(20%)))

    // Small annotation at intersection point
    circle((intersection-x, intersection-y), radius: 0.08, fill: black, stroke: none)

    // Scale labels
    content((1.5, 4.8), text(size: 0.65em)[Scale 1])
    content((5.5, 4.8), text(size: 0.65em)[Scale 2])

    // Annotations for which is better at each scale
    content((1.5, -0.3), text(size: 0.6em, fill: rgb("#2c61c2"))[A better])
    content((5.5, -0.3), text(size: 0.6em, fill: rgb("#cc392a"))[B better])

    // Legend
    line((5.8, 3.8), (6.8, 3.8), stroke: rgb("#2c61c2") + 2pt)
    content((8.5, 3.8), text(size: 0.75em)[Method A])
    line((5.8, 3.3), (6.8, 3.3), stroke: rgb("#cc392a") + 2pt)
    content((8.5, 3.3), text(size: 0.75em)[Method B])

    // Tick marks on x-axis
    for i in range(1, 8) {
      line((i, 0), (i, -0.1), stroke: black)
    }
    // Tick marks on y-axis
    for i in range(1, 5) {
      line((0, i), (-0.1, i), stroke: black)
    }
  })
}

// Figure 5: Chinchilla vs Gopher comparison (bar chart style)
#let chinchilla-comparison() = {
  cetz.canvas({
    import cetz.draw: *

    // Model size comparison
    content((2, 5.2), text(weight: "bold", size: 0.9em)[Parameters])

    // Gopher bar
    rect((0.5, 0), (2, 4), fill: rgb("#4a90d9"))
    content((1.25, 4.3), text(size: 0.8em)[280B])
    content((1.25, -0.4), text(size: 0.8em)[Gopher])

    // Chinchilla bar
    rect((2.5, 0), (4, 1), fill: rgb("#e07b53"))
    content((3.25, 1.3), text(size: 0.8em)[70B])
    content((3.25, -0.4), text(size: 0.8em)[Chinchilla])

    // Loss comparison
    content((7.5, 5.2), text(weight: "bold", size: 0.9em)[Final Loss])

    // Gopher loss bar
    rect((6, 0), (7.5, 3.68), fill: rgb("#4a90d9").lighten(30%))
    content((6.75, 4.0), text(size: 0.8em)[1.84])
    content((6.75, -0.4), text(size: 0.8em)[Gopher])

    // Chinchilla loss bar (lower = better)
    rect((8, 0), (9.5, 3.54), fill: rgb("#e07b53").lighten(30%))
    content((8.75, 3.86), text(size: 0.8em)[1.77])
    content((8.75, -0.4), text(size: 0.8em)[Chinchilla])

    // Arrow indicating "better"
    line((10.2, 3.54), (10.2, 0.5), stroke: gray + 1pt, mark: (end: ">"))
    content((10.2, 0.2), text(size: 0.6em)[better])
  })
}
