= I. How to Choose the Optimiser

== Traditional optimisation
// Convex -> optimality gap
// Non-convex -> gradient norm bound
// online/adverserial -> regret bound

// influences on rate: stochasticity and smoothness
// worst case guarantees are too pessimistic, e.g. SGD is optimal but clearly we can do better in practice
// limitation: these assumptions don't use strcuture inherint in deep learning problems, e.g. compositional, hierachicial nature of architectures and data distribution

- Traditionally, we can derive *convergence guarantees* by making assumptions
- However, it is *questionable* if to what extent the *assumptions* (e.g., convexity, smoothness) are justified in modern deep learning settings #cite(<tran2025reevaluating>)
- At the same time, worst case results are typically *too conservative* since they do not leverage the unique structure of the problem (e.g., architecture, data)
- Ultimately, we rely on *empirical evaluations*

#let hd = text.with(size: 11pt, fill: luma(90), weight: "bold")
#let note = text.with(size: 14pt, fill: luma(140))
#let col-green = rgb("#16a34a")
#let col-amber = rgb("#d97706")
#let col-violet = rgb("#7c3aed")
#table(
  columns: (0.85fr, 2fr, 2fr, 2fr),
  align: (col, row) => if row == 0 { center + horizon } else if col == 0 { center + horizon } else { center + horizon },
  inset: (x: 10pt, y: 9pt),
  stroke: 0.5pt + luma(200),
  fill: (col, row) => if row == 0 { luma(245) } else if col == 0 { luma(250) } else { none },
  // Header
  [],
  [#text(size: 14pt, weight: "bold", fill: col-green)[Convex]],
  [#text(size: 14pt, weight: "bold", fill: col-amber)[Non-Convex]],
  [#text(size: 14pt, weight: "bold", fill: col-violet)[Online / Adversarial]],
  // Row 1: Guarantee (math)
  [#hd[GUARANTEE]],
  [#set text(size: 11pt)
   $ f(x_T) - f(x^*) <= epsilon $
   #v(0.1cm)
   #note[Function-value gap to \ the #text(weight: "bold")[global optimum]]],
  [#set text(size: 11pt)
   $ min_(t <= T) || nabla f(x_t) ||^2 <= epsilon $
   #v(0.1cm)
   #note[Proximity to a \ #text(weight: "bold")[stationary point]]],
  [#set text(size: 11pt)
   $ R_T = sum_(t=1)^T f_t (x_t) - min_x sum_(t=1)^T f_t (x) $
   #v(0.1cm)
   #note[#text(weight: "bold")[Regret] vs. best fixed action]],
  // Row 2: Smoothness effect
  [#hd[SMOOTHNESS \ EFFECT]],
  [#set text(size: 11pt)
   Non-smooth: $O(1 \/ sqrt(T))$ \
   Smooth: $O(1 \/ T)$ \
   Further acceleration through access to curvature],
  [#set text(size: 11pt)
   Required as baseline \ ($L$-Lipschitz $nabla f$)
   #v(0.05cm)
   Smooth: $O(1 \/ T)$ on $||nabla f||^2$],
  [#set text(size: 11pt)
   Non-smooth: $O(sqrt(T))$ regret \
   Smooth: $O(log T)$ regret],
  // Row 3: Stochasticity effect
  [#hd[STOCHASTIC \ ORACLE]],
  [#set text(size: 11pt)
   Costs $sqrt(T)$: \ smooth rate $O(1\/T) arrow.r O(1\/sqrt(T))$],
  [#set text(size: 11pt)
   $ 1/T sum_(t=0)^(T-1) EE || nabla f(x_t) ||^2 <= O(1\/sqrt(T)) $],
  [#set text(size: 11pt)
   Adversarial is already worst-case, \ stochastic problems are easier],
)


== On the difficulty of optimiser comparisons
// if theory can't tell us what to use (1) how do we come up with new algorithms, and (2) decide which one to use?
// (1) more or less well motivated modifications, e.g. point out divergence of Adam in some unrealistic setting and propose fix
// (2) benchmarking
// We have hundreds or thousends of optimisers, but comparing them empirically is extremely hard
// a) setup depends on exact questions asked, e.g. care about compute (measured how?) or data efficiency
// b) implementation differences
// c) costly hyperparameter tuning
// d) possible problem specificity

// Examples of efforts trying to deal with this: ...

// However, we can't possibly benchmark every new variation that is proposed.
// Luckily, practially all effective algorithms share the same building blocks!

*Comparing neural network optimisation algorithms is harder than it seems!*

1. The precise *definition of performance* (e.g. compute vs. data efficiency or optimisation vs. generalisation) results in very different setups.
2. We can only compare specific optimiser instances, but typically *multiple optimiser instantiations* exist and are often not clearly distinguished.
3. We cannot compare optimisers independent of their *hyperparameters*.
4. Optimiser performance is often *problem specific*.

Notably attempts include MLCommons AlgoPerf #cite(<dahl2023benchmarking>) and recent efforts focusing on language models #cite(<semenov2025benchmarking>) #cite(<wen2025fantastic>).

== So we just benchmark all optimisers?

// #grid(
//   columns: (1fr, 1fr),
//   gutter: 1em,
//   [
//     - *There are way too many!*
//       - In 2021, the list on the right was just a subset of all optimisers
//     - Luckily, the *core mechanisms behind most optimisers are the same*
//     - We will focus on the these building blocks, which will enable you to understand the state of the art methods
//   ],
  
// )

#place(right)[
  #figure(
    image("../figures/optimiser_list.png", width: 50%, height: 80%),
    caption: [Subset of optimisers in 2021 #cite(<schmidt2021descending>).],
  )
]

#block(width: 50%)[
  - *There are way too many!*
      - In 2021, the list on the right was just a subset of all optimisers
  - Luckily, the *core mechanisms behind most optimisers are the same*
  - We will focus on the these building blocks, which will enable you to understand the state of the art methods
]

== Setup

Neural network training is typically modelled as *expected
risk minimisation*
$
  min_(theta in RR^d) quad underbrace(EE_(cal(B) ~ cal(D)) [ ell_(cal(B))(theta) ], cal(L)(theta)),
$
where
- $theta$ are the neural network parameters#footnote[Because the methods considered here operate layer-wise,
we let $theta$ denote the (flattened) parameters of a single layer.],
- $cal(B)$ is a mini-batch of data sampled from a stationary distribution $cal(D)$,
- $ell_(cal(B)) : RR^d -> RR$ is a loss function evaluated on batch $cal(B)$.

// This objective is optimised using iterative algorithms with the stochastic gradient 
// $
// g_t = nabla_theta ell_(cal(B)_t)(theta_t)
// $
// at iteration $t$.

We use *iterative algorithms* to optimise this objective.

=== Elements of modern optimisation algorithms

I. Steepest descent

II. Preconditioning

III. Momentum


== I. Steepest descent
// Euclidean and non-euclidean?

- At iterate $theta_t$, approximate the objective using a
  first-order Taylor expansion:
  $
    cal(L)(theta) approx cal(L)(theta_t) + <nabla cal(L)(theta_t), theta - theta_t>.
  $

- Choose the next iterate by minimising this local linear model subject to an update norm penalty:
  $
    theta_(t+1)
    = arg min_theta [ <nabla cal(L)(theta_t), theta - theta_t> + 1 / (2 eta) norm(theta - theta_t)^2 ].
  $
  - Norm penalty defines trust region and should reflect the geometry of our linear model's error; update rule dependents on the choice of the norm
  - Equivalent to minimising upper bound assuming $L$-smoothness with $L=eta^(-1)$

*$ell_2$ norm: Gradient descent (GD)*

Minimised by gradient descent $theta_(t+1) = theta_t - eta nabla cal(L)(theta_t)$.


*$ell_infinity$ norm: SignGD*

#let sign = math.op("sign")

The update becomes $theta_(t+1) = theta_t - eta ||nabla cal(L)(theta_t)||_1 sign(nabla cal(L)(theta_t))$.

- Instead of treating all parameters as vectors, we can maintain the matrix structure of linear layers and choose a matrix norm instead.

// - Linearise the objective at current weight matrix $W_t in RR^(m times n)$:
//   $
//     cal(L)(W) approx cal(L)(W_t) + <nabla cal(L)(W_t), W - W_t>,
//   $
  
- Compute the steepest descent update in an arbitrary matrix norm $norm(·)$:
  $
    W_(t+1) = arg min_W [ <nabla cal(L)(W_t), W - W_t> + 1 / (2 eta) norm(W - W_t)^2 ].
  $
  where $<A, B> = tr(A^T B)$ is the Frobenius inner product.

*$S_infinity$ norm: SpectralGD*

#let Diag = math.op("Diag")

For the Schatten-$infinity$ ($S_infinity$) norm, also called spectral norm, we have
$
  W_(t+1) = W_t - eta ||sigma_t||_1 underbrace(U_t V_t^T, "matrix sign"),
$
where $nabla cal(L)(W_t) = U_t Sigma_t V_t^T$ with $Sigma_t = Diag(sigma_t)$ is the reduced SVD.

*Why these norms?*

They should reflect the _strcuture_ of neural networks layers. Matrix norms can be motivated by operator norms and we can extend to the multi-layer setting by constructing a modular norm #cite(<bernstein2025modular>).

The unique advantages of the $ell_infinity$ and $S_infinity$ geometry in deep learning are still actively researched #cite(<balles2020geometry>) #cite(<davis2025spectral>).

*Stochastic setting*

Replacing $nabla cal(L)(theta_t)$ with the stochastic gradient $g_t = nabla ell_(cal(B)_t)(theta_t)$ (or $G_t$ for matrix parameters) as an unbiased estimate yields *stochastic* variants of each update.

== II. Preconditioning
// Newton's method
// GGN
// NGD
// Full matrix AdaGrad
// Kronecker-factored approximations (K-FAC + Shampoo)
// Diagonal methods

Instead of changing the norm to model the error of our linear model, we can explicitly consider second order information.

=== Quadratic model

- Second-order Taylor expansion around $theta_t$:
  $
    cal(L)(theta) approx cal(L)(theta_t) + < nabla cal(L)(theta_t), theta - theta_t> + 1 / 2 (theta - theta_t)^T nabla^2 cal(L)(theta_t) (theta - theta_t).
  $

- Minimise the quadratic model:
  $
    theta_(t+1) = arg min_theta [ <nabla cal(L)(theta_t), theta - theta_t> + 1 / 2 (theta - theta_t)^T nabla^2 cal(L)(theta_t) (theta - theta_t) ].
  $

- If the Hessian $nabla^2 cal(L)(theta_t)$ is invertible, we get *Newton's method*
  $
    theta_(t+1)
    = theta_t - nabla^2 cal(L)(theta_t)^(-1)  nabla cal(L)(theta_t).
  $

// - We call the inverse matrix in front of the gradient *preconditioner*
- The Hessian might have negative eigenvalues (non-convexity), but we can approximate it with the *generalised Gauss-Newton matrix* #cite(<martens2014new>), which is guaranteed to be positive semi-definite

- For many common loss functions this corresponds to *natural gradient descent* #cite(<amari1998natural>), which uses the Fisher information matrix

- Alternatively, we consider a preconditioner that is not defined at a single iterate $theta_t$ like all the choices above, but across the *parameter trajectory*

- The canonical preconditioner in this setting is used in full-matrix AdaGrad:
  $
    A_t = sum_(i=1)^t g_i g_i^T
  $

- The resulting update is $theta_(t+1) = theta_t - eta A_t^(-1/2) g_t$, where we now use an inverse matrix square root instead of just the inverse

- This is originally motivated by a convex, but potentially non-smooth online learning setting #cite(<duchi2011adaptive>)

- In principle, we can also use other forms of accumulation, e.g. an exponential moving average (EMA)


*But:* all these matrices are squared in the number of parameters!

#sym.arrow too expensive to compute, store, and invert

=== Approximations

#image("../figures/matrix_approx.svg")


== III. Momentum
// Polyak and Nesterov momentum
// Primal and dual averaging
- We can use parameter iterate (primal) and gradient (dual) averaging to speed up convergence


== Putting things together: Adam

-


// Duality of preconditioning and non-euclidean GD

== Adam and SignGD
// Let's put things together -> Adam
// Default optimiser!
// Many alternatives, unclear why this should be best.
// what is it really doing?
// focus on language models. not noise, but sign might be

// connect to SignGD with beta1=beta2=0
// decomposition in scaled Signum
-

== Shampoo and SpectralGD
// Remember when I said traditional optimisation doesn't leverage structure?

// let's use matrix structure!

-

== So which one should I use?
// Trade-offs: implementation, computational, memory, communication overhead and track record

// Other reasons to care about optimiser: generalisation, quantisation, continual learning, etc, but we focus on optimisation.

- 

== Other components

=== Weight decay
// traditionally thought of as regularisation (l2)
// in practice, decoupled weight decay dominates (AdamW)
// implementation note: most implementations don't fully decouple from learning rate
// weight decay likely multiple roles:
// ...

=== Adapting hyperparameters
// instead of just considering a few global, constant hyperparameters, we can
// 1) schedule them across iterations. most common is a learning rate schedule, but other hyperparamters (beta1/2, weight decay) might also benefit!
// 2) allow different hyperparamters for different subsets of the model

// However, as we scale training iterations, batch and model size, do we have to retune hyperparamters? Isn't this prohibitively expensive?
