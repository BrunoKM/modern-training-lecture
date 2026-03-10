#import "../setup.typ": *

= I. How to Choose the Optimiser

== Traditional optimisation
// Convex -> optimality gap
// Non-convex -> gradient norm bound
// online/adverserial -> regret bound

// influences on rate: stochasticity and smoothness
// worst case guarantees are too pessimistic, e.g. SGD is optimal but clearly we can do better in practice
// limitation: these assumptions don't use strcuture inherint in deep learning problems, e.g. compositional, hierachicial nature of architectures and data distribution

- Traditionally, we can derive *convergence guarantees* by making assumptions
- However, it is *questionable* to what extent common *assumptions* (e.g., convexity, smoothness) are justified in modern deep learning settings #cite(<tran2025reevaluating>)
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
- $theta$ are the neural network parameters#footnote[Because the methods considered here operate layer-wise, we let $theta$ denote the (flattened) parameters of a single layer.],
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

// *Why these norms?*

// They should reflect the _strcuture_ of neural networks layers. Matrix norms can be motivated by operator norms and we can extend to the multi-layer setting by constructing a modular norm #cite(<bernstein2025modular>).

// The unique advantages of the $ell_infinity$ and $S_infinity$ geometry in deep learning are still actively researched #cite(<balles2020geometry>) #cite(<davis2025spectral>).

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
- We can use parameter iterate (primal) and gradient (dual) *averaging* to speed up convergence; the latter is also called momentum

- While there are many formulations of averaging, we focus on a simple form of momentum, initialised with $m_0 = 0$:

*Polyak momentum*
  $
    m_t = beta_1 m_(t-1) + g_t
  $

*EMA*
    $
    m_t = beta_1 m_(t-1) + (1 - beta_1) g_t
    $
    
- Then we update the parameters using the averaged gradient (momentum)
  $
  theta_(t+1) = theta_t - eta m_t
  $
  - Note that the update with the EMA and $tilde(eta) = eta / (1-beta_1)$ recovers Polyak momentum

- See Table 1 and 3 in #shortcite(<defazio2026smoothing>) for a more general overview


== Putting things together: Adam

Using these building blocks, we can assemble Adam, the *de facto standard optimiser in deep learning*

=== I. Steepest descent

#sym.arrow We simply use $g_t$

=== II. Preconditioner

#sym.arrow We use the diagonal of an AdaGrad-like preconditioner with an EMA initialised with $v_0 = 0$ instead of sum accumulation
$
  v_t = beta_2 v_(t-1) + (1-beta_2) g_t^2
$

=== III. Momentum

#sym.arrow We use the EMA
$
  m_t = beta_1 m_(t-1) + (1 - beta_1) g_t
$

The resulting update is $theta_(t+1) = theta_t - eta m_t / (sqrt(v_t) + epsilon)$, where $epsilon > 0$.#footnote[Ignoring bias corrections.] #cite(<kingma2014adam>)

== Putting things together: Shampoo

- What about a *non-diagonal preconditioner*?

  #sym.arrow block-diagonal, Kronecker-factored

- *Shampoo* #cite(<gupta2018shampoo>) #cite(<shi2023distributed>)
  $
    L_t = beta_2 L_(t-1) + (1-beta_2) G G^T\
    R_t = beta_2 R_(t-1) + (1-beta_2) G^T G \
    W_(t+1) = W_t - eta (L_t + epsilon I)^(-p) M_t (R_t + epsilon I)^(-p)
  $
  - update equivalent to $eta ((R_t + epsilon I) times.o (L_t + epsilon I))^(-p) m_t$
  - only difference to Adam is the preconditioner

- *Won the MLCommons AlgoPerf training algorithms competition* (external tuning track), arguably the most rigorous non-problem-specific benchmark #cite(<kasimbeg2025accelerating>)

#figure(
  image("../figures/algoperf.png"),
  caption: [#shortcite(<kasimbeg2025accelerating>)]
)


== But why?

- You might have noticed that these choices seem *arbitrary*

  - For example, we could compute the momentum over the preconditioned gradient (LaProp) instead of preconditioning the momentum (Adam)

- That is because they are!

- However, it clearly works, so there has to be some explanation

- We will provide one somewhat opinionated perspective here



== Adam and SignGD
// Let's put things together -> Adam
// Default optimiser!
// Many alternatives, unclear why this should be best.
// what is it really doing?
// focus on language models. not noise, but sign might be

// Duality of preconditioning and non-euclidean GD
// connect to SignGD with beta1=beta2=0
// decomposition in scaled Signum
- In the *extreme setting* $beta_1 = beta_2 = epsilon = 0$, the update becomes (assuming all elements of $g_t$ are non-zero)
  $
    theta_(t+1) = theta_t - eta g_t / sqrt(g_t^2) = theta_t - eta g_t / (|g_t|) = theta_t - eta sign(g_t)
  $
  - we *recover SignGD* (up to a scalar scaling)! #cite(<bernstein2018signsgd>)

- In *general*, setting $epsilon=0$ for simplicity, we can decompose the update as
  $
    m_t / sqrt(v_t) = (|m_t|) / sqrt(v_t) sign(m_t) = 1 / sqrt(1 + (v_t - m_t^2) / m_t^2) quad underbrace(sign(m_t), "Signum")
  $
  - we get *element-wise scaled Signum*, where the scaling lies in $(0, 1]$ #cite(<balles2017dissecting>)
  - this has originally been interpreted as _variance adaptation_

#sym.arrow These results show that Adam is connected to *$ell_infinity$ geometry*

== Adam and Signum

But is this connection meaningful?

- SignGD, Signum, and Adam benefit from larger batch sizes and Signum is closer to Adam's performance than GD with momentum in the full-batch setting #cite(<kunstner2023noise>)

- In a modern language modelling setting, Signum bridges most of the gap between SGD with momentum and Adam #cite(<orvieto2025search>)

#figure(
  image("../figures/adam_sgd_gap.png"),
  // caption: [],
)

- However, the element-wise scaling of Signum is still significant!


#figure(
  align(center, image("../figures/adam_signum_gap.png", height: 90%)),
// caption: []
)

== Shampoo and SpectralGD
// Remember when I said traditional optimisation doesn't leverage structure?
// let's use matrix structure!

- In the *extreme setting* $beta_1 = beta_2 = epsilon=0$, $p=1/4$, and assuming $G_t$ has full rank:
  $
  ( G_t G_t^T )^(-1/4) G_t (G_t^T G_t)^(-1/4) &= (U Sigma^2 U^T)^(-1/4) U Sigma V^T (V Sigma^2 V^T)^(-1/4) \
  &= U Sigma^(-1/2) U^T U Sigma V^T V Sigma^(-1/2) V^T \
  &= U V^T,
  $
  where $G_t = U Sigma V^T$ is the reduced SVD.
  - we recover *SpectralGD* (up to a scalar scaling) #cite(<bernstein2024oldoptimizernewnorm>)

- In *general*, for $beta_1 eq.not 0$ and $beta_2 eq.not 0$, we can decompose the update as
  $
  L_t^(-p) M_t R_t^(-p)= L_t^(-p) (M_t M_t^T)^(1/4) quad underbrace(U_t V_t^T, "Muon") quad (M_t^T M_t)^(1/4) R_t^(-p),
  $
  where $M_t = U_t Sigma_t V^T$ is the reduced SVD.
  - we get *left- and right-adapted Muon* #cite(<eschenhagen2026clarifying>)

#sym.arrow These results show that Shampoo is connected to *$S_infinity$ geometry*

== Notes on Muon

- We called the matrix sign of the momentum $M_t$ Muon

- But: *Muon* = #[*M*]oment#[*u*]m #[*o*]rthogonalized by #[*N*]ewton-Schulz

  - The name implies a specific efficient numerical method for computing the matrix sign!
  
  - We overload this terminology here

- Muon has become increasingly popular for LLM training #cite(<liu2025muon>)

  - Up to 1 trillion parameter models (32B active)

  - When people say they use Muon, they typically mean the use Muon for the *hidden weight matrices* and Adam for all other parameters

#figure(
  image("../figures/muon_is_scalable.png"),
  caption: [#shortcite(<liu2025muon>)]
)


== Shampoo and Muon


#figure(
  image("../figures/figure1.png"),
  caption: [#shortcite(<eschenhagen2026clarifying>)]
)

#figure(
  image("../figures/shampoo_muon_table1.png"),
  caption: [#shortcite(<eschenhagen2026clarifying>)]
)


== Open questions

*1. Adaptation in Adam and Shampoo*

  - Adam's and Shampoo's preconditioner relax the strict constraints on updates that are enforced by SignGD and SpectralGD

  - Both adapt to stochasticity and the parameter trajectory

  - How to combine preconditioning (adaptivity) with momentum?

#figure(
  align(center, image("../figures/full_batch.png", height: 65%)),
// caption: []
)

== Open questions

*2. Why $ell_infinity$ and $S_infinity$ geometry?*

  - modular duality #cite(<bernstein2025modular>)

  - robustness to heavy-tailed class imbalance #cite(<kunstner2024heavytailed>)

  - Benefits of SignGD depend on the Hessian structure #cite(<balles2020geometry>)

  - structure of gradients and activations benefits SpectralGD #cite(<davis2025spectral>)

== So what optimiser should I use?
// Trade-offs: implementation, computational, memory, communication overhead and track record

// Other reasons to care about optimiser: generalisation, quantisation, continual learning, etc, but we focus on optimisation.
*
Signum #sym.arrow Adam #sym.arrow Muon #sym.arrow Shampoo*

Each arrow roughly means

- _increased overhead_ (implementation, computational, memory, communication)

- _faster convergence_ (data efficiency)

- more modelling choices necessary (Muon and, assuming the perspective presented here, Shampoo are_ not parameter shape-agnostic_)

*Some heuristics:*

- if you don't want to think, use Adam

- if you have time to tailor the optimiser to the problem, use Muon or Shampoo

  - since LLM training is highly specialised and costly, it seems worth it to use a more sophisticated optimiser (downside is less track-record)

== Other components

=== Weight decay
// traditionally thought of as regularisation (l2)
// in practice, decoupled weight decay dominates (AdamW)
// implementation note: most implementations don't fully decouple from learning rate
// weight decay likely multiple roles:
// ...
- Traditionally, $ell_2$ regularisation is used to avoid overfitting

- In the scaling era of deep learning, we typically use _decoupled_ weight decay

  - Surprisingly, the *role of weight decay* is complex and *not well-understood*
  

=== Adapting hyperparameters
// instead of just considering a few global, constant hyperparameters, we can
// 1) schedule them across iterations. most common is a learning rate schedule, but other hyperparamters (beta1/2, weight decay) might also benefit!
// 2) allow different hyperparamters for different subsets of the model

// However, as we scale training iterations, batch and model size, do we have to retune hyperparamters? Isn't this prohibitively expensive?
- We can *schedule hyperparameters* across iterations

  - Learning rate schedules are always used (warumup + linear, cosine, or stable-decay)

  - In principle, scheduling other hyperparameters like $beta_1, beta_2$, and weight decay might also be beneficial

- We can tune per-layer (type) hyperparameters

== Additional resources

#show link: set text(orange)

- #link("https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2022/")[Neural Net Training Dynamics] by #link("https://www.cs.toronto.edu/~rgrosse")[Roger Grosse]

- #link("https://institute-tue.ellis.eu/en/lecture-deep-optimization")[Nonconvex Optimization for Deep Learning] by #link("http://orvi.altervista.org/")[Antonio Orvieto]

- #link("https://damek.github.io/STAT-4830/")[Numerical optimization for data science and machine learning] by #link("https://damek.github.io/")[Damek Davis]
