
#let shortcite(key) = [
  (#cite(key, form: "author") #cite(key, form: "year"))#cite(key)
]

#let loss = $cal(L)$
#let dataset = $cal(D)$

#let argmin = math.op("arg min", limits: true)
#let argmax = math.op("arg max", limits: true)

#let Span = $"Span"$
#let Var = math.op("Var")
#let Cov = math.op("Cov")
#let iid = [i.i.d.]


#let palette1 = rgb(204, 57, 42)
#let palette2 = rgb(79, 155, 143)
#let palette3 = rgb(44, 97, 194)
#let palette4 = rgb(217, 116, 89)
#let palette5 = rgb(228, 197, 119)
#let palette6 = rgb(63, 100, 67)


#let mono(body) = {
  text(font: "New Computer Modern Mono")[#body]
}
#let Rd(body) = {
  $RR^(d_#mono[#body])$
}
#let rightbar(body, subscript) = {
  $lr(#body |)_subscript$
}

#let sfrac(x, y, size: 0.8em, baseline: 0.2em, shrink: 0.1em, endshrink: 0.05em) = [
  #text(size: size, baseline: -baseline, [#x])
  #h(-shrink) / #h(-shrink)
  #text(size: size, [#y], baseline: baseline)
  #h(-endshrink)
]

