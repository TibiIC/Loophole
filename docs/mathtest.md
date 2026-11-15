# Math rendering test (scratch, to be deleted)

Inline: let $q = (t, R)$ be tube B's pose; $p_{\max}$ = `penetration_threshold`; $p_{\text{end}} = 0.472 \le 1$;
the best held cost-to-go is $c^{*}$ and the start is $c_0$; $x_b(t - \Delta t)$; $\varepsilon$ = 16.6 mm.

* list item with math: $N_{\text{warn}}(t) > N_{\text{warn}}(t - \Delta t)$
* list item with underscores outside math: `max_substep_displacement` and snake_case_word

Display, single line:

$$
V > 0 \;\Longrightarrow\; S = 0 .
$$

Display with fractions, min and max:

$$
S_{\text{unsolved}} = \kappa\, p, \qquad
p = \mathrm{clip}\!\left(1 - \frac{c^{*}}{c_0},\ 0,\ 1\right), \qquad
c^{*} = \min\!\Bigl(c_0,\ \min_{t}\ \max\bigl(c_{t-2},\, c_{t-1},\, c_t\bigr)\Bigr),
$$

Display with subscripted operators and sets:

$$
c(q) = \min_{n \,\in\, N_{10}(q),\ \overline{q\,n} \subset \mathcal C_{\text{free}}} \bigl[\, d(q, n) + \mathrm{ctg}(n) \,\bigr] ,
$$

Cases with row breaks:

$$
S =
\begin{cases}
0 & V > 0 \\
1 - \alpha\,\min(a, T)/T & \text{solved} \\
\kappa\, p & \text{otherwise}
\end{cases}
\qquad \alpha = 0.5,\ \kappa = 0.3,
$$

Sum and norms:

$$
\sum_{t=0}^{T-1} \gamma^{t}\, F(s_t, s_{t+1}) = \lambda\bigl(\gamma^{T}\,\bar\Phi(s_T) - \Phi(s_0)\bigr), \qquad
\delta\bigl(x(t),\, c_k\bigr) = \max_i \lVert x_i(t) - c_{k,i} \rVert < \varepsilon
$$

Things that must stay literal: `$HOME` in inline code, and in a block:

```bash
echo "$RUNS runs, $N_CONCURRENT at a time"
```

| table cell with math | value |
|---|---|
| $\alpha$ | 0.5 |
| $\kappa$ | 0.3 |
