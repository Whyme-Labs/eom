# Harmonic mean p-value

The harmonic mean p-value (HMP) is a statistical technique for addressing the multiple comparisons problem that controls the strong-sense family-wise error rate). It improves on the power of Bonferroni correction by performing combined tests, i.e. by testing whether groups of p-values are statistically significant, like Fisher's method. However, similar to other extensions of Fisher's method, it avoids the restrictive assumption that the p-values are independent, unlike Fisher's method. This is because the power of the HMP to detect significant groups of hypotheses is greater than the power of BH to detect significant individual hypotheses.

When \alpha is small (e.g. \alpha), the following multilevel test based on direct interpretation of the HMP controls the strong-sense family-wise error rate at level approximately \alpha:

# Define the HMP of any subset \mathcal{R} of the L p-values to be
\overset{\circ}{p}_\mathcal{R} = \frac{\sum_{i\in\mathcal{R}} w_{i}}{\sum_{i\in\mathcal{R}} w_{i}/p_{i}}.

# Reject the null hypothesis that none of the p-values in subset \mathcal{R} are significant if \overset{\circ}{p}_\mathcal{R}\leq\alpha\,w_\mathcal{R}, where w_\mathcal{R}=\sum_{i\in\mathcal{R}}w_i. (Recall that, by definition, \sum_{i=1}^L w_i=1.)

An asymptotically exact version of the above replaces \overset{\circ}{p}_\mathcal{R}in step 2 with p_{\overset{\circ}{p}_\mathcal{R}} = \max\left\{\overset{\circ}{p}_\mathcal{R}, w_{\mathcal{R}} \int_{w_{\mathcal{R}}/\overset{\circ}{p}_\mathcal{R}}^\infty f_\textrm{Landau}\left(x\,|\,\log L +0.874,\frac{\pi}{2}\right) \mathrm{d} x\right\}, where L gives the number of p-values, not just those in subset \mathcal{R}.

Since direct interpretation of the HMP is faster, a two-pass procedure may be used to identify subsets of p-values that are likely to be significant using direct interpretation, subject to confirmation using the asymptotically exact formula.

The HMP has a range of properties that arise from generalized central limit theorem.

If the distributions of the p-values under the alternative hypotheses follow Beta distributions with parameters \left(0, a form considered by Sellke, Bayarri and Berger, then the inverse proportionality between the model-averaged Bayes factor and the HMP can be formalized as\overline{\textrm{BF}}=\sum_{i=1}^L \mu_i\,\textrm{BF}_i=\sum_{i=1}^L \mu_i\,\xi_i\,p_i^{\xi_i-1}\approx\bar\xi\sum_{i=1}^L w_i\,p_i^{-1}=\frac{\bar\xi}{\overset{\circ}{p}}, where

*\mu_i is the prior probability of alternative hypothesis i, such that \sum_{i=1}^L\mu_i=1,
*\xi_i/(1+\xi_i) is the expected value of p_i under alternative hypothesis i,
*w_i=u_i/\bar\xi is the weight attributed to p-value i,
*u_i = \left(\mu_i\,\xi_i\right)^{1/(1-\xi_i)} incorporates the prior model probabilities and powers into the weights, and
*\bar\xi = \sum_{i=1}^L u_i normalizes the weights.

The approximation works best for well-powered tests (\xi_i\ll 1).

For likelihood ratio tests with exactly two degrees of freedom, Wilks' theorem implies that p_i=1/R_i, where R_i is the maximized likelihood ratio in favour of alternative hypothesis i, and therefore \overset{\circ}{p}=1/\bar{R}, where \bar{R} is the weighted mean maximized likelihood ratio, using weights w_1,\dots,w_L. Since R_i is an upper bound on the Bayes factor, \textrm{BF}_i, then 1/\overset{\circ}{p} is an upper bound on the model-averaged Bayes factor:\overline{\textrm{BF}}\leq\frac{1}{\overset{\circ}{p}}.While the equivalence holds only for two degrees of freedom, the relationship between \overset{\circ}{p} and \bar{R}, and therefore \overline{\textrm{BF}}, behaves similarly for other degrees of freedom.

*Extensions of Fisher's method

Category:Multiple comparisons
