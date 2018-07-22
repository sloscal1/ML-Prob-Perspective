""" Worked examples and exercises from Chapter 3.

"""
import math
import random

import numpy as np
import pandas as pd
from scipy.stats import beta


class Concept:
    """ This is a concept or hypothesis from the Numbers Game.

    Attributes:
        name (str): The common name of the concept.
        extension (list int): The values in the event space this concept describes.
        prior (float): The prior probability of this concept being selected.
    """
    def __init__(self, name, extension=[], prior=-1):
        self.name = name
        self.extension = extension
        self.prior = prior

    def likelihood(self, n_samples):
        r"""
        Compute the :math:`p(D|h) = \frac{p(D,h)}{p(h)}`.

        Args:
            n_samples (int): the number of samples collected from the target concept.

        Returns:
            list float: the likelihood of the data given the hypothesis.
                This comes from the strong sampling assumption.
        """
        return 1.0/(len(self.extension)**n_samples)

    def __repr__(self):
        return f"Concept(name={self.name},extension={self.extension},prior={self.prior})"

    def __eq__(self, other):
        return self.name == other.name


class NumberGame(object):
    """ The game as described in Chapter 3 but used by Joshua Tenenbaum in his PhD Thesis.

    The game demonstrates that it is possible to select a correct hypothesis using only
    positive examples from the target.

    Note: It is critical that all the candidate concepts are distinct. This might seem obvious,
    but there are often many ways to describe the same concept in language. For example, consider
    ``multiples of 10`` and ``ends in 0`` for the space :math:`[1, 100]`. These are clearly different
    ideas, but they have the same extension in the space. This means that if one is the target, the
    other is equally likely. This causes the optimization to converge to 0.5 instead of 1.0, playing
    havoc with the program.

    Attributes:
        concepts (list Concept): The possible concepts in the game.
        active_concept: (Concept): The current target of the game, selected at random from ``concepts``.


    """
    def __init__(self, *, seed=None):
        if seed:
            np.random.seed = seed
            random.seed(seed)
        self.max_val = 100
        primes = [2]
        for num in range(3, self.max_val+1, 2):
            if all([num % prime for prime in primes]):
                primes.append(num)
        self.concepts = []
        self.concepts.extend(
            [
                Concept("odd", [num for num in range(1, self.max_val+1, 2)], 0.15),
                Concept("squares", [num*num for num in range(1, 11)]),
                Concept("primes", primes, 0.05),
                Concept("all", list(range(1, self.max_val+1))),
            ]
        )
        for val in range(2, 11):
            self.concepts.append(Concept(f"mult of {val}", [num for num in range(val, self.max_val+1, val)]))
            self.concepts.append(Concept(f"ends in {val}", [num for num in range(val, self.max_val+1, 10)]))
            self.concepts.append(Concept(f"power of {val}", [val**num for num in range(1, int(math.floor(math.log(self.max_val, val)))+1)]))
        # Find the concept mult of 2 and replace it with even
        pos = self.concepts.index(Concept("mult of 2"))
        self.concepts[pos].name = "even"
        self.concepts[pos].prior = 0.15
        self.concepts.append(
            Concept(
                "power of 2 + (37)",
                [
                    val for val in np.add(
                        self.concepts[self.concepts.index(Concept("power of 2"))].extension,
                        37)
                    if val <= 100
                ],
                0.0005
            )
        )
        self.concepts.append(
            Concept(
                "power of 2 - (37)",
                [
                    val for val in np.add(
                        self.concepts[self.concepts.index(Concept("power of 2"))].extension,
                        -37)
                    if val > 0
                ],
                0.0005
            )
        )
        self.concepts.remove(Concept("ends in 10"))
        self.active_concept = None
        other_priors = 1.0/(len(self.concepts) - len([concept for concept in self.concepts if concept.prior != -1]))
        for concept in self.concepts:
            if concept.prior == -1:
                concept.prior = other_priors
        self.active_concept = random.choice(list(self.concepts))

    def sample_from_concept(self):
        """ Generate a single sample from the ``active_concept``.

        Returns:
            int: a number from the active concept.

        """
        return random.choice(self.active_concept.extension)

    def posterior(self, samples):
        r""" Generate the full posterior probability of all ``concepts``.

        Computes the posterior :math:`p(C|\mathcal{D}) = \frac{p(\mathcal{D}|h)p(h)}{p(\mathcal{D})}`,
        substituting all possible concepts for :math:`h`.

        Args:
            samples (list: int): All samples generated by the target concept so far.

        Returns:
            list float: the posterior probability of all ``concepts``.
        """
        unique_samps = set(samples)
        denominator = 0
        posteriors = []
        n_samps = len(samples)
        for concept in self.concepts:
            num = 0
            if unique_samps.issubset(set(concept.extension)):
                num = concept.prior*concept.likelihood(n_samps)
            denominator += num
            posteriors.append(num)
        return np.divide(posteriors, denominator)

    def post_predictive_distribution(self, samples):
        r"""

        What is the probability that any point belongs to the
        target concept given the data we've seen so far?

        .. math::
            p(\tilde{x} \in C|\mathcal{D}) = \Sigma_h p(y=1|\tilde{x},h)p(h|\mathcal{D}).

        Where :math:`\tilde{x}` is a future observation and :math:`y=1` states that the
        observation is consistent with the given concept.

        Args:
            samples (list int): the samples from the target concept we've observed so far.

        Returns:
            list float: the posterior predictive distribution at this time.
        """
        post_pred_dist = []
        posteriors = self.posterior(samples)
        for point in range(1, self.max_val+1):
            post_pred = 0
            for concept, posterior in list(zip(self.concepts, posteriors)):
                if point in concept.extension:
                    post_pred += posterior
            post_pred_dist.append(post_pred)
        return post_pred_dist

    def plugin_distribution(self, samples):
        """
        What is the probability that any point belongs to the target concept
        given that we "plug in" the most likely concept a posteriori?

        Args:
            samples: the samples seen so far.

        Returns:
            List of the posterior probabilities using the plug-in estimator.
        """
        plugin = self.concepts[np.argmax(self.posterior(samples))]
        return [1.0 if x in plugin.extension else 0.0 for x in range(1, self.max_val+1)]


def likelihood_ratio(posteriors):
    temps = sorted(posteriors)
    lr = temps[-1]/temps[-2]
    return lr


class BetaBinomial:
    def __init__(self, alpha_0, alpha_1, seed=1337):
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        random.seed = seed
        self.rate = random.random()

    def sample(self):
        return 0.0 if random.random() > self.rate else 1.0

    def posterior(self, samples, freq=100):
        counts = list((0, 0))
        for sample in samples:
            counts[sample] += 1
        alpha = counts[1]+self.alpha_1-1
        beta = counts[0]+self.alpha_0-1
        posterior = pd.DataFrame(columns=["theta", "prob"])
        posterior["theta"] = np.linspace(0, 1, freq)
        posterior = posterior.assign(prob=lambda x: x.theta**alpha*(1-x.theta)**beta)
        norm = posterior.prob.sum()
        posterior = posterior.assign(prob=lambda x: x.prob/norm)
        posterior = posterior.set_index("theta")
        return posterior


def question_1():
    r"""
    Optimize the log likelihood of :math:`p(\mathcal{D}|\theta) = \theta^{N_1}(1-\theta)^{N_0}`
    to prove :math:`\frac{N_1}{N}`, the MLE of the Bernoulli/binomial model.

    .. math::
        log(p(\mathcal{D}|\theta) &= log(\theta^{N_1}(1-\theta)^{N_0})\\
                                  &= N_1 log(\theta)+ N_0 log(1-\theta)\\
                                  &= N_1 log(\theta)+ (N-N_1)log(1-\theta).

    Now, optimizing for :math:`\theta` by taking the derivative of the above and setting it
    equal to 0.

    .. math::
        \frac{d}{d\theta} [N_1 log(\theta)+ (N-N_1)log(1-\theta)] &= \frac{N_1}{\theta} - \frac{N-N_1}{1-\theta}\\
        0 &= \frac{N_1}{\theta} - \frac{N-N_1}{1-\theta}\\
        N_1(1-\theta) &= (N-N_1)\theta\\
        N_1 - N_1\theta &= N\theta - N_1\theta\\
        N_1 &= N\theta\\
        \frac{N_1}{N} &= \theta\\

    Returns:
        None.
    """
    return None


def question_2():
    r"""
    Show that:

    .. math:: \frac{[(\alpha_1)\cdots(\alpha_1 + N_1 - 1)][(\alpha_0)\cdots(\alpha_0+N_0-1)]}{(\alpha)\cdots(\alpha+N-1)}

    Can be reduced to:

    .. math:: \frac{[\Gamma(\alpha_1+N_1)\Gamma(\alpha_0+N_0)]}{\Gamma(\alpha_1+\alpha_0+N)}\frac{\Gamma(\alpha_1+\alpha_0)}{\Gamma(\alpha_1)\Gamma(\alpha_0)}

    Using :math:`(\alpha-1)! = \Gamma(\alpha)`.

    .. math::
        \frac{[(\alpha_1)\cdots(\alpha_1 + N_1 - 1)][(\alpha_0)\cdots(\alpha_0+N_0-1)]}{(\alpha)\cdots(\alpha+N-1)} &=\\
        \frac{[(\alpha_1)\cdots(\alpha_1 + N_1 - 1)][(\alpha_0)\cdots(\alpha_0+N_0-1)]}{(\alpha)\cdots(\alpha+N-1)}\cdot\frac{(\alpha-1)!}{(\alpha-1)!} &=\\
        \frac{[(\alpha_1)\cdots(\alpha_1 + N_1 - 1)][(\alpha_0)\cdots(\alpha_0+N_0-1)]}{(\alpha+N-1)!}\cdot\frac{(\alpha-1)!}{1} &=~,~&\textrm{Def. of factorial}\\
        \frac{[(\alpha_1)\cdots(\alpha_1 + N_1 - 1)][(\alpha_0)\cdots(\alpha_0+N_0-1)]}{(\alpha+N-1)!}\cdot\frac{(\alpha-1)!(\alpha_1-1)!(\alpha_0-1)!}{(\alpha_1-1)!(\alpha_0-1)!} &=\\
        \frac{(\alpha_1 + N_1 - 1)!(\alpha_0+N_0-1)!}{(\alpha+N-1)!}\cdot\frac{(\alpha-1)!}{(\alpha_1-1)!(\alpha_0-1)!} &=~,~&\textrm{Def. of factorial}\\
        \frac{\Gamma(\alpha_1 + N_1)\Gamma(\alpha_0+N_0)}{\Gamma(\alpha+N)}\cdot\frac{\Gamma(\alpha)}{\Gamma(\alpha_1)\Gamma(\alpha_0)} &=~,~&\textrm{By the given}\\
        \frac{\Gamma(\alpha_1 + N_1)\Gamma(\alpha_0+N_0)}{\Gamma(\alpha_1+\alpha_0+N)}\cdot\frac{\Gamma(\alpha_1+\alpha_0)}{\Gamma(\alpha_1)\Gamma(\alpha_0)} &=~,~&\textrm{Def.}~\alpha = \alpha_1+\alpha_0

    So we see that even without appealing to the Beta distribution, we can by sheer counts of the probability of the
    data occurring arrive at the marginal likelihood for the Beta-Bernoulli model.

    Returns:
        None.
    """
    return None


def question_3():
    r""" Posterior predictive for Beta-Binomial model

    Prove that :math:`p(x|n, \mathcal{D})=\frac{B(x+\alpha_1',n-x+\alpha_0')}{B(\alpha_1',\alpha_0')}\binom{n}{x}`
    reduces to :math:`p(\tilde{x}=1|\mathcal{D})=\frac{\alpha_1'}{\alpha_1'+\alpha_0'}` when :math:`n=1`.

    .. math::
       \frac{B(x+\alpha_1',n-x+\alpha_0')}{B(\alpha_1',\alpha_0')}\binom{n}{x} &=\\
       \frac{B(x+\alpha_1',1-x+\alpha_0')}{B(\alpha_1',\alpha_0')}\binom{1}{x} &=~&,~\textrm{Given}\\
       \frac{B(1+\alpha_1',1-1+\alpha_0')}{B(\alpha_1',\alpha_0')}\cdot 1 &=~&,~\textrm{Given}~x=1\\
       \frac{\Gamma(1+\alpha_1')\Gamma(\alpha_0')\Gamma(\alpha_1'+\alpha_0')}{\Gamma(1+\alpha_1'+\alpha_0')\Gamma(\alpha_1')\Gamma(\alpha_0')} &=~&,~\textrm{Def. of}~Beta\\
       \frac{\alpha_1'\Gamma(\alpha_1')\Gamma(\alpha_1'+\alpha_0')}{(\alpha_1'+\alpha_0')\Gamma(\alpha_1'+\alpha_0')\Gamma(\alpha_1')} &=~&,~\Gamma(a+1)=a\Gamma(a)\\
       \frac{\alpha_1'}{\alpha_1'+\alpha_0'}.

    So we can see that after a single trial, the posterior predictive of getting a 1 in that trial is simply the rate
    of getting a 1 as given by the prior, which makes sense because we haven't yet observed any data.
    Returns:
        None.
    """
    return None

def question_4():
    r""" Simple mixture distribution.

    Let's say we tossed a fair coin 5 times and know that < 3 heads appeared. Compute the posterior up to
    normalization constant..

    .. math::
        p(X < 3 | \theta) &= p(X=0 | \theta) + p(X=1 | \theta) + p(X=2|\theta),~&\textrm{Union of mutually exclusive events.}\\
                          &\propto B(\theta|1,1)Bin(0|\theta,5) + B(\theta|1,1)Bin(1|\theta,5) + B(\theta|1,1)Bin(2|\theta,5),~&\textrm{Bayes law}\\
                          &\propto B(\theta|1,6) + B(\theta|2,5) + B(\theta|3,4),~&\textrm{Conjugate prior}.

    Returns:
        None.
    """
    return None


def question_5():
    r""" Uninformative prior for log-odds ratio.

    Let :math:`\phi = \textrm{logit}(\theta) = log\frac{\theta}{1-\theta}`.
    Show that if :math:`p(\phi) \propto 1`, then :math:`p(\theta) \propto Beta(\theta|0, 0)`.

    .. math::
        p(\phi) &= p(\theta)\left\vert\frac{d \theta}{d \phi}\right\vert,~&\textrm{Change of variables}\\
                &= log\frac{\theta}{1-\theta}\left\vert\frac{d\theta}{d\phi}\right\vert,\\
                &= log\theta - log(1-\theta)\left\vert\frac{d\theta}{d\phi}\right\vert,\\
                &= \frac{1}{\theta} + \frac{1}{1-\theta},\\
                &= \frac{\theta + 1 - \theta}{\theta(1-\theta)},\\
                &= \frac{1}{\theta(1-\theta)},\\
                &= \theta^{-1}(1-\theta)^{-1},\\
                &= B(\theta|0, 0),~&\textrm{Def. of Beta}.

    Returns:
        None.
    """
    return None


def question_6():
    r""" MLE for the Poisson distribution.
    :math:`Poi(x|\lambda) = e^{-\lambda}\frac{\lambda^x}{x!}`. Derive the MLE.

    .. math::
        p(\lambda|x_1,\ldots,x_n) &= e^{-\lambda}\frac{\lambda^{x_1}}{x_1!}\cdots e^{-\lambda}\frac{\lambda^{x_n}}{x_n!},~&X~\sim~Poi(\lambda)\\
            &= e^{-n\lambda}\frac{\lambda^{x_1+\cdots+x_n}}{\prod_i^n x_i!}.

    Set take the derivative and set it equal to 0 to find the maximum:

    .. math::
        \frac{d}{d\lambda}p(\lambda|x_1,\ldots,x_n) &= -ne^{-n\lambda}\frac{\lambda^{x_1+\cdots+x_n}}{\prod_i^n x_i!} + e^{-n\lambda}\frac{(x_1+\cdots+x_n)\lambda^{x_1+\cdots+x_n-1}}{\prod_i^n x_i!},\\
            ne^{-n\lambda}\frac{\lambda^{x_1+\cdots+x_n}}{\prod_i^n x_i!} &= e^{-n\lambda}\frac{(x_1+\cdots+x_n)\lambda^{x_1+\cdots+x_n-1}}{\prod_i^n x_i!},\\
            n\lambda^{x_1+\cdots+x_n} &= (x_1+\cdots+x_n)\lambda^{x_1+\cdots+x_n-1},\\
            n\lambda &= \sum_i^n x_i,\\
            \lambda &= \frac{1}{n}\sum_i^n x_i.

    Returns:
        None.
    """
    return None

def question_7():
    r""" Bayesian derivation of Poisson MLE.

    a) Derive the posterior assuming a conjugate prior :math:`p(\lambda) = Ga(\lambda|a,b) \propto \lambda^{a-1}e^{-\lambda b}`.

    From the above, the likelihood of the Poisson distribution is: :math:`e^{-n\lambda}\frac{\lambda^{x_1+\cdots+x_n}}{\prod_i^n x_i!}`,
    so we can write the posterior as:

    .. math::
        p(\lambda|D) &\propto \lambda^{a-1}e^{-\lambda b}e^{-n\lambda}\lambda^{x_1+\cdots+x_n}\\
            &= e^{-\lambda b -\lambda n}\lambda^{a+x_1+\cdots+x_n-1}\\
            &= Ga(a+x_1+\cdots+x_n-1, b+n)

    b) The MLE of the posterior looks can be found:

    .. math::
        0 &= \frac{d}{d\lambda}e^{-\lambda b -\lambda n}\lambda^{a+x_1+\cdots+x_n-1}\\
          &= -(b+n)e^{-\lambda(b+n)}\lambda^{a+x_1+\cdots+x_n-1}+e^{-\lambda b -\lambda n}(a+x_1+\cdots+x_n-1)\lambda^{a+x_1+\cdots+x_n-2}\\
        (b+n)e^{-\lambda(b+n)}\lambda^{a+x_1+\cdots+x_n-1} &= e^{-\lambda b -\lambda n}(a+x_1+\cdots+x_n-1)\lambda^{a+x_1+\cdots+x_n-2}\\
        (b+n)\lambda^{a+x_1+\cdots+x_n-1} &= (a+x_1+\cdots+x_n-1)\lambda^{a+x_1+\cdots+x_n-2}\\
        (b+n)\lambda &= a+x_1+\cdots+x_n-1\\
        \lambda &= \frac{a-1+\sum_i  x_i}{b+n}.

    If we look at what happens as the prior parameters :math:`a, b \rightarrow 0`, we see that the mean of the posterior
    approaches the MLE of the Poisson distribution.

    Returns:
        None.
    """
    return None


def question_8():
    r""" MLE for the uniform distribution.

    a) What is the MLE for data :math:`x_1,\ldots,x_n`?

    .. math::
        p(a|x_1,\ldots,x_n) &= \frac{\sum_i x_i}{(2a)^2}&\\
            &= \frac{-\sum_i x_i}{8a},~&\textrm{Der. with respect to}~a\\
        0   &= \frac{-\sum_i x_i}{8a},~&\textrm{Set equal to 0 to find maximum}.

    This is where some insight comes in. Solving for :math:`a` doesn't really work, but
    if you think about plotting this function (for some fixed sum of data), you see that
    it approaches 0 as :math:`a` increases. The MLE occurs when :math:`\hat{a} = max |x_i|, x_i \in {x_1,\ldots,x_n}`
    since it captures all the data seen so far and there's no support for anything further out
    in magnitude on the number line.

    b) The probability the model would assign to point :math:`x_{n+1}` is:

    .. math::
        p(x_{n+1}|\hat{a}) &= \frac{1}{2\hat{a}},\\
            &= \frac{1}{2x_{\textrm{max}}}.

    c) This doesn't make a great deal of sense, especially if only a few data points have been observed.

    It states that
    the only points that will be observed will be no bigger than :math:`|x_{\textrm{max}}|`. In reality, the only thing
    we know for sure is that :math:`a` is at least that large. We may instead set a prior on :math:`a` that takes into
    account the range of feasible values based on the specific problem. It would have the effect of broadening the range
    of values we expect to see, eventually tightening to all the points we've seen so far.

    Returns:
        None.
    """
    return None


def question_9():
    r""" Bayesian analysis of the uniform dist.

    Given a Pareto prior, the joint distribution of :math:`\theta` and :math:`\mathcal{D}` is
    :math:`p(\mathcal{D}, \theta) = \frac{Kb^K}{\theta^{N+K+1}}\mathbb{I}(\theta \geq max(\mathcal{D},b))`. We're also
    given :math:`p(\mathcal{D})`, and are asked to derive the posterior :math:`p(\theta|\mathcal{D})`. So...

    .. math::
        p(\theta|\mathcal{D}) &= \frac{p(\theta)p(\mathcal{D}|\theta)}{p(\mathcal{D})},~&\textrm{Bayes rule}\\
            &= \frac{p(\theta)p(\mathcal{D},\theta)}{p(\theta)p(\mathcal{D})},~&\textrm{Def. of Cond Prob.}\\
            &= \frac{p(\mathcal{D},\theta)}{p(\mathcal{D})}\\
            &= \frac{Kb^K}{\theta^{N+K+1}}\cdot\frac{(N+K)m^{N+K}}{Kb^K},~&\textrm{If max is }\geq b,~\textrm{or}\\
            &= \frac{(N+K)m^{N+K}}{\theta^{N+K+1}},\\
            &= \frac{Kb^K}{\theta^{N+K+1}}\cdot\frac{(N+K)b^{N}}{K},~&\textrm{If max is }< b,\\
            &= \frac{(N+K)b^{N+K}}{\theta^{N+K+1}},\\
            &= Pareto(\theta|N+K, max(\mathcal{D},b)),~&\textrm{Def. of the Pareto distribution}.

    Returns:
        None.
    """
    return None


def question_10():
    r""" Taxicab hijinks.

    You go to a city and see a taxi numbered 100. Can we figure out how many taxis there are in this city?

    a) Assuming we start with a :math:`Pareto(\theta,0,0)` distribution on the number, what's the posterior after
    seeing that first taxicab numbered 100?

    .. math::
        p(\theta|\mathcal{D}) &= Pareto(\theta|N+K,max(0,100))\\
            &= Pareto(\theta|1,100)

    b) Compute the posterior mean, mode, and median:

        i) mean = DNE, the rate parameter needs to be bigger.
        ii) mode = 100, we've only seen 1 data point!
        iii) median = 200...

        .. math::
            P(\theta \leq 0.5) &= \\
            0.5 &= \int_m^x km^k\theta^{-(k+1)}d\theta\\
                &= 100\int_{100}^x \theta^{-2}d\theta\\
                &= 100\left[-\theta^{-1}\rvert_{100}^x\right]\\
                &= \frac{100}{100} - \frac{100}{x}\\
            0.5 &= \frac{100}{x}\\
            x = 200.

    c) Derive an expression for the posterior predictive after :math:`\mathcal{D}=\{100\}`.

        .. math::
            p(\mathcal{D'}|\mathcal{D},\alpha) &= \int p(\mathcal{D'}|\theta)p(\theta|\mathcal{D},\alpha)d\theta\\
               &= \int_c^{\infty} Unif(x|\theta)Pareto(\theta|N+K,m)d\theta\\
               &= \int_c^{\infty}\theta^{-1}(N+K)m^{N+K}\theta^{-(N+K+1)}d\theta\\
               &= (N+K)m^{N+K}\int_c^{\infty}\theta^{-(N+K+2)}d\theta\\
               &= (N+K)m^{N+K}\left[\left.\frac{-1}{(N+K+1)\theta^{N+K+1}}\right|_c^{\infty}\right]\\
               &= \frac{m^{N+K}}{c^{N+K+1}}.

        This is all predicated on :math:`c = max(\mathcal{D'},m)`. We need to notice that the likelihood of a future
        point falling into the existing range doesn't need to stretch the max, so any future value less than the current
        max will get equal probability of happening. If we want to predict the probability of a higher numbered taxi,
        then we need to start the integral from that point forward given the evidence we've collected so far, so that
        should be less likely than a uniform distribution up to that number (since we haven't seen such a large value
        before). Using this formula, we can see that:

        i) :math:`p(x=50|\mathcal{D},\alpha) = \frac{1}{100}`
        ii) :math:`p(x=100|\mathcal{D},\alpha) = \frac{1}{100}`
        iii) :math:`p(x=150|\mathcal{D},\alpha) = \frac{1}{225}`

    e) There aren't an infinite number of taxis, so there should be a reasonable upper bound in the integral. The prior
    should also be set to a more reasonable value because the 150 case seems unusually low.

    Returns:
        None.
    """
    return None


def question_11():
    r""" Bayesian analysis of the exponential distribution.

    a) Derive the MLE of :math:`Expon(x|\theta)`.

        .. math::
            p(x|\theta) &= Expon(x|\theta)\\
                &= \theta e^{-x_1\theta}\cdots\theta e^{-x_n\theta}\\
                &= \theta^n e^{-\theta \sum_i x_i},~&\textrm{Take the der. and set to 0}\\
            0   &= n\theta^{n-1}e^{-\theta \sum_i x_i} - \theta^n \sum_i x_i e^{-\theta \sum_i x_i}\\
            \theta^n \sum_i x_i &= n\theta^{n-1}\\
            \frac{\sum_i x_i}{n} &= \frac{1}{\theta}\\
            \bar{x} &= \frac{1}{\theta}\\
            \frac{1}{\bar{x}} &= \theta.

    b) Given 3 observations of :math:`X, {5, 4, 6}`, what is the MLE of this data? :math:`\theta = \frac{1}{5}`.

    c) An expert thinks :math:`p(\theta) = Expon(\theta|\lambda)`. Choose the prior :math:`\hat{\lambda}` such that
    :math:`\mathbb{E}[\theta] = 1/3`. We can do the MLE route again, to see that :math:`\theta = \frac{1}{\lambda}` so
    we end up with :math:`\hat{\lambda} = 3` to get the desired expected value of :math:`\theta`.

    d) What is the posterior, :math:`p(\theta|\mathcal{D},\hat{\lambda})`?

        .. math::
            p(\theta|\mathcal{D},\hat{\lambda}) &= p(\theta)p(\mathcal{D}|\theta\hat{\lambda})\\
                &\propto Expon(\theta|\hat{\lambda})Expon(x|\theta)\\
                &= \theta e^{-\theta\hat{\lambda}}\theta^{n}e^{-\theta\sum_i x_i}\\
                &= \theta^{n+1}e^{-\theta(\sum_i x_i + \hat{\lambda}}\\
                &= Gamma(\theta|n+2, \sum_i x_i + \hat{\lambda}).

    e) The exponential prior is not conjugate to the exponential likelihood. It results in a Gamma, but it turns out
    that the exponential we selected was just a special case of the Gamma distribution. In general, based on this
    analysis, I would say that we used a :math:`Gamma(\theta|2,\hat{\lambda})` prior and not a :math:`Expon(\theta|\hat{\lambda})`
    prior since the posterior distribution should be the same form as the prior if it was conjugate.

    f) The posterior mean is :math:`\frac{n+2}{\sum_i x_i + \hat{\lambda}}` based on the statistics of the Gamma dist.

    g) The MLE and the posterior mean differ because there wasn't a prior involved in the MLE derivation. The prior
    suggests that the rate of failure is somewhat shorter than we have observed so far, and this additional information
    wasn't available in just the likelihood alone. The posterior mean is probably more reasonable assuming the experts
    can give a reaonable prior estimate from their experience of studying other machines produced by a similar process.


    Returns:
        None.
    """
    return None


def question_12():
    r""" Bernoulli MAP estimate with non-conjugate priors.

    a) What if you used a prior: :math:`p(\theta) = 0.5 if \theta = 0.5, 0.5 if \theta = 0.4, 0 otherwise`.

        .. math::
            p(\theta|D) &= p(\theta)p(D|\theta)\\
                &= p(\theta)\binom{N}{N_1}\theta^{N_1}(1-\theta)^{N-N_1}\\
                &= 0.5\binom{N}{N_1}0.5^{N_1}0.5^{N-N_1}+0.5\binom{N}{N_1}0.4^{N_1}0.6^{N-N_1}\\
                &= 0.5\binom{N}{N_1}\left[0.5^{N}+0.4^{N_1}0.6^{N-N_1}\right].

    b) What if the true parameter is :math:`\theta = 0.41`. Which prior works better? For small :math:`N`, we'll likely
    find that the new prior is more accurate because it wants a :math:`\theta` close to 0.45. Unfortunately, its
    effect never really diminishes with increasing trials and so the data doesn't really overwhelm it. The Beta prior
    is more likely to have greater errors in the early stages (unless very specific parameters are selected), but
    instead of blending two specific values of :math:`\theta` in fixed amounts, the blending is a part of the data
    collections and in the limit will converge to the true value.

    Returns:
        None.
    """
    return None


def numbers_main():
    game = NumberGame()
    samples = []
    while (
            not samples
            or max(game.posterior(samples)) < 0.99
    ):
        samples.append(game.sample_from_concept())
        print(f"Samples: {samples}")
        print(f"Posteriors: {game.posterior(samples)}")
        print(f"Plug-in Distribution: {game.plugin_distribution(samples)}")
        print(f"Posterior Predictive Distribution: {game.post_predictive_distribution(samples)}")
    print(f"MAP: {game.concepts[np.argmax(game.posterior(samples))].name}")
    print(f"True concept: {game.active_concept}")


def bb_main():
    bb = BetaBinomial(5, 2)
    samples = list(np.ones(11, dtype=np.int))+list(np.zeros(13, dtype=np.int))
    df = bb.posterior(samples)
    print(df)
    print(df.prob.idxmax())
    print(df.prob.sum())
    print(4/22)
    print(15/29)


if __name__ == "__main__":
    #bb_main()
    question_4()

