""" Worked examples and exercises from Chapter 3.

"""
import math
import random

import numpy as np


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

def main():
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


if __name__ == "__main__":
    main()


