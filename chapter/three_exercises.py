import math
import random

import numpy as np


class NumberGame(object):
    def __init__(self, *, seed=None):
        if seed:
            np.random.seed = seed
            random.seed(seed)
        self.max_val = 100
        primes = [2]
        for num in range(3, self.max_val+1, 2):
            if all([num % prime for prime in primes]):
                primes.append(num)
        self.concepts = {
            "odd": [num for num in range(1, self.max_val+1, 2)],
            "squares": [num*num for num in range(1, 11)],
            "primes": primes,
            "all": list(range(1, self.max_val+1)),
        }
        for val in range(2, 11):
            self.concepts[f"mult of {val}"] = [num for num in range(val, self.max_val+1, val)]
            self.concepts[f"ends in {val}"] = [num for num in range(val, self.max_val+1, 10)]
            self.concepts[f"power of {val}"] = [val**num for num in range(0, int(math.ceil(math.log(self.max_val, val))))]
        self.concepts["evens"] = self.concepts["mult of 2"]
        self.concepts.pop("mult of 2")
        self.concepts["power of 2 + (37)"] = [val for val in np.add(self.concepts["power of 2"], 37) if val <= 100]
        self.concepts["power of 2 - (37)"] = [val for val in np.add(self.concepts["power of 2"], -37) if val > 0]+[128-37]
        self.active_concept = None
        self.priors = dict()
        print(self.concepts)
        for key in self.concepts:
            print(f"{key}: {len(self.concepts[key])}")

    def select_concept(self):
        self.active_concept = random.choice(list(self.concepts.keys()))
        print(self.active_concept)
        return None

    def sample_from_concept(self):
        return self.concepts[self.active_concept][random.randint(0, len(self.concepts[self.active_concept]))-1]

    def likelihood(self, samples):
        """
        Compute the :math:`p(D|h) = \frac{p(D,h)}/{p(h)}`.
        Args:
            samples (list int): a list of samples from the target concept.

        Returns:
            list float: the likelihood of the data given the hypothesis. This comes
                from the strong sampling assumption
        """
        n = len(samples)
        # I think this likelihood should include the probability of NOT seeing certain numbers.
        return [1.0/(len(h)**n) for h in self.concepts.values()]

    def better_likelihood(self, samples):
        # what is the probabililty of seeing a subset of the numbers from this concept?
        # If this was the true concept, we would expect to see a uniform distribution across
        # these values.
        # 4, 16, 64
        # 2, 4, 8, 16, ,32, 64
        # p(X=2) = 1/6
        # 5/6**N
        return None

    def set_priors(self):
        self.priors.clear()
        self.priors["even"] = 0.15
        self.priors["odd"] = 0.15
        self.priors["power of 2 + (37)"] = 0.0005
        self.priors["power of 2 - (37)"] = 0.0005
        self.priors["primes"] = 0.05
        others = (1.0 - sum(self.priors.values()))/(len(self.concepts) - len(self.priors))
        for concept in self.concepts.keys():
            if concept not in self.priors:
                self.priors[concept] = others
        return self.priors

    def posterior(self, samples):
        unique_samps = set(samples)
        denominator = 0
        posteriors = []
        likelihoods = self.likelihood(samples)
        for pos, (name, hyp) in enumerate(self.concepts.items()):
            num = 0
            if unique_samps.issubset(set(hyp)):
                num = self.priors[name]/likelihoods[pos]
            denominator += num
            posteriors.append(num)
        return np.divide(posteriors, denominator)


def likelihood_ratio(posteriors):
    temps = sorted(posteriors)
    lr = temps[-1]/temps[-2]
    return lr

def main():
    game = NumberGame(seed=1337)
    game.select_concept()
    game.set_priors()
    samples = []
    while (
            not samples
            or (likelihood_ratio(game.posterior(samples)) < 1000
            and max(game.posterior(samples)) < 0.99)):
        samples.append(game.sample_from_concept())
        print(f"Samples: {samples}")
        print(f"Posteriors: {game.posterior(samples)}")


if __name__ == "__main__":
    main()


