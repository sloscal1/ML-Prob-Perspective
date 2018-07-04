import random

def question_1():
    """
    p(gender=boy) = 0.5
    p(gender=girl) = 0.5

    Possible outcomes of 2 children:
    boy, girl
    boy, boy
    girl, boy
    girl, girl

    a) If you know the neighbor has at least one boy, what is the probability the neighbor has a girl?
    Sample space: (b,g), (b,b), (g,b). 2/3 events have a girl involved, and they all have equal probability so 2/3.

    b) What is the probability that the other child is a girl if you see that one is a boy?
    Sample space: (b,g), (b,b). 1/2. The children are independent of each other, so it's the same as the probability
    of one child being a girl.

    Returns:
        None.
    """
    return None


def question_2():
    """
    There is a blood found at a crime scene that has no innocent explanation. The blood is of a type found in
    1% of the population.

    a) Prosecutor's fallacy: 1% chance that the defendant would have the crime scene blood type if he was innocent,
    therefore there is a 99% chance that he is guilty.

    This is not what the evidence states: 1% of the population could have committed the crime because only they have
    the suspect blood type. The defendant has that blood type, so he is 1/K people who are in consideration for
    committing the crime, not 99% likely to have committed the crime. 99% of the population is not in consideration
    for the crime at all, but based on the blood evidence alone we cannot state the likelihood of this single
    defendent having committed this crime, only that he is in the consideration set.

    b) Defendant's fallacy: There are 800K people in the city, 8000 have the blood type in question. There is just
    1 in 8000 chance that the defendant is guilty and so has no relevance.

    While it is true that the defendant is just 1 of 8000 city dwellers that have the matching blood type, the blood
    is relevant. The true culprit must have that blood type, and so it establishes that further evidence must be
    produced to establish the innocence or guilt of the defendant. This is far from the situation that we can ignore
    the blood type, the guilty part(ies) must have that match to be considered for the crime.

    Returns:
        None.

    """
    return None

def question_3():
    r"""
    Variance of a sum.

    .. math::
        cov[X, Y] &= \mathbb{E}[[X - \mathbb{E}[X]][Y - \mathbb{E}[Y]]]\\
            &= \mathbb{E}[XY - X\mathbb{E}[Y] - Y\mathbb{E}[X] + \mathbb{E}[X]\mathbb{E}[Y]]\\
            &= \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] - \mathbb{E}[X]\mathbb{E}[Y] + \mathbb{E}[X]\mathbb{E}[Y]\\
            &= \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]


    .. math::
        var[X + Y] &= \mathbb{E}[(X + Y - \mathbb{E}[X+Y])^2]\\
            &= \mathbb{E}[X^2] + \mathbb{E}[XY] - \mathbb{E}[X\mathbb{E}[X+Y]] + \mathbb{E}[XY] + \mathbb{E}[Y^2] - \mathbb{E}[Y\mathbb{E}[X+Y]] - \mathbb{E}[X\mathbb{E}[X+Y]] - \mathbb{E}[Y\mathbb{E}[X+Y]] + \mathbb{E}[X+Y]^2\\
            &= \mathbb{E}[X^2] - \mathbb{E}[X]^2 - \mathbb{E}[X]\mathbb{E}[Y] + \mathbb{E}[Y^2] - \mathbb{E}[Y]^2 - \mathbb{E}[X]\mathbb{E}[Y] +2\mathbb{E}[XY] - \mathbb{E}[X]^2 - 2\mathbb{E}[X]\mathbb{E}[Y] - \mathbb{E}[Y]^2 + \mathbb{E}[X+Y]^2\\
            &= var(X) + var(Y) + 2\mathbb{E}[XY] - 4\mathbb{E}[X]\mathbb{E}[Y] - \mathbb{E}[X]^2 - \mathbb{E}[Y]^2 + \mathbb{E}[X+Y]^2\\
            &= var(X) + var(Y) + 2cov(X, Y) - 2\mathbb{E}[X]\mathbb{E}[Y] - \mathbb{E}[X]^2 - \mathbb{E}[Y]^2 + \mathbb{E}[X]^2 + 2\mathbb{E}[X]\mathbb{E}[Y] + \mathbb{E}[Y]^2\\
            &= var(X) + var(Y) + 2cov(X, Y)\\

    Returns:
        None.
    """
    return None

def question_4():
    r"""
    Given:

    .. math::
        P(T=p|D=p) &= 0.99\\
        P(T=n|D=n) &= 0.99\\
        P(D=p) &= 1/10,000


    This is an application of Bayes Theorem since we want to update the prior probability of having
    the disease after knowing the test came back positive. So we have:

    .. math::
        P(D=p|T=p) &= \frac{P(T=p|D=p) \cdot P(D=p)}{P(T=p)}, &~\textrm{Bayes Thm.}\\
                   &= \frac{P(T=p|D=p) \cdot P(D=p)}{\Sigma_d P(T=p|D=d)\cdot P(D=d)}, &~\textrm{Law of Total Prob.}\\
                   &= \frac{P(T=p|D=p) \cdot P(D=p)}{P(T=p|D=p) \cdot P(D=p) + P(T=p|D=n) \cdot P(D=n)}, &~\textrm{Notation}\\
                   &= \frac{0.99 \cdot 0.0001}{0.99 \cdot 0.0001 + 0.01 \cdot 0.9999}, &~\textrm{Law of Total Prob.}\\
                   &\approx 0.0098.


    This means that the good news is the probability of having the disease is still a little less than 1/100. Also,
    The second application of the Law of Total Probability is actually two applications:

    .. math::
        1 &= P(D=p) + P(D=n)\\
        1 &= P(T=p|D=p) + P(T=p|D=n)


    Returns:
        None.
    """
    print(0.99*0.0001/(0.99*0.0001+0.01*0.9999))
    return None


def question_5(num_samples=1000000, seed=1337):
    r""" The Monty Hall Problem using Bayes theorem.

    We're interested in determining whether switching doors is better than sticking with the original.

    Let :math:`C \sim Unif(3)` be the random variable representing where the car (prize) is,
    :math:`F \sim Unif(3)` be the random variable
    representing the first selection made by the contestant, and :math:`O` be the random variable representing
    which door is opened after the first selection is made. This variable is deterministic when the first guess does
    not equal the prize value but has a choice otherwise.

    .. math::
        P(F=P|O, P) &= \frac{P(O|F=P) \cdot P(F=P)}{P(O|P=F)},~&\textrm{Bayes Theorem}\\
                    &= \frac{1/2 \cdot 1/3}{1/2},~&\textrm{Counting}\\
                    &= 1/3.\\
        P(F\neq P|O, P) &= \frac{P(O|F\neq P) \cdot P(F\neq P)}{P(O|P\neq F)},~&\textrm{Bayes Theorem}\\
                        &= \frac{1 \cdot 2/3}{1},~&\textrm{Counting}\\
                        &= 2/3.

    So from this we see that our first guess has a 2/3 chance of being wrong given the open door, so switching would
    give us a 2/3 of being correct in that case. Additionally, by the Law of Total Probability, we could've computed
    the chances of the first guess being correct (1/3) and taking the complement of that.

    Side-effect:
        This code runs a simulation of the Monty Hall Problem to compute the probabilities and prints the
        probability of being right when staying with the original choice or switching to the remaining door.

    Args:
         num_samples (int): the number of times to sample the distribution, must be positive.
         seed (int): the random seed to ensure repeatability.

    Returns:
        None.
    """
    random.seed = seed
    stay = 0
    switch = 0
    for _ in range(num_samples):
        prize = random.randint(0, 2)
        first = random.randint(0, 2)
        if prize != first:
            # Trick: 3 - (0 + 1): 2; 3 - (0 + 2): 1; 3 - (1 + 2): 0.
            open_door = 3 - (first + prize)
        else:
            # Trick: 1 + 0: 1, 2 + 0: 2; 1 + 1= 2, 1 + 2 = 0; 2 + 1 = 0, 2 + 2 = 1.
            open_door = (random.randint(1, 2) + prize) % 3
        if first == prize:
            stay += 1
        # Trick: 0 + 1 = 2, 0 + 2 = 1, 1 + 0 = 2, 1 + 2 = 0, 2 + 1 = 0 2 + 0 = 1
        second = 3 - (open_door + first)
        if prize == second:
            switch += 1
    print(f"Correct stay probability: {stay/num_samples*100:0.3f}%;"
          f"\nCorrect switch probability: {switch/num_samples*100:0.3f}%")


def question_6():
    r"""
    Want to know if you can compute :math:`P(H|e_1,e_2)` with different givens.

    Let's look at what this formula looks like after rearranging.

    .. math::
        P(H|e_1,e_2) &= \frac{P(e_1,e_2|H) \cdot P(H)}{P(e_1,e_2)},~&\textrm{Bayes Thm.}\\
                     &= \frac{P(e_1|H) \cdot P(e_2|H) \cdot P(H)}{P(e_1,e_2)},~&\textrm{Def. of Cond. Ind.}\\
                     &= \frac{P(e_1|H) \cdot P(e_2|H) \cdot P(H)}{\Sigma_h P(e_1,e_2|H) \cdot P(H)},~&\textrm{Total Probability}\\
                     &= \frac{P(e_1|H) \cdot P(e_2|H) \cdot P(H)}{\Sigma_h P(e_1|H)\cdot P(e_2|H) \cdot P(H)},~&\textrm{Def. of Cond. Ind.}


    i.      :math:`P(e_1,e_2), P(H), P(e_1|H), P(e_2|H)`. This is sufficient from the second line above if we assume
            independence between the :math:`E` variables.
    ii.     :math:`P(e_1,e_2), P(H), P(e_1,e_2|H)`. This is sufficient from the first line above, a single
            applications of Bayes Theorem.
    iii.    :math:`P(e_1|H), P(e_2|H), P(H)`. This is sufficient from the last line, after applying the Law of total
            probability and Conditional Independence.


    So (ii) is the answer to part a), when we don't know anything about the relationship between :math:`E_1` and
    :math:`E_2`. All sets of givens are sufficient if we know the two variables are conditionally independent.

    Returns:
        None.

    """
    # What is an example of conditional independence:
    # https://en.wikipedia.org/wiki/Conditional_independence
    # P(R|Y) = 4/12, P(B|Y) = 6/12, P(R|Y)*P(B|Y) = 6/36 = 2/12 = P(R,B|Y)
    # P(!R|Y) = 8/12, P(!B|Y) = 6/12, P(!R|Y)*P(!B|Y) = 8/24 = 4/12 = P(!R,!B|Y)
    # P(!R|Y) = 8/12, P(B|Y) = 6/12, P(!R|Y)*P(B|Y) = 8/24 = 4/12 = P(!R,B|Y)
    # P(R|Y) = 4/12, P(!B|Y) = 6/12 P(R|Y)*P(!B|Y) = 6/36 = 2/12 = P(R,!B|Y)
    # So R \ind B | Y.
    return None


def question_7():
    r""" Pairwise independence does not imply mutual independence.

    Mutual independence means that :math:`P(X_i|X_S) = P(X_i) \forall S \subseteq \{1,\ldots,n\}\setminus\{i\}`
    and so the joint distribution of :math:`P(X_{1:n}) = \prod_{i=1}^n P(X_i)`.

    So it would be enough to show that for 3 variables that are all pairwise independent that they are
    not mutually independent.

    Consider a 5x5 grid where one variable :math:`(X_1)` is true only along the bottom 5 squares, another is true only
    along the right side :math:`(X_2)`, and a third is true only along the main diagonal :math:`(X_3)`. The only overlap
    any variable has with any other is in the lower right corner square.

    .. math::
        P(X_1=T) &= 5/25\\
        P(X_1=F) &= 20/25\\
        P(X_1=T,X_2=T) &= 1/25 = 5/25*5/25 = P(X_1=T)P(X_2=T)\\
        P(X_1=T,X_2=F) &= 4/25 = 5/25*20/25 = P(X_1=T)P(X_2=F)\\
        P(X_1=F,X_2=T) &= 4/25 = 20/25*5/25 = P(X_1=F)P(X_2=T)\\
        P(X_1=F,X_2=F) &= 16/25 = 20/25*20/25 = P(X_1=F)P(X_2=F)\\

    In this way, we see that each pair of variable is conditionally independent. The question is if they are
    mutually independent. If they were, then :math:`P(X_1,X_2,X_3) = P(X_1)P(X_2)P(X_3)`, but we see for
    :math:`P(X_1=T,X_2=T,X_3=T) = 1/25` (the lower right corner), but :math:`P(X_1=T)P(X_2=T)P(X_3=T) = 1/125` so
    we see that being pairwise conditionally independent does not imply mutual independence.

    Returns:
        None.
    """
    return None


def question_8():
    r""" Conditional independence iff joint factorizes.

    Prove that :math:`p(x,y|z)=g(x,z)h(y,z)~\textrm{iff}~X \perp Y | Z.`

    First, let :math:`g(x,z) = p(x|z), h(y,z) = p(y|z)` since conditional probabilities
    are functions of random variables these are permissible definitions of :math:`g, h`.

    :math:`\textrm{The forward direction:}~X \perp Y | Z \Rightarrow p(x,y|z)=g(x,z)h(y,z).`

    .. math::
       p(x,y|z) &= p(x|z)p(y|z),~&\textrm{Def. of Cond. Ind.}\\
                &= g(x,z)h(y,z),~&\textrm{Defined above.}.

    Lemma: :math:`p(x|y,z) = p(x|z)~\textrm{if}~X \perp Y | Z.`

    Proof:

    .. math::
        p(x|y,z) &= \frac{p(x,y,z)}{p(y,z)},~&\textrm{Def. of Cond. Prob.}\\
                 &= \frac{p(x,y|z)p(z)}{p(y|z)p(z)}~&\textrm{Def. of Cond. Prob.}\\
                 &= \frac{p(x|z)p(y|z)p(z)}{p(y|z)p(z)}~&\textrm{Def. of Cond. Ind.}\\
                 &= p(x|z).

    :math:`\textrm{The reverse direction:}~p(x,y|z)=g(x,z)h(y,z) \Rightarrow X \perp Y | Z.`

    .. math::
        p(x,y|z) &= \frac{p(x,y,z)}{p(z)},~&\textrm{Def. of Cond. Prob.}\\
                 &= \frac{p(z)p(y|z)p(x|y,z)}{p(z)},~&\textrm{Chain rule of prob.}\\
                 &= p(y|z)p(x|z),~&\textrm{By the above lemma, Def. Cond. Ind.}\\
                 &= g(x,z)h(y,z),~&\textrm{Defined above.}

    I don't think I ever really proved that if you have two functions that you can combine
    them into the conditional probability statement desired.



    Returns:

    """

if __name__ == "__main__":
    question_5()
