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

    """
    return None