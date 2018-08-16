def question_1():
    r""" Pessimism of LOOCV. I didn't like the question because I thought it was a little too open...

    You want to show that LOOCV can be a really bad estimate of error in some situations. For example, if you
    have a random class label that you're trying to predict (binary, equal proportions), LOOCV can report
    a really bad error bound. You're asked what the best classifier is for this data and what happens when you
    do its LOOCV estimate.

    The question is leading (though not prescribing) you towards a simple majority classifier. It will be right the
    maximum number of times (0.5), and it's LOOCV error rate will be extremely pessimistic, at 100%. The reason is
    simple enough - if you remove a single sample, then the other class becomese the majority so you'll predict that
    but it's only the majority because the test sample belonged to the other class and so you'll be wrong every time.

    But this isn't the only optimal classifier specification for this problem. You can also have a probabilistic
    classifier that outputs a label in proportion to the input. In this case you'll again achieve an accuracy of 0.5,
    but the LOOCV will essentially also be 0.5 as the training data grows. For small amounts of training data the
    process will give you something closer to a 0.49 error rate, but it's still not that bad.

    In short, the LOOCV is pessimistic, but to get the wild swing desired by the question it should have been explicit
    and asked the reader to investigate the majority classifier (as is done in the source discussion referred to in the
    question (Witten, Frank, Hall p.154 in Data mining 3rd edition).

    See ``chapter.six.demo_loocv`` for simulation study of this question.

    Returns:
        None.

    """