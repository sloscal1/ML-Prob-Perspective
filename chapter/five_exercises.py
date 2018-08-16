def question_1():
    r""" Posterior of a mixture of conjugate priors is a mixture of conjugates.

    .. math::
        p(\theta|D) &= \frac{p(\theta, D)}{p(D)},~&\textrm{Def. Cond. Prob},\\
            &= \frac{\sum_k p(z=k)p(\theta, D | z=k)}{\sum_{k'} p(z=k')p(D|z=k')},~&\textrm{Law of Total Prob},\\
            &= \frac{\sum_k p(z=k)p(D|z=k)p(\theta|D,z=k)}{\sum_{k'} p(z=k')p(D|z=k')},~&\textrm{Chain Rule of Prob.},\\
            &= \sum_k p(z=k|D)p(\theta|D,z=k),~&\textrm{Bayes Thm.}

    So we have the posterior of a mixture equal to the weighted sum of the posterior of each individual posterior, with
    the weights being the posteriors of the mixture distribution.

    Returns:
        None.
    """
    return None


def question_3():
    r""" Reject options.

    Basically, you have C possible classes that you can select as your prediction (as defined by a particular problem)
    plus another option: you can reject making a prediction. The cost of a misprediction is :math:`\lambda_s`, while
    the cost for rejecting is :math:`\lambda_r`.

    a) Figure out the minimum risk attained given the above parameters.
        We were given :math:`p(y=j|x) = \theta`, but the trick is to figure out where the threshold :math:`T` is between
        getting a rejection and misprediction error. That probability is :math:`p(\hat{y}=j|x) > T`, so we can state
        the loss function in terms of :math:`T` and go from there. Make sure you pay attention to :math:`\hat{y}` versus
        :math:`y`! Once the expected value expression is available, we take the derivative and set it equal to 0 to
        find where it is maximized.

        .. math::
            \mathbb{E}[\ell p(\hat{y}=j|x)] &= \int_0^T \lambda_r t dt+ \int_T^1 \lambda_s (1-\theta) t dt,~\textrm{Def. of Expected Value}\\
                &= \frac{\lambda_rT^2}{2} + \lambda_s(1-\theta)\left[\frac{1}{2} - \frac{T^2}{2}\right]\\
            0 &= \frac{\lambda_rT^2}{2} + \lambda_s(1-\theta)\left[\frac{1}{2} - \frac{T^2}{2}\right]\frac{d}{dT}\\
                &= \lambda_rT - \lambda_s(1-\theta)T\\
                \frac{\lambda_rT}{\lambda_sT} &= 1-\theta\\
                1-\frac{\lambda_r}{\lambda_s} &= \theta.

    b) Qualitatively, we see a linear function in terms of the cost matrix and the max class :math:`j`.
        As loss due to rejection grows from 0 to the cost of a misprediction error, we see at first that
        the system must be 100% confident in its prediction to go forward with it, otherwise it's free to reject. when
        the two equal each other then there is no benefit to rejecting anything anymore and the majority prediction will
        be selected at all times instead of the reject option.

    Returns:
        None.
    """
    return None


def question_4():
    r""" More reject options.

    Based on question 3 we know that the risk threshold when a reject option is in play is :math:`1-\frac{\lambda_r}{\lambda_s}`,
    and given the table in this problem, we see that it is a safer decision to reject a prediction if it is less than
    :math:`1-\frac{3}{10} = 0.7`. Using this fact we can solve the next couple of questions:

    a) Suppose :math:`p(y=1|x) = 0.2`. Which decision minimizes the expected loss? :math:`\hat{y} = 0`
        Its probability is 0.8.
    b) Suppose :math:`p(y=1|x) = 0.4`. Which decision minimizes the expected loss? Reject.
        The maximum confidence is 0.6 < 0.7.
    c) The thresholds can be specified a little more crisply than the previous question where there were C classes.
        In the binary case, predict 0 if :math:`p_1 < \frac{\lambda_r}{\lambda_s} = \theta_0`, predict 1 if
        :math:`p_1 > 1-\frac{\lambda_r}{\lambda_s} = \theta_1`, and reject if :math:`\theta_0 \leq p_1 \leq \theta_1`.
        The idea is that any posterior distribution is a distribution and therefore sums to one. If :math:`p_1` is
        small, then :math:`p_0` is large and we should select that, and vice versa. Maximum uncertainty occurs when
        :math:`p_1=p_0=0.5` in which case we should reject as long as the cost for doing so is less than the cost of
        a misprediction. The reject region grows wider (symmetrically around 0.5 if the loss is equal for a misprediction
        in either class) as the cost of a rejection grows smaller proportional to the cost of a misprediction.

    Returns:
        None.
    """
    return None
