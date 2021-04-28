from pfs.utils.constants import FIBERS_PER_SPECTROGRAPH

__all__ = ("FIBERS_PER_SPECTROGRAPH", "calculateFiberId", "spectrographFromFiberId", "fiberHoleFromFiberId")


def calculateFiberId(spectrograph, fiberHole):
    """Calculate the fiberId from spectrograph, fiberHole

    The fiberId is a unique integer identifier for each fiber, covering science
    and engineering fibers.

    Parameters
    ----------
    spectrograph : array_like
        Spectrograph number (1..4).
    fiberHole : array_like
        Fiber hole number (1..651).

    Returns
    -------
    fiberId : array_like
        Fiber identifier (1..2604)
    """
    return FIBERS_PER_SPECTROGRAPH*(spectrograph - 1) + fiberHole


def spectrographFromFiberId(fiberId):
    """Calculate spectrograph number from fiberId

    Parameters
    ----------
    fiberId : array_like
        Fiber identifier (1..2604).

    Returns
    -------
    spectrograph : array_like
        Spectrograph number (1..4).
    """
    return (fiberId - 1)//FIBERS_PER_SPECTROGRAPH + 1


def fiberHoleFromFiberId(fiberId):
    """Calculate fiber hole number from fiberId

    Parameters
    ----------
    fiberId : array_like
        Fiber identifier (1..2604).

    Returns
    -------
    fiberHole : array_like
        Fiber hole number (1..651).
    """
    return (fiberId - 1) % FIBERS_PER_SPECTROGRAPH + 1
