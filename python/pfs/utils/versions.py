import os
import pathlib

try:
    import eups
except ImportError:
    eups = None

__all__ = ["getVersion"]


def getVersion(productName):
    """
    Return a single version string for a named product. Prefers git over eups.

    Args
    ----
    productName : str
      The EUPS product name to check.

    Returns
    -------
    version : str
        The determined version string.

    Raises
    ------
    RuntimeError
        If the version could not be determined from either git or eups.
    """

    def getGitVersion(prodDir):
        """Retrieve the git version of the product."""
        try:
            import git
        except ModuleNotFoundError:
            return None
        try:
            gitRepo = git.Repo(prodDir)
            return gitRepo.git.describe(dirty=True, always=True)
        except git.InvalidGitRepositoryError:
            return None

    def getEupsVersion(product):
        """Retrieve the eups version of the product."""
        if eups is None:
            return None
        eupsEnv = eups.Eups()
        eupsProd = eupsEnv.findSetupProduct(product)
        return eupsProd.version if eupsProd else None

    def guessProductDir(product):
        """Guess product directory without eups."""
        try:
            prodDir = os.environ[f'{product.upper()}_DIR']
        except KeyError:
            prodDir = pathlib.Path().cwd()

        return prodDir

    def getProductVersions(product):
        """Determine git and eups versions of the product."""
        eupsVersion = getEupsVersion(product)
        prodDir = guessProductDir(product) if eupsVersion is None else eups.Eups().findSetupProduct(product).dir
        gitVersion = getGitVersion(prodDir)
        return gitVersion, eupsVersion

    gitVersion, eupsVersion = getProductVersions(productName)

    if gitVersion:
        return gitVersion
    elif eupsVersion:
        return eupsVersion

    raise RuntimeError(f"Could not determine version for product: {productName}")
