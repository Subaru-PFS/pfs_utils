import pathlib

import eups
try:
    import git
except ImportError:
    git = None
    
def version(productName):
    """
    Return a single version string for a named product. Prefers git over eups.

    Args
    ----
    productName : str
      The EUPS product name to check.

    Prefer the git version if available, since that is the most
    descriptive and the best managed.

    Failing a live git version, use the EUPS version. If that is a
    tag, it is expected to match a git tag. But we do not check.
    """

    gitVersion, eupsVersion = versions(productName)
    if gitVersion is None:
        return eupsVersion if eupsVersion is not None else 'unknown'

    return gitVersion

def allSetupVersions():
    eupsEnv = eups.Eups()
    setupProds = eupsEnv.findProducts(tags='setup')

    res = dict()
    for p in setupProds:
        prodInfo = eupsEnv.findSetupVersion(p.name)
        if prodInfo[0] is None:
            continue
        res[p.name] = version(p.name)
    return res

def _eupsProductIsLocal(prod):
    """ Return True if the eups version is for a local (setup -r $dir) setup. """

    return prod.version.startswith(prod.LocalVersionPrefix)

def versions(product):
    """ Try to return the current git and eups versions of the given product.

    Args
    ----
    product : string or eups Product instance
      EUPS product name to query

    Returns
    -------
    gitVersion : standard git version string or None
    eupsVersion : eups version name or None

    We expect three environments for a running program:
      - a setup EUPS directory with a live git repo
      - a live git repo in `pwd`
      - a setup EUPS-tagged version.

    """

    if isinstance(product, str):
        eupsEnv = eups.Eups()
        eupsProd = eupsEnv.findSetupProduct(product)
    else:
        eupsProd = product
        
    if eupsProd is None:
        eupsVersion = None
        prodDir = pathlib.Path().cwd()
    else:
        eupsVersion = eupsProd.version
        prodDir = eupsProd.dir

    try:
        gitRepo = git.Repo(prodDir)
        gitVersion = gitRepo.git.describe(dirty=True)
    except AttributeError:
        gitRepo = None
        gitVersion = None
    except git.InvalidGitRepositoryError:
        gitRepo = None
        gitVersion = None

    # Could check for git-eups conformabity or warn?
    return gitVersion, eupsVersion

