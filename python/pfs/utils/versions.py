import os
import pathlib
import sys

import git
import eups

def gitVersion(repoDir=None):
    try:
        gitRepo = git.Repo(path=repoDir)
    except git.InvalidGitRepositoryError:
        gitRepo = None

def _eupsProductIsLocal(prod):
    return prod.version.startswith(prod.LocalVersionPrefix)

def versions(productName):
    """
    Try to return the current version of the given product.

    We expect three environments for a running program:
      - a setup EUPS directory with a live git repo
      - a live git repo in `pwd`
      - a setup EUPS-tagged version.

    We always use a git version if available, since that is the most descriptive and the best
    managed.

    Failing a live git version, use the EUPS version. If that is a
    tag, it is expected to match a git tag.
    """

    eupsEnv = eups.Eups()
    eupsProd = eupsEnv.findSetupProduct(productName)
    if eupsProd is None:
        eupsVersion = None
        prodDir = pathlib.Path().cwd()
    prodDir = eupsProd.dir
    eupsVersion = eupsProd.version if not _eupsProductIsLocal(eupsProd) else None


    try:
        gitRepo = git.Repo(prodDir)
    except git.InvalidGitRepositoryError:
        gitRepo = None

    if gitRepo is None:
        if eupsVersion is None:
            return 'none', 'none'
        return eupsVersion, 'eups'

    # Could check for git-eups conformabity or warn?
    return gitRepo.git.describe(), 'git'
