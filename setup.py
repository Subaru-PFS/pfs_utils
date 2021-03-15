from setuptools import setup, find_packages

setup(name="pfs.utils",
      # version="x.y",
      author="",
      # author_email="",
      # description="",
      url="https://github.com/Subaru-PFS/pfs_utils/",
      packages=find_packages("python"),
      package_dir={'': 'python'},
      package_data={
        "pfs.utils": ["coordinates/data/*"]},
      license="",
      install_requires=["numpy"],
      )
