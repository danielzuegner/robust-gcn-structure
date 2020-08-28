from setuptools import setup

setup(name='robust_gcn_structure',
      version='0.1',
      description='Certifiably Robust GCN under Structure Perturbations',
      author='Daniel Zügner',
      author_email='zuegnerd@in.tum.de',
      packages=['robust_gcn_structure'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'torch', 'tqdm', 'cvxpy==1.1.4'],
      zip_safe=False)
