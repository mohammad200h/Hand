from setuptools import setup, find_packages

setup(
    name='hand_orch_multiprocessing',  # Ensure the correct name
    version='0.0.1',
    packages=find_packages(),  # Automatically includes all Python modules under the package
    install_requires=['gym', 'numpy', 'pybullet'],  # Dependencies
)
