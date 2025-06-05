from setuptools import setup, find_packages
setup(name='fingers',
      version='0.0.1',
      install_requires=['gym', 'numpy', 'pybullet'],
        packages=find_packages(),
      package_data={
          'fingers': ['envs/finger_ws/*',
                      'envs/goal/*',
                      'envs/initial_env_setting/*',
                      'env/model/*'
                    ],  # Include everything under the 'data' folder and its subfolders
      },
)
