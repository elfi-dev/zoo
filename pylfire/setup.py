from setuptools import setup


with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup_config = {
    'name': 'pylfire',
    'version': 0.1,
    'packages': ['pylfire'],
    'install_requires': requirements,
    'author': 'Jan Kokko',
    'author_email': 'jan.kokko@helsinki.fi',
    'url': 'https://github.com/elfi-dev/zoo/pylfire',
    'licence': 'BSD'
}

setup(**setup_config)
