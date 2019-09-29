from setuptools import setup


with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup_config = {
    'name': 'lfire',
    'version': 0.1,
    'packages': ['lfire'],
    'install_requires': requirements,
    'author': 'Jan Kokko',
    'author_email': 'jan.kokko@helsinki.fi',
    'url': 'https://github.com/elfi-dev/zoo/lfire',
    'licence': 'BSD'
}

setup(**setup_config)
