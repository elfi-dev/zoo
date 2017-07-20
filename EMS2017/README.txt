Installation
------------

Make sure that you have the python libraries in requirements.txt installed.
Example:
 virtualenv -p python3 --system-site-packages .venv
 source .venv/bin/activate
 pip install -r requirements.txt

For RISE (https://github.com/damianavila/RISE) with virtualenv, also run:
 jupyter-nbextension install rise --py --sys-prefix
 jupyter-nbextension enable rise --py --sys-prefix

Usage
-----

Run the jupyter notebook server and load the notebook by:
 jupyter notebook demo.ipynp
