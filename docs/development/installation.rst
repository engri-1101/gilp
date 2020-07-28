Installation
============

To develop and run tests on gilp, first download the source code in the desired
directory.

.. code-block:: bash

    git clone https://github.com/henryrobbins/gilp

Next, cd into the gilp directory and create a Python virtual enviroment called
:code:`env_name`

.. code-block:: bash
    :linenos:

    cd gilp
    python -m venv env_name

Activate the virtual enviroment.

.. code-block:: bash

    source env_name/bin/activate

Run the following in the virtual enviroment. The :code:`-e` flag lets you make
adjustments to the source code and see changes without re-installing. The
:code:`[dev]` installs necessary dependencies for developing and testing.

.. code-block:: bash

    pip install -e .[dev]

To run tests and see coverage, run the following in the virtual enviroment.

.. code-block:: bash
    :linenos:

    coverage run -m pytest
    coverage report --include=gilp/*

Next, we will provide an extensive overview of the gilp package and the
contained modules.