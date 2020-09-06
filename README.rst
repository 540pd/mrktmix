========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/mrktmix/badge/?style=flat
    :target: https://readthedocs.org/projects/mrktmix
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.org/540pd/mrktmix.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/540pd/mrktmix

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/540pd/mrktmix?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/540pd/mrktmix

.. |requires| image:: https://requires.io/github/540pd/mrktmix/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/540pd/mrktmix/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/540pd/mrktmix/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/540pd/mrktmix

.. |version| image:: https://img.shields.io/pypi/v/mrktmix.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/mrktmix

.. |wheel| image:: https://img.shields.io/pypi/wheel/mrktmix.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/mrktmix

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/mrktmix.svg
    :alt: Supported versions
    :target: https://pypi.org/project/mrktmix

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/mrktmix.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/mrktmix

.. |commits-since| image:: https://img.shields.io/github/commits-since/540pd/mrktmix/v0.0.2.svg
    :alt: Commits since latest release
    :target: https://github.com/540pd/mrktmix/compare/v0.0.2...master



.. end-badges

Market Mix Modeling

* Free software: GNU Lesser General Public License v3 or later (LGPLv3+)

Installation
============

::

    pip install mrktmix

You can also install the in-development version with::

    pip install https://github.com/540pd/mrktmix/archive/master.zip


Documentation
=============


https://market_mix_model.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
