vismatch
========

Vis(ion)Match(ers) is a unified API for 50+ image matching models with a consistent interface.


.. image:: https://img.shields.io/badge/GitHub-vismatch-blue?logo=github
   :target: https://github.com/gmberton/vismatch
   :alt: GitHub

.. image:: https://img.shields.io/badge/Models-HuggingFace-yellow
   :target: https://huggingface.co/vismatch
   :alt: GitHub

.. image:: https://img.shields.io/badge/Downloads-Tracker-blue
   :target: https://gmberton.github.io/vismatch-downloads-tracker/downloads_per_day.html
   :alt: Downloads Tracker

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/gmberton/vismatch/blob/main/demo.ipynb
   :alt: Open In Colab

.. image:: https://img.shields.io/pypi/v/vismatch
   :target: https://pypi.org/project/vismatch/
   :alt: PyPI

.. code-block:: python

   from vismatch import get_matcher

   matcher = get_matcher("superpoint-lightglue", device="cuda")
   img0 = matcher.load_image("img0.jpg", resize=512)
   img1 = matcher.load_image("img1.jpg", resize=512)
   result = matcher(img0, img1)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   model_details
   contributing

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/vismatch
   api/vismatch.base_matcher
   api/vismatch.utils
   api/vismatch.viz
   api/vismatch.im_models

.. toctree::
   :maxdepth: 2
   :caption: Model Specific Info

   model_specific/matchanything

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
