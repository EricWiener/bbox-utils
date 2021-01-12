==========
bbox-utils
==========

bbox-utils allows you to easily convert between different bounding box formats (YOLO, XYWH, XYXY, etc.).

It's as simple to use as::

    from bbox_utils import BoundingBox

    xy1 = np.array([100, 50])
    xy2 = np.array([200, 75])
    bbox = BoundingBox.from_xyxy(xy1, xy2)

    # Get XYWH
    xy, w, h = bbox.to_xywh()

    # Get XYXY
    xy1, xy2 = bbox.to_xyxy()

    # Get YOLO
    image_dim = 640, 420
    yolo_bbox = bbox.to_yolo(image_dim)


You can install bbox-utils with PyPI: ``pip install bbox-utils``

Conversions
===========================

2D Bounding Box Conversions:
----------------------------
* List of points [top left, top right, bottom right, bottom left]
* XYWH: top left, width, height
* XYXY: top left, bottom right
* YOLO
* 3D Bounding Box Conversions You can create a 3D bounding box with either:

3D Bounding Box Conversions:
----------------------------
You can create a 3D bounding box with either:
* A center point, width, height, depth, and rotation
* The eight vertices

You can convert between the two forms and also get a triangular polygon to use for plotting triangular meshes.


Contents
========

.. toctree::
   :maxdepth: 2

   Overview <readme>
   License <license>
   Authors <authors>
   Changelog <changelog>
   Module Reference <api/modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: http://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: http://sphinx-doc.org/domains.html#the-python-domain
.. _Sphinx: http://www.sphinx-doc.org/
.. _Python: http://docs.python.org/
.. _Numpy: http://docs.scipy.org/doc/numpy
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: http://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: http://scikit-learn.org/stable
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
.. _Google style: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists
