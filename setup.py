from distutils.core import setup
from distutils.extension import Extension
import os.path

import numpy
numpy_path = os.path.join(numpy.__path__[0], 'core', 'include')

setup(
    name = "rsvd",
    ext_modules = [Extension("rsvd/rsvd", ["rsvd/rsvd.c"],
                             include_dirs=[numpy_path],
                             extra_link_args=["-O3","-ffast-math"]
                             ),
                  ],
    version = "0.2",
    description="A regularized SVD solver for partial matrices",
    author='Peter Prettenhofer',
    author_email='peter.prettenhofer@gmail.com',
    url="http://code.google.com/p/pyrsvd/",
    scripts = ["rsvd_train","rsvd_predict","rsvd_lc"],
    packages=['rsvd'],
)
