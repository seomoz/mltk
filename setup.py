
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension(
        "mltk.aptagger",
        sources=['mltk/aptagger.pyx'],
        extra_compile_args=['-std=c++0x'],
        language="c++"),
    Extension(
        "mltk.np_chunker",
        sources=['mltk/np_chunker.pyx'],
        extra_compile_args=['-std=c++0x'],
        language="c++")
]

setup(name='mltk',
    version='0.0',
    description='Moz Language Tool Kit',
    author='Moz Data Science',
    packages=['mltk'],
    package_dir={'mltk': 'mltk'},
    package_data={'mltk': ['models/*']},
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)

