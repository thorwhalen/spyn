from setuptools import setup, find_packages

p = dict(version="0.0.2",
         root_url='https://github.com/thorwhalen',
         name='spyn')

extras = dict(
    description='Tools to transform data into "model-less" operable inference objects.',
    author='Thor Whalen',
    license='MIT',
    keywords=['machine learning', 'artificial intelligence',
              'AI', 'ML', 'probabilistic inference', 'Bayesian'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Either
        # "3 - Alpha",
        # "4 - Beta" or
        # "5 - Production/Stable" as the current state.
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=[  # to get list, can use imports_for.third_party (py2store.ext.module_imports)
        'matplotlib',
        'numpy',
        'pandas'
    ]
)

assert {'root_url', 'name', 'version'}.issubset(p.keys())


def readme():
    try:
        with open('README.md') as f:
            return f.read()
    except:
        return ""


ujoin = lambda *args: '/'.join(args)

assert not p['root_url'].endswith('/'), f"root_url should not end with /: {p['root_url']}"

attrs = dict(name=p['name'],
             version=f"{p['version']}",
             long_description=readme(),
             long_description_content_type="text/markdown",
             url=f"{p['root_url']}/{p['name']}",
             packages=find_packages(),
             include_package_data=True,
             zip_safe=False,
             download_url=f"{p['root_url']}/{p['name']}/archive/v{p['version']}.zip",
             )

for k in extras:
    attrs[k] = extras[k]

setup(**attrs)
