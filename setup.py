from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="sptx_ccf_registration",
    version="0.1.0",
    description="Registration pipeline of SPTx data to the Allen CCF. Includes"
    "preprocessing, registration, and posthoc analysis steps.",
    author="Mike Huang, Min Chen",
    author_email="mike.huang@alleninstitute.org",
    url="https://github.com/AllenInstitute/sptx_ccf_registration",
    license=license,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"sptx_ccf_registration.metrics_dashboard": ["ccf_merfish_metrics/requirements.txt", "ccf_merfish_metrics/app.py", "ccf_merfish_metrics/Procfile", "ccf_merfish_metrics/templates/index.html"]},
    include_package_data=True,
    install_requires=required,
)
