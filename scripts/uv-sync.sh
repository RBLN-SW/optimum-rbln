#!/bin/bash

# Install every dependencies and ensure src/optimum/rbln/__version__.py is generated
uv sync --active \
    --frozen \
    --all-groups \
    --all-extras \
    --reinstall-package optimum-rbln
