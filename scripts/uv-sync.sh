#!/bin/bash

# Install every dependencies and ensure src/optimum/rbln/__version__.py is generated
uv sync \
    --frozen \
    --all-groups \
    --all-extras \
    --reinstall-package optimum-rbln
