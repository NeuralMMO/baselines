#!/bin/bash

python -u -m train \
--runs-dir=/fsx/proj-nmmo/runs/ \
"${@}"
