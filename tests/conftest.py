# -*- coding: utf-8 -*-
"""Torna os modulos do TCC importaveis a partir da pasta de testes."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
