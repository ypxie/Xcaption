#!/usr/bin/env bash
THEANO_FLAGS=device=gpu,floatX=float32 python ./evaluate_coco.py