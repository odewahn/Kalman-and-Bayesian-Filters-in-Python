#! /bin/bash
cd ..
git checkout gh-pages
git checkout master Kalman_and_Bayesian_Filters_in_Python.pdf
git checkout README.md
git add Kalman_and_Bayesian_Filters_in_Python.pdf
git add README.md
git commit -m 'updating PDF'
git push
git checkout master
cd pdf