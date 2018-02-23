#!/bin/sh

git add .
git commit -m "changes"
git push
sudo pip install -U git+https://github.com/yunhuiguo/tensorpack.git

