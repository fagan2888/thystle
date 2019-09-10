#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hyss
imps = ["utils.py", "select_pixels.py"]
for ii in imps:
    exec(open(ii).read())


imgLcc = select_pixels(159)
flist = get_file_list()
clist = get_clipsL_list()
