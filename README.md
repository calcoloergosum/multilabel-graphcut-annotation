# Multiway Minimum Graph Cut based Semantic Segmentation Annotation Tool

![build](https://github.com/studentofkyoto/multilabel-graphcut-annotation/actions/workflows/pythonpackage.yml/badge.svg)

![Usage demo](asset/demo.gif)

## Use

```bash
$ python script/annotate.py $DIR_PATH
```

All files under `$DIR_PATH$` is searched.

## Note

This program depends on `gco-3.0` and `pygco`. Please note that `gco-3.0` comes with [research-only license](https://vision.cs.uwaterloo.ca/code/).

## References

[1] Efficient Approximate Energy Minimization via Graph Cuts.
    Y. Boykov, O. Veksler, R.Zabih. IEEE TPAMI, 20(12):1222-1239, Nov 2001.

[2] What Energy Functions can be Minimized via Graph Cuts?
    V. Kolmogorov, R.Zabih. IEEE TPAMI, 26(2):147-159, Feb 2004. 

[3] An Experimental Comparison of Min-Cut/Max-Flow Algorithms for 
    Energy Minimization in Vision. Y. Boykov, V. Kolmogorov. 
    IEEE TPAMI, 26(9):1124-1137, Sep 2004.

[4] Fast Approximate Energy Minimization with Label Costs. 
        A. Delong, A. Osokin, H. N. Isack, Y. Boykov. In CVPR, June 2010.
 