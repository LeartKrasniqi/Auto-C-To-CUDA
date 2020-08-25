# Auto-C-To-CUDA
> An automatic C to CUDA transcompiler built with the [ROSE Compiler](http://rosecompiler.org).  This system is capable of handling some `while` loops and some imperfectly nested `for` loops.  It makes use of loop fission and extended cycle shrinking to extract parallelism out of certain loops, if dependency tests allow for the transformations.  The system was created as a Master's Thesis project.  
<hr>

## Table of Contents
* [How To Build/Run](#how-to-run)
* [Directories](#directories)
* [System Description](#system-description)

## <a name="how-to-run"></a>How To Build/Run
1. [Install ROSE](https://github.com/rose-compiler/rose/wiki/How-to-Set-Up-ROSE)
2. Set path variable `ROSE_INSTALL` to `/path/to/rose/rose/install`
3. Clone this repo
4. Run `make`
5. Execute:
```
./translate.out [input.c] -rose:o [output.cu]
```

## <a name="directories"></a>Directories
Each directory contains methods with detailed explanations.  We give a brief bit of info for each one here:

## <a name="system-description"></a>System Description
Will be filled in soon...
