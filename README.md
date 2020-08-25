# Auto-C-To-CUDA
> An automatic C to CUDA transcompiler built with the [ROSE Compiler](http://rosecompiler.org).  This system is capable of handling some `while` loops and some imperfectly nested `for` loops.  It makes use of loop fission and extended cycle shrinking to extract parallelism out of certain loops, if dependency tests allow for the transformations.  The system was created as a Master's Thesis project.  
<hr>

## Table of Contents
* [How To Build/Run](#how-to-run)
* [Directories and Files](#directories)
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

## <a name="directories"></a>Directories and Files
Here's a bit of info about the project directories and files:
* `benchmarks`: Series of benchmark C programs and their transcompiled CUDA versions
* `include`: The methods required for the analysis and transformations
  * `affine`: Checks whether loop nest is affine
  * `dependency`: Performs dependency analysis 
  * `kernel`: CUDA code generation
  * `normalize`: Transform loop nests into a standardized/normalized form
  * `parallel`: Attempt to extract parallelism from loops with dependencies
  * `preprocess`: Convert `while` and imperfectly nested `for` loops into perfectly nested loops
  * `loop_attr.hpp`: Useful ROSE attributes
* `tests`: A bunch of unit tests for each analysis/transformation stage
* `translate.cpp`: Main driver program

## <a name="system-description"></a>System Description
Will be filled in soon...
