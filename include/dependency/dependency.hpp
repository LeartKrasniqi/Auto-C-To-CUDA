/* Header file for dependency tests */

#ifndef DEP_TESTS
#define DEP_TESTS
#include "rose.h"
#include "../loop_attr.hpp"

/* Function that determines whether dependencies exist within the loop nest
   
   Input: Loop nest
   Output: 0 if no dependency exists, 1 if dependencies MAY exist, 2 if some expression is too complicated for analysis (so we skip the loop)

   This function calls dependencyTests()..
   If the loop has NO dependencies, then it can be run in parallel safely.
   Otherwise, some transformations have to be made to the loop.
*/
int dependencyExists(SgForStatement *loop_nest);



/* Function that calls each of the dependency tests

   Input: Read references, Write references, Loop attributes, Test name  
   Output: 0 if no dependency exists, 1 if dependencies MAY exists, 2 if some expression is too complicated
  
   This function calls dependency test functions
*/
int dependencyTests(std::list<std::vector<SgExpression*>> reads, std::list<std::vector<SgExpression*>> writes, LoopNestAttribute *attrbute, std::string test_name);




/* Function that performs Zero-Iter-Variable Test on two array subscript expressions

   Input: Vectors of array subscripts for first and second array reference
   Output: 0 if no integer solutions (i.e. NO DEPENDENCIES), 1 if there are integer solutions (i.e. INCONCLUSIVE) , 2 if skip loop 

   If two array subscripts are constants and are different, there is no possible way that there are data dependencies.
   
   Ex: a[1][i] = a[2][i]

   Since a[1] and a[2] are two DIFFERENT data locations, there is no data dependence.

*/
int ZIVTest(std::vector<SgExpression*> ref1, std::vector<SgExpression*> ref2);




/* Function to extract coefficients from expressions 
   
   Input: Expression, fixed-size vector to hold coeffs, variable to hold result, Loop nest attribute
   Output: Coeffs/Result are written to the vector/var which are passed by reference, and true is returned if successful, false if not

   Ex:
   	iter_vec = [i,j]
  	expr = 3*i + 4*k + 1  -->  coeff_arr = [3,4], result = 1
*/
bool extractCoeff(SgExpression *expr, std::vector<int> &coeff_arr, int &result, LoopNestAttribute *attr);




/* Function that performs GCD test to find out if array references are dependent 
   
   Input: Expressions for the first array reference and the second array reference, Loop nest attribute
   Output: 0 if no integer solutions (i.e. NO DEPENDENCIES), 1 if there are integer solutions (i.e. INCONCLUSIVE) , 2 if skip loop 
   
   This test is a simple dep. test, but returns inconclusive most of the time.
   If this test does not return 0, a series of other tests are performed.
*/
int GCDTest(std::vector<SgExpression*> ref1, std::vector<SgExpression*> ref2, LoopNestAttribute *attr);




/* Function that uses the Euclidean Algorithm to find the GCD of a two INTs
 
   Input: Two INTs
   Output: Greatest common divisor of the two inputs 

   This is used in the GCDTest() function. 
*/
int euclidGCD(int a, int b);




/* Function that performs the Banerjee Test to determine if array references are dependent

   Input: Expressions for first and second array references, Loop nest attribute
   Output: 0 if no integer solutions (i.e. NO DEPENDENCIES), 1 if there are integer solutions (i.e. INCONCLUSIVE) , 2 if skip loop 


*/
int banerjeeTest(std::vector<SgExpression*> ref1, std::vector<SgExpression*> ref2, LoopNestAttribute *attr);




/* Functions to return positive/negative part of a number 
 
   Input: Number
   Output: Positive/negative part of that number
*/
int pos(int a);
int neg(int a);

#endif
