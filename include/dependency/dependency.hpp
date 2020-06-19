/* Header file for dependency tests */

#ifndef DEP_TESTS
#define DEP_TESTS
#include "rose.h"
#include "../loop_attr.hpp"


/* Function that performs GCD test to find out if array references are dependent 
   
Input: Expressions for the write (i.e. lhs) array index and the read (i.e. rhs) array index
   Output: true if dependency exists, false if inconclusive

   This test is the simplest dep. test, but returns inconclusive most of the time.
   If this test does not return true, a series of other tests are performed.
*/
bool gcdTest(SgExpression *write, SgExpression *read)




#endif
