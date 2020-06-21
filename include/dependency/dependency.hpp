/* Header file for dependency tests */

#ifndef DEP_TESTS
#define DEP_TESTS
#include "rose.h"
#include "../loop_attr.hpp"

/* Function that determines whether dependencies exist within the loop nest
   
   Input: Loop nest
   Output: true if dependencies MAY exist, false if they do not

   This function calls on various dependency tests.
   If the loop has NO dependencies, then it can be run in parallel safely.
   Otherwise, some transformations have to be made to the loop.
*/

bool dependencyExists(SgForStatement *loop_nest);


/* Function that performs GCD test to find out if array references are dependent 
   
   Input: Expressions for the write (i.e. lhs) array index and the read (i.e. rhs) array index
   Output: true if there are integer solutions (i.e. INCONCLUSIVE) , false if no integer solutions (i.e. NO DEPENDENCIES)

   This test is the simplest dep. test, but returns inconclusive most of the time.
   If this test does not return true, a series of other tests are performed.
*/
bool gcdTest(SgExpression *write, SgExpression *read);




#endif
