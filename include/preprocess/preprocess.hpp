/* Header file for preprocessing the input code */

#ifndef INPUT_PREPROC
#define INTUT_PREPROC
#include "rose.h"
#include "../parallel/parallel.hpp"

/* Function to convert while loops into for loops 
 
   Input: 
   Output: While loop if conversion is successful, NULL otherwise

   This handles cases of the following form:
       i = a;
       while(i<n)
       {
           <statements not involving updates to i or n>
	   i = i+1;
       }

   So, the index variable must be set RIGHT BEFORE the loop, the inner-statements must NOT INVOLVE UPDATES to iter/bound, and the INCREMENT IS THE LAST STATEMENT. 
   Makes a call to SageInterface::getPreviousStatement() and SageInterface::isLastStatement()
*/
SgStatement * convertWhileToFor(SgWhileStmt *loop_nest);


#endif
