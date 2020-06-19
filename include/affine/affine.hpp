/* Header file for testing for affine loops */

#ifndef AFFINE
#define AFFINE
#include "rose.h"
#include "../loop_attr.hpp"

/* Function to test whether given loop nest is affine
 
   Input: Perfectly nested loop nest
   Output: true if nest is affine, false otherwise
 
   An affine loop contains only affine array accesses.
   An array access is affine iff references to the index variables are linear:
   	ex:
		a[i+10] = a[i];     <-- This is affine
		a[b[i]] = a[c[i]];  <-- This is NOT affine

   This function uses SageInterface::constantFolding() in addition to the below functions 
*/
bool affineTest(SgForStatement *body);



/* Helper function to perform additional constant folding after normalization step 

   Input: Array index expression
   Output: Simplified expression if successful, original expression otherwise. 

   Expressions that are supported:
   	     A*i + B		--> No possible folding
	     A*i + B +/- C  	--> A*i + (B +/- C)
	     D*(A*i + B)	--> (D*A)*i + (D*B)
	     D*(A*i + B) +/- C  --> (D*A)*i + (D*B +/- C)

*/
SgExpression * indexConstantFolding(SgExpression *expr);



/* Function to test whether a single expression is affine 
   
   Input: Expression and loop nest 
   Output: true if exression is affine, false otherwise

   This function is ultimately used by affineTest().
*/ 
bool affineTestExpr(SgExpression *expr, SgForStatement *loop_nest);

#endif
