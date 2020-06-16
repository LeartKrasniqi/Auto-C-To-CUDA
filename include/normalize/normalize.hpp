/* Header file for loop normalization */

#ifndef LOOP_NORMALIZATION
#define LOOP_NORMALIZATION
#include "rose.h"

/* Function to normalize a loop nest 
  	Input: Loop nest
  	Output: true if normalization succeeded, false otherwise
 
  	This function ultimately calls normalizeLoop() for each loop in the nest
*/
bool normalizeLoopNest(SgForStatement *loop_nest);



/* Function to normalize a single loop 
 	Input: Single loop
	Output: true if normalization succeeded, false otherwise 
	
	Handles loops of the form:  for(INIT; TEST; INC)
	where
		INIT can have the form:
			int i = INTEGER_CONSTANT 
		      	i = INTEGER_CONSTANT
		
		TEST can have the form:
			i OP INTEGER_BOUND, where OP can be <,>,<=,>=,=,!= 	
		
		INC can have the form:
			i++, i--, ++i, --i
			i+= INTEGER_CONSTANT, i-=INTEGER_CONSTANT

	This function makes a call to SageInterface::forLoopNormalization() as an initial step			
*/	
bool normalizeLoop(SgForStatement *loop);

#endif
