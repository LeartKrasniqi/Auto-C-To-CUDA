#include "rose.h"
#include <iostream>
#include "./include/normalize/normalize.hpp"
#define DEBUG 1

int main(int argc, char **argv)
{
	ROSE_INITIALIZE;
	SgProject *project = frontend(argc, argv);
	
	/* Get all function definitions */
	Rose_STL_Container<SgNode*> functions = NodeQuery::querySubTree(project, V_SgFunctionDefinition);
	Rose_STL_Container<SgNode*>::const_iterator func_iter = functions.begin();
	
	/* Loop through each function definition */
	while(func_iter != functions.end())
	{
		/* Get the actual definition node */
		SgFunctionDefinition* defn = isSgFunctionDefinition(*func_iter);

		/* Query for the for loops */
		Rose_STL_Container<SgNode*> forLoops = NodeQuery::querySubTree(defn,V_SgForStatement);
		
		/* Loop thru the for loops */
		Rose_STL_Container<SgNode*>::const_iterator for_iter = forLoops.begin();
		//while(for_iter != forLoops.end())
		for( ; for_iter != forLoops.end(); for_iter++)
		{
			/* Get the loop itself */
			SgForStatement* loop = isSgForStatement(*for_iter);
			
			/* Perform normalization */
			if(!normalizeLoop(loop))
			{
				std::cout << "Loop skipped" << std::endl;
				continue;
			}


			/* Perform other things ... */

		}

		
		#if DEBUG
		std::cout << defn->unparseToString() << std::endl;
		#endif
		

		func_iter++;
	}

	return 0;
}
