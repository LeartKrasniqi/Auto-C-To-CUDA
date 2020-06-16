#include "rose.h"
#include <iostream>
#include "./include/normalize/normalize.hpp"
#define DEBUG 1

/* Class for setting attributes of loop nest */
class LoopNestAttribute : public AstAttribute {
	public:
		LoopNestAttribute(int s, bool f) {this->size = s; this->flag = f;}
		virtual LoopNestAttribute * copy() const override {return new LoopNestAttribute(*this);}
		virtual std::string attribute_class_name() const override {return "LoopNestAttribute";}
		int get_nest_size() {return size;}
		bool get_nest_flag() {return flag;}
		void set_nest_flag(bool new_flag) {flag = new_flag;}
	private:
		int size;
		bool flag;
};


int main(int argc, char **argv)
{
	ROSE_INITIALIZE;
	SgProject *project = frontend(argc, argv);
	
	/* Get all function definitions */
	Rose_STL_Container<SgNode*> functions = NodeQuery::querySubTree(project, V_SgFunctionDefinition);
	Rose_STL_Container<SgNode*>::const_iterator func_iter = functions.begin();
	
	/* Will hold each of the loop nests */
	std::list<SgForStatement*> loop_nest_list;
	
	/* Loop through each function definition */
	while(func_iter != functions.end())
	{
		/* Get the actual definition node */
		SgFunctionDefinition* defn = isSgFunctionDefinition(*func_iter);

		/* Query for the for loops */
		Rose_STL_Container<SgNode*> forLoops = NodeQuery::querySubTree(defn,V_SgForStatement);
		
		/* Obtain the loop nests */
		Rose_STL_Container<SgNode*>::const_iterator for_iter = forLoops.begin();
		while(for_iter != forLoops.end())
		{
			/* Get the outer most loop */
			SgForStatement* loop_nest = isSgForStatement(*for_iter);
			
			/* Find loop nest size */
			Rose_STL_Container<SgNode*> inner_loops = NodeQuery::querySubTree(loop_nest, V_SgForStatement);
			int nest_size = inner_loops.size();

			/* Set attributes (applied to outermost loop) */
			loop_nest->setAttribute("LoopNestInfo", new LoopNestAttribute(nest_size, true));
			
			/* Append loop_nest to list of loop nests */
			loop_nest_list.push_back(loop_nest);
			
			/* Increment to get to next loop_nest */
			for_iter += nest_size;
		}

			
		/* Iterate through the loop nests */
		std::list<SgForStatement*>::iterator nest_iter; 
		for(nest_iter = loop_nest_list.begin(); nest_iter != loop_nest_list.end(); nest_iter++)
		{	
			SgForStatement *loop_nest = *nest_iter;
			
			/* TODO: Check if loop nest is perfectly nested */
			

			/* Obtain the attribute of the nest */
			LoopNestAttribute *attr = dynamic_cast<LoopNestAttribute*>(loop_nest->getAttribute("LoopNestInfo"));
			
			
			/* Perform normalization */
			if(!normalizeLoopNest(loop_nest))
			{
				std::cout << "Loop Nest Skipped" << std::endl;
				attr->set_nest_flag(false);
				continue;
			}

			if(attr->get_nest_flag())
				std::cout << loop_nest->unparseToString() << std::endl;

			
		}


			/* Perform other things ... */

		#if DEBUG
		std::cout << defn->unparseToString() << std::endl;	
		#endif
		

		func_iter++;
	}

	return 0;
}
