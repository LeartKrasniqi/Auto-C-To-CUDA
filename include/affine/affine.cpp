/* Implementation of affine test functions */
#include "affine.hpp"
#include <algorithm>

/* Test whether loop nest is affine */
bool affineTest(SgForStatement *loop_nest)
{
	/* Obtain the attributes of the loop */
	LoopNestAttribute *attr = dynamic_cast<LoopNestAttribute*>(loop_nest->getAttribute("LoopNestInfo"));
	std::list<std::string> loop_iter_vec = attr->get_iter_vec();	
	
	/* Obtain the body of the loop nest (assuming the nest is perfectly nested) */
	SgStatement *body = loop_nest->get_loop_body();
	while(isSgForStatement(body))
		body = isSgForStatement(body)->get_loop_body();

	/* Get the array references */
	Rose_STL_Container<SgNode*> array_refs = NodeQuery::querySubTree(body, V_SgPntrArrRefExp); 
	
	/* Loop through each reference and check if affine */
	Rose_STL_Container<SgNode*>::iterator arr_iter;
	for(arr_iter = array_refs.begin(); arr_iter != array_refs.end(); arr_iter++)
	{
		SgExpression *expr = isSgPntrArrRefExp(*arr_iter)->get_rhs_operand();
		
		if(expr)
		{
			/* Perform any constant folding */
			SgExpression *new_expr = indexConstantFolding(expr);
			
			/* Set the new array index expression */
			isSgPntrArrRefExp(*arr_iter)->set_rhs_operand(new_expr);
			
			/* Perform built-in constant folding now */
			SageInterface::constantFolding(*arr_iter);
		
			/* Find references to variables in the expression */
			Rose_STL_Container<SgNode*> var_refs = NodeQuery::querySubTree(new_expr, V_SgVarRefExp);
			
			/* Get only unique references */
			Rose_STL_Container<SgNode*>::iterator var_it;
			std::list<std::string> var_ref_strings;
			for(var_it = var_refs.begin(); var_it != var_refs.end(); var_it++)
			{
				std::string index_string = isSgVarRefExp(*var_it)->get_symbol()->get_name().getString();
				
				if( std::find(var_ref_strings.begin(), var_ref_strings.end(), index_string) != var_ref_strings.end() )
					continue;
				else
					var_ref_strings.push_back(index_string);
			}

			/* Loop thru the references in the expression and count the number of index variables being referenced */
			int num_refs = 0;
			std::list<std::string>::iterator var_ref_it;
			for(var_ref_it = var_ref_strings.begin(); var_ref_it != var_ref_strings.end(); var_ref_it++)
			{
				std::string index_string = *var_ref_it;
				
				/* Check to see if expression contains reference to an index variable */
				if( std::find(loop_iter_vec.begin(), loop_iter_vec.end(), index_string) != loop_iter_vec.end() )
					num_refs += 1;
			}

			/* If only one index variable is being referenced, the expression could potentially be affine */
			if(num_refs == 1)
			{
				if(!affineTestExpr(new_expr))
					return false;
			
			}
			/* If more than one index variable is being referenced, the expression is not affine */
			else if(num_refs > 1)
				return false;
			
		}
	}


	/* If we get here, all expressions within the body are affine, so the loop nest is affine */
	return true;
}


/* Perform some additional constant folding in the array expression */
SgExpression * indexConstantFolding(SgExpression *expr)
{
	SgBinaryOp *op = isSgBinaryOp(expr);
	
	if(!op)
		return expr;
	
	/* Four cases we deal with:
	     A*i + B		--> No possible folding
	     A*i + B +/- C  	--> A*i + (B +/- C)
	     D*(A*i + B)	--> (D*A)*i + (D*B)
	     D*(A*i + B) +/- C  --> (D*A)*i + (D*B +/- C)
	*/

	/* Will be the new expression */
	SgExpression *new_expr;
	
	/* Cases I and IV */
	if( (isSgAddOp(op) || isSgSubtractOp(op)) && isSgMultiplyOp(op->get_lhs_operand()) )
	{
		/* If mult_op->rhs() is not an add_op, we are in case I, so just return */
		if(!isSgAddOp(isSgBinaryOp(op->get_lhs_operand())->get_rhs_operand()))
			return expr;

		/* Obtain A, i, B, C, and D */
	 	SgExpression *A, *i, *B, *C, *D;	
		
		/* Keeping these just in case since I needed to add error checking and it looks less fancy now :( */
		#if 0
		SgExpression *A = isSgBinaryOp(
				  isSgBinaryOp(
				  isSgBinaryOp(
					op->get_lhs_operand())
				        ->get_rhs_operand())
				        ->get_lhs_operand())
			                ->get_lhs_operand();
		SgExpression *i = isSgBinaryOp(
				  isSgBinaryOp(
				  isSgBinaryOp(	
					op->get_lhs_operand())
				  	->get_rhs_operand())
					->get_lhs_operand())
					->get_rhs_operand();
		SgExpression *B = isSgBinaryOp(
				  isSgBinaryOp(
					op->get_lhs_operand())
					->get_rhs_operand())
					->get_rhs_operand();
		#endif
		
		SgBinaryOp *A_intermed = isSgBinaryOp(isSgBinaryOp(op->get_lhs_operand())->get_rhs_operand());
		if(A_intermed)
			if( (A_intermed = isSgBinaryOp(A_intermed->get_lhs_operand())) )
				A = A_intermed->get_lhs_operand();
			else
				return expr;
		else
			return expr;


		SgBinaryOp *i_intermed = isSgBinaryOp(isSgBinaryOp(op->get_lhs_operand())->get_rhs_operand());
		if(i_intermed)
			if( (i_intermed = isSgBinaryOp(i_intermed->get_lhs_operand())) )
				i = i_intermed->get_rhs_operand();
			else
				return expr;
		else
			return expr;

	
		SgBinaryOp *B_intermed = isSgBinaryOp(isSgBinaryOp(op->get_lhs_operand())->get_rhs_operand());
		if(B_intermed)
			B = B_intermed->get_rhs_operand();
		else
			return expr;

		
		C = op->get_rhs_operand();
		D = isSgBinaryOp(op->get_lhs_operand())->get_lhs_operand();

		/* Perform distribution */
		SgExpression *slope = SageBuilder::buildMultiplyOp( SageBuilder::buildMultiplyOp(D, A) , i);
		SgExpression *intermed = SageBuilder::buildMultiplyOp(D, B);

		SgExpression *y_inter;
		if(isSgAddOp(op))
			y_inter = SageBuilder::buildAddOp(intermed, C);
		else
			y_inter = SageBuilder::buildSubtractOp(intermed, C);

		/* Create the new expression */
		new_expr = SageBuilder::buildAddOp(slope, y_inter);
 
	}
	/* Case II */
	else if( (isSgAddOp(op) || isSgSubtractOp(op)) && isSgAddOp(op->get_lhs_operand()) )
	{
		/* This should be A*i */
		SgExpression *inner_lhs = isSgBinaryOp(op->get_lhs_operand())->get_lhs_operand();

		/* This should be B */
		SgExpression *inner_rhs = isSgBinaryOp(op->get_lhs_operand())->get_rhs_operand();

		/* This should be C */
		SgExpression *outer_rhs = op->get_rhs_operand();

		/* Holds intermediate calculation */
		SgExpression *intermed;
		if(isSgAddOp(op))
			intermed = SageBuilder::buildAddOp(inner_rhs, outer_rhs);
		else
			intermed = SageBuilder::buildSubtractOp(inner_rhs, outer_rhs);

		/* Create the new expression */
		new_expr = SageBuilder::buildAddOp(inner_lhs, intermed);

	}
	/* Case III */
	else if(isSgMultiplyOp(op))
	{	
		/* This should be (A*i+B) */
		SgBinaryOp *inner_rhs = isSgBinaryOp(op->get_rhs_operand());
		if(!inner_rhs)
			return expr;

		/* Obtain A, i, B, and D */
		SgExpression *A, *i, *B, *D;
		
		if(isSgBinaryOp(inner_rhs->get_lhs_operand()))
			A = isSgBinaryOp(inner_rhs->get_lhs_operand())->get_lhs_operand();
		else
			return expr;

		if(isSgBinaryOp(inner_rhs->get_lhs_operand()))
			i = isSgBinaryOp(inner_rhs->get_lhs_operand())->get_rhs_operand();
		else
			return expr;

		B = inner_rhs->get_rhs_operand();
		D = op->get_lhs_operand();

		/* Distributive property time */
		SgExpression *slope = SageBuilder::buildMultiplyOp( SageBuilder::buildMultiplyOp(D, A) , i);
		SgExpression *y_inter = SageBuilder::buildMultiplyOp(D, B);
		
		/* Create the new expr */
		new_expr = SageBuilder::buildAddOp(slope, y_inter);
	
	}
	else
		return expr;

	
	return new_expr;
}


/* Perform affine check on single expression */
bool affineTestExpr(SgExpression *expr)
{
	/* Expression is affine iff is it of form A*i +/- B, where i is the index */
	
	/* If exp is a constant, return true */
	if(isSgValueExp(expr))
		return true;

	if(isSgBinaryOp(expr))
	{
		SgExpression *lhs = isSgBinaryOp(expr)->get_lhs_operand();
		SgExpression *rhs = isSgBinaryOp(expr)->get_rhs_operand();
		
		/* An add/sub op is affine iff both lhs and rhs are affine */
		if( isSgAddOp(expr) || isSgSubtractOp(expr) )
			return ( affineTestExpr(lhs) && affineTestExpr(rhs) );
		
		/* A multiply op is affine iff it is A*i or i*A */
		if( isSgMultiplyOp(expr) )
		{
			/* Case I: A*i --> Affine */
			if(isSgValueExp(lhs) && isSgVarRefExp(rhs))
				return true;

			/* Case II: i*A --> Affine */
			if(isSgVarRefExp(lhs) && isSgValueExp(rhs))
				return true;

			/* Otherwise, check to see if both lhs and rhs are affine */
			return ( affineTestExpr(lhs) && affineTestExpr(rhs) );
		}
	}

	/* If the expression is just a single variable, this is equivalent to 1*i, so it should be affine as long as i is of integer type */
	if(isSgVarRefExp(expr))
		return expr->get_type()->isIntegerType();

	
	/* If we get here, the expression did not match any of our accepted affine forms */
	return false;
}

