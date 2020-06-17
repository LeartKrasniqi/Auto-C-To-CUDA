/* Implementation of affine test functions */
#include "affine.hpp"

/* Test whether loop nest is affine */
bool affineTest(SgForStatement *loop_nest)
{
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

			/* Check if new_expr is affine and return false immediately if not */
			if(!affineTestExpr(new_expr))
				return false;
		}
	}


	/* If we get here, then the loop nest is affine */
	return true;
}


/* Perform some additional constant folding in the array expression */
SgExpression * indexConstantFolding(SgExpression *expr)
{
	SgBinaryOp *op = isSgBinaryOp(expr);

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

		SgExpression *C = op->get_rhs_operand();
		SgExpression *D = isSgBinaryOp(op->get_lhs_operand())->get_lhs_operand();

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

		/* Obtain A, i, B, and D */
		SgExpression *A = isSgBinaryOp(inner_rhs->get_lhs_operand())->get_lhs_operand();
		SgExpression *i = isSgBinaryOp(inner_rhs->get_lhs_operand())->get_rhs_operand();
		SgExpression *B = inner_rhs->get_rhs_operand();
		SgExpression *D = op->get_lhs_operand();

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
	/* TODO: Implement */

	return true;
}
