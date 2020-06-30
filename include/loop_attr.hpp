/* Header File for the LoopNestAttribute Class */

#ifndef LOOP_ATTR
#define LOOP_ATTR

#include "rose.h"

/* Class for setting attributes of loop nest */
class LoopNestAttribute : public AstAttribute {
	public:
		LoopNestAttribute(int s, bool f) {this->size = s; this->flag = f;}
		virtual LoopNestAttribute * copy() const override {return new LoopNestAttribute(*this);}
		virtual std::string attribute_class_name() const override {return "LoopNestAttribute";}
		
		/* Getters */
		int get_nest_size() {return size;}
		bool get_nest_flag() {return flag;}
		std::list<std::string> get_iter_vec() {return iter_vec;}
		std::list<SgExpression*> get_bound_vec() {return bound_vec;}
		std::list<std::string> get_symb_vec() {return symb_vec;}
		std::list<std::list<std::list<std::vector<SgExpression*>>>> get_arr_dep_info() {return arr_dep_info;}
		
		/* Setters */
		void set_nest_flag(bool new_flag) {flag = new_flag;}
		void set_iter_vec(std::list<std::string> vec) {iter_vec = vec;}
		void set_bound_vec(std::list<SgExpression*> vec) {bound_vec = vec;}
		void set_symb_vec(std::list<std::string> vec) {symb_vec = vec;}
		void set_arr_dep_info(std::list<std::list<std::list<std::vector<SgExpression*>>>> info) {arr_dep_info = info;}

	private:
		int size;
		bool flag;
		std::list<std::string> iter_vec;
		std::list<SgExpression*> bound_vec;
		std::list<std::string> symb_vec;
		std::list<std::list<std::list<std::vector<SgExpression*>>>> arr_dep_info;

};
#endif
