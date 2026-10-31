import solvers.«93.py.lean.d».Rational
import ProjectEulerSolutions.Lib.RationalChoices
import ProjectEulerSolutions.Lib.RationalHeap

namespace solvers.«93».pair_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
set_option Elab.async false

abbrev program := Solvers.P93Ast.module
abbrev number := Heap.number
abbrev object := Heap.object

structure Ready (env : HeapCalls.Env) (a b : Pair) : Prop where
  numeratorA : env "numerator_a" = some (number a.1)
  denominatorA : env "denominator_a" = some (number a.2)
  numeratorB : env "numerator_b" = some (number b.1)
  denominatorB : env "denominator_b" = some (number b.2)
  denominator : env "denominator" = some (number (a.2 * b.2))

abbrev na : Expr := .name "numerator_a"
abbrev da : Expr := .name "denominator_a"
abbrev nb : Expr := .name "numerator_b"
abbrev db : Expr := .name "denominator_b"
abbrev denominator : Expr := .name "denominator"
abbrev crossA : Expr := .binOp na .mult db
abbrev crossB : Expr := .binOp nb .mult da
abbrev addExpr : Expr := .call (.name "rational") [.binOp crossA .add crossB, denominator]
abbrev subExpr : Expr := .call (.name "rational") [.binOp crossA .sub crossB, denominator]
abbrev reverseSubExpr : Expr := .call (.name "rational") [.binOp crossB .sub crossA, denominator]
abbrev mulExpr : Expr := .call (.name "rational") [.binOp na .mult nb, denominator]
abbrev forwardExpr : Expr := .call (.name "rational") [crossA, .binOp da .mult nb]
abbrev backwardExpr : Expr := .call (.name "rational") [crossB, .binOp db .mult na]

variable {env : HeapCalls.Env} {heap : Mutable.Heap} {globals : Mutable.Globals}
variable {left right : Expr} {x y : Int}

theorem times (a : Evaluates program env left heap (number x) heap globals)
    (b : Evaluates program env right heap (number y) heap globals) :
    Evaluates program env (.binOp left .mult right) heap (number (x * y)) heap globals :=
  Evaluates.binary a b (by simp [Mutable.binary, scalarBinary])

theorem plus (a : Evaluates program env left heap (number x) heap globals)
    (b : Evaluates program env right heap (number y) heap globals) :
    Evaluates program env (.binOp left .add right) heap (number (x + y)) heap globals :=
  Evaluates.binary a b (by simp [Mutable.binary, scalarBinary])

theorem minus (a : Evaluates program env left heap (number x) heap globals)
    (b : Evaluates program env right heap (number y) heap globals) :
    Evaluates program env (.binOp left .sub right) heap (number (x - y)) heap globals :=
  Evaluates.binary a b (by simp [Mutable.binary, scalarBinary])

/-- Normalization allocates one canonical tuple after evaluating both scalar arguments. -/
theorem normalized (nonzero : y ≠ 0)
    (numerator : Evaluates program env left heap (number x) heap globals)
    (denominator : Evaluates program env right heap (number y) heap globals) :
    Evaluates program env (.call (.name "rational") [left, right]) heap (.ref heap.length)
      (heap ++ [object (normalize x y)]) globals :=
  Evaluates.call (.cons numerator (.cons denominator .nil)) (rational.result x y nonzero heap globals)

theorem addition (a b : Pair) (positiveA : 0 < a.2) (positiveB : 0 < b.2) (ready : Ready env a b) :
    Evaluates program env addExpr heap (.ref heap.length) (heap ++ [object (canonical (add a b))]) globals :=
  normalized (mul_ne_zero positiveA.ne' positiveB.ne')
    (plus (times (.lookup ready.numeratorA) (.lookup ready.denominatorB))
      (times (.lookup ready.numeratorB) (.lookup ready.denominatorA))) (.lookup ready.denominator)

theorem subtraction (a b : Pair) (positiveA : 0 < a.2) (positiveB : 0 < b.2) (ready : Ready env a b) :
    Evaluates program env subExpr heap (.ref heap.length) (heap ++ [object (canonical (sub a b))]) globals :=
  normalized (mul_ne_zero positiveA.ne' positiveB.ne')
    (minus (times (.lookup ready.numeratorA) (.lookup ready.denominatorB))
      (times (.lookup ready.numeratorB) (.lookup ready.denominatorA))) (.lookup ready.denominator)

theorem reverse_subtraction (a b : Pair) (positiveA : 0 < a.2) (positiveB : 0 < b.2) (ready : Ready env a b) :
    Evaluates program env reverseSubExpr heap (.ref heap.length) (heap ++ [object (canonical (sub b a))]) globals := by
  have evaluated := normalized (mul_ne_zero positiveA.ne' positiveB.ne')
    (minus (times (heap := heap) (globals := globals) (.lookup ready.numeratorB) (.lookup ready.denominatorA))
      (times (.lookup ready.numeratorA) (.lookup ready.denominatorB))) (.lookup ready.denominator)
  simpa only [canonical, sub, mul_comm a.2 b.2] using evaluated

theorem multiplication (a b : Pair) (positiveA : 0 < a.2) (positiveB : 0 < b.2) (ready : Ready env a b) :
    Evaluates program env mulExpr heap (.ref heap.length) (heap ++ [object (canonical (mul a b))]) globals :=
  normalized (mul_ne_zero positiveA.ne' positiveB.ne')
    (times (.lookup ready.numeratorA) (.lookup ready.numeratorB)) (.lookup ready.denominator)

theorem forward (a b : Pair) (positiveA : 0 < a.2) (nonzeroB : b.1 ≠ 0) (ready : Ready env a b) :
    Evaluates program env forwardExpr heap (.ref heap.length) (heap ++ [object (division a b)]) globals :=
  normalized (mul_ne_zero positiveA.ne' nonzeroB)
    (times (.lookup ready.numeratorA) (.lookup ready.denominatorB))
    (times (.lookup ready.denominatorA) (.lookup ready.numeratorB))

theorem backward (a b : Pair) (positiveB : 0 < b.2) (nonzeroA : a.1 ≠ 0) (ready : Ready env a b) :
    Evaluates program env backwardExpr heap (.ref heap.length) (heap ++ [object (division b a)]) globals :=
  normalized (mul_ne_zero positiveB.ne' nonzeroA)
    (times (.lookup ready.numeratorB) (.lookup ready.denominatorA))
    (times (.lookup ready.denominatorB) (.lookup ready.numeratorA))

end solvers.«93».pair_results
