import solvers.«84.py.lean.d».OutcomeHeap

namespace solvers.«84».outcomes
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
def count (sides move : Int) : Int := min (move - 1) (2 * sides + 1 - move)
def double (move : Int) : Int := if move % 2 = 0 then 1 else 0

def countExpression : Expr := .builtinCall .min [.binOp (.name "move") .sub (.constant (.int 1)),
  .binOp (.binOp (.binOp (.constant (.int 2)) .mult (.name "dice_sides")) .add (.constant (.int 1))) .sub (.name "move")]
def doubleExpression : Expr := .ifExp (.compare (.binOp (.name "move") .mod (.constant (.int 2))) [(.eq, .constant (.int 0))])
  (.constant (.int 1)) (.constant (.int 0))

theorem add (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (left right : Expr) (a b : Int)
    (lhs : Evaluates program env left heap (number a) heap globals)
    (rhs : Evaluates program env right heap (number b) heap globals) :
    Evaluates program env (.binOp left .add right) heap (number (a + b)) heap globals :=
  Evaluates.binary lhs rhs (by simp [Mutable.binary, scalarBinary])

theorem sub (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (left right : Expr) (a b : Int)
    (lhs : Evaluates program env left heap (number a) heap globals)
    (rhs : Evaluates program env right heap (number b) heap globals) :
    Evaluates program env (.binOp left .sub right) heap (number (a - b)) heap globals :=
  Evaluates.binary lhs rhs (by simp [Mutable.binary, scalarBinary])

theorem mul (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (left right : Expr) (a b : Int)
    (lhs : Evaluates program env left heap (number a) heap globals)
    (rhs : Evaluates program env right heap (number b) heap globals) :
    Evaluates program env (.binOp left .mult right) heap (number (a * b)) heap globals :=
  Evaluates.binary lhs rhs (by simp [Mutable.binary, scalarBinary])

theorem count_result (sides move : Int) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getSides : env "dice_sides" = some (number sides)) (getMove : env "move" = some (number move)) :
    Evaluates program env countExpression heap (number (count sides move)) heap globals := by
  apply Evaluates.builtin
    (.cons (sub env heap globals _ _ move 1 (Evaluates.lookup getMove) Evaluates.constant)
      (.cons (sub env heap globals _ _ (2 * sides + 1) move
        (add env heap globals _ _ (2 * sides) 1
          (mul env heap globals _ _ 2 sides Evaluates.constant (Evaluates.lookup getSides)) Evaluates.constant)
        (Evaluates.lookup getMove)) .nil))
  by_cases smaller : 2 * sides + 1 - move < move - 1
  · simp [HeapCalls.primitive, Calls.positionalMinimum, Calls.minimumFrom, scalarCompare, evalCmpOp,
      smaller, count, scalarResult]
    omega
  · simp [HeapCalls.primitive, Calls.positionalMinimum, Calls.minimumFrom, scalarCompare, evalCmpOp,
      smaller, count, scalarResult]
    omega

theorem double_result (move : Int) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getMove : env "move" = some (number move)) :
    Evaluates program env doubleExpression heap (number (double move)) heap globals := by
  have remainder : Evaluates program env (.binOp (.name "move") .mod (.constant (.int 2))) heap (number (move % 2)) heap globals :=
    Evaluates.binary (Evaluates.lookup getMove) Evaluates.constant (by simp [Mutable.binary, scalarBinary])
  have tested : Evaluates program env (.compare (.binOp (.name "move") .mod (.constant (.int 2))) [(.eq, .constant (.int 0))])
      heap (.scalar (.bool (decide (move % 2 = 0)))) heap globals :=
    Evaluates.compare remainder Evaluates.constant (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  apply Evaluates.conditional tested rfl
  by_cases even : move % 2 = 0 <;> simpa [double, even] using (Evaluates.constant (module := program) (env := env) (heap := heap)
    (globals := globals) (value := .int (double move)))

end solvers.«84».outcomes
