import solvers.«84.py.lean.d».Transport
import ProjectEulerSolutions.Lib.Monopoly

namespace solvers.«84»
open FormalSW.PyAst HeapCalls
set_option Elab.async false

private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)

private theorem add (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (left right : Expr) (a b : Int)
    (lhs : Evaluates program env left heap (number a) heap globals)
    (rhs : Evaluates program env right heap (number b) heap globals) :
    Evaluates program env (.binOp left .add right) heap (number (a + b)) heap globals :=
  Evaluates.binary lhs rhs (by simp [Mutable.binary, scalarBinary, number])

private theorem mul (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (left right : Expr) (a b : Int)
    (lhs : Evaluates program env left heap (number a) heap globals)
    (rhs : Evaluates program env right heap (number b) heap globals) :
    Evaluates program env (.binOp left .mult right) heap (number (a * b)) heap globals :=
  Evaluates.binary lhs rhs (by simp [Mutable.binary, scalarBinary, number])

/-- The actual Python encoding returns the flattened index and preserves the caller heap. -/
theorem encode_state.result (square doubles : Int) (heap : Mutable.Heap)
    (globals : Mutable.Globals := []) :
    ReturnsValue program "encode_state" [number square, number doubles] heap
      (number (square * 3 + doubles)) heap globals := by
  let args : List Arg := [⟨"pos", none, none⟩, ⟨"doubles", none, none⟩]
  let body : List Stmt := [.return (.binOp (.binOp (.name "pos") .mult (.constant (.int 3)))
    .add (.name "doubles"))]
  apply BlockReturns.function (args := args) (body := body) rfl rfl
  apply BlockReturns.return_value
  apply add
  · apply mul
    · exact Evaluates.lookup (by cases globals <;> rfl)
    · exact Evaluates.constant
  · exact Evaluates.lookup (by cases globals <;> rfl)

theorem encode_state.natural (square doubles : Nat) (heap : Mutable.Heap)
    (globals : Mutable.Globals := []) :
    ReturnsValue program "encode_state" [number square, number doubles] heap
      (number (ProjectEulerSolutions.Monopoly.encode square doubles)) heap globals := by
  simpa [ProjectEulerSolutions.Monopoly.encode] using result square doubles heap globals

end solvers.«84»
