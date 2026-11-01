import solvers.«92.py.lean.d».Digits

namespace solvers.«92».next_term
open FormalSW.PyAst
set_option Elab.async false

private abbrev inputCondition (n : Int) :=
  Calls.Condition program (fun name => if name = "n" then some (.int n) else none)
    Solvers.P92Ast.requires_next_term_0

theorem «Returns» (n : Int) (valid : inputCondition n) :
    (HeapCalls.scalarFunction program "next_term" [.int n]).Returns := by
  have nonnegative : 0 ≤ n := by
    simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare valid rfl
  lift n to Nat using nonnegative
  obtain ⟨_, returned⟩ := result n []
  exact returned.scalar_returns (values := [.int n])

theorem «Supported» (n : Int) (valid : inputCondition n) :
    (HeapCalls.scalarFunction program "next_term" [.int n]).Supported :=
  («Returns» n valid).supported

end solvers.«92».next_term
