import solvers.«93.py.lean.d».Rational

namespace solvers.«93».rational
open FormalSW.PyAst
set_option Elab.async false

private abbrev inputCondition (denominator : Int) :=
  Calls.Condition program (fun name => if name = "denominator" then some (.int denominator) else none)
    Solvers.P93Ast.requires_rational_0

theorem «Returns» (numerator denominator : Int) (valid : inputCondition denominator) :
    (HeapCalls.call program "rational" [number numerator, number denominator] []).Returns := by
  have nonzero : denominator ≠ 0 := by
    simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare valid rfl
  exact (result numerator denominator nonzero [] []).returns

theorem «Supported» (numerator denominator : Int) (valid : inputCondition denominator) :
    (HeapCalls.call program "rational" [number numerator, number denominator] []).Supported :=
  («Returns» numerator denominator valid).supported

end solvers.«93».rational

namespace solvers.«93»
open FormalSW.PyAst

theorem rational_source (numerator denominator : Int) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    denominator ≠ 0 → rational.Verified numerator denominator heap globals :=
  fun nonzero => rational.verified numerator denominator nonzero heap globals

end solvers.«93»
