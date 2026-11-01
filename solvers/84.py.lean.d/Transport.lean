import «.formalsw».solvers.«84.py.ast»
import ProjectEulerSolutions.Lib.CyclicStops
import lang.py.FormalSW.HeapCalls

namespace solvers.«84»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.CyclicStops (nextStop)
set_option Elab.async false

abbrev program := Solvers.P84Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)

private def stopCode (stops : List Int) (start : Int) : List Stmt :=
  match stops with
  | [] => [.return (.constant (.int start))]
  | stop :: rest =>
      .ifThen (.compare (.name "pos") [(.lt, .constant (.int stop))])
        [.return (.constant (.int stop))] [] :: stopCode rest start

private theorem stops_return (module : Module) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (stops : List Int) (start position : Int) (globals : Mutable.Globals)
    (localPosition : env "pos" = some (number position)) :
    BlockReturns module env (stopCode stops start) heap (number (nextStop stops start position)) heap globals := by
  induction stops with
  | nil => exact BlockReturns.return_value Evaluates.constant
  | cons stop rest ih =>
      have tested : Evaluates module env (.compare (.name "pos") [(.lt, .constant (.int stop))])
          heap (.scalar (.bool (decide (position < stop)))) heap globals := by
        apply Evaluates.compare (Evaluates.lookup localPosition) Evaluates.constant
        simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      by_cases later : position < stop
      · rw [nextStop, if_pos later]
        apply BlockReturns.branch (flag := true) (by simpa only [later, decide_true] using tested) rfl
        exact BlockReturns.return_value Evaluates.constant
      · rw [nextStop, if_neg later]
        apply BlockReturns.branch (flag := false) (by simpa only [later, decide_false] using tested) rfl
        exact ih

/-- Select the next railway in route order, with wraparound and no heap effects. -/
theorem next_railway.result (position : Int) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "next_railway" [number position] heap
      (number (nextStop [5, 15, 25, 35] 5 position)) heap globals := by
  let args : List Arg := [⟨"pos", some (.name "int"), none⟩]
  apply BlockReturns.function (args := args) (body := stopCode [5, 15, 25, 35] 5) rfl rfl
  exact stops_return program heap (functionEnv args (stopCode [5, 15, 25, 35] 5) [number position] globals)
    [5, 15, 25, 35] 5 position globals (by cases globals <;> rfl)

/-- Select the next utility in route order, retaining arbitrary caller globals. -/
theorem next_utility.result (position : Int) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "next_utility" [number position] heap
      (number (nextStop [12, 28] 12 position)) heap globals := by
  let args : List Arg := [⟨"pos", some (.name "int"), none⟩]
  apply BlockReturns.function (args := args) (body := stopCode [12, 28] 12) rfl rfl
  exact stops_return program heap (functionEnv args (stopCode [12, 28] 12) [number position] globals)
    [12, 28] 12 position globals (by cases globals <;> rfl)

end solvers.«84»
