import «.formalsw».solvers.«1.py.ast»
import lang.py.FormalSW.CallsRuntimeCost

namespace solvers.«1».runtimeCost
open FormalSW.PyAst FormalSW.PyAst.Calls
set_option maxRecDepth 4096
set_option maxHeartbeats 1000000

/-- Fixed-cost arithmetic, including all three calls to the Python helper. -/
theorem fixed_bound (n : Nat) :
    Calls.Runtime.TimeBound .word Solvers.P1Ast.module "solve" [.int n] 256 := by
  apply Exists.intro
  refine ⟨.ok (.int (3 * (((n : Int) - 1) / 3) * (((n - 1) / 3) + 1) / 2 +
    5 * ((n - 1) / 5) * (((n - 1) / 5) + 1) / 2 -
    15 * ((n - 1) / 15) * (((n - 1) / 15) + 1) / 2)), ?_, ?_⟩
  · repeat first
      | exact RuntimeCost.Finishes.done rfl rfl
      | apply RuntimeCost.Finishes.next <;> first | rfl | skip
  · dsimp
    decide

end solvers.«1».runtimeCost

theorem solvers.«1».solve.«BigO(1)» :
    FormalSW.PyAst.Calls.Runtime.IsBigO .word Solvers.P1Ast.module "solve"
      (fun n : Nat => [.int n])
      (fun n => FormalSW.PyAst.Calls.Condition Solvers.P1Ast.module
        (fun name => if name = "n" then some (.int n) else none) Solvers.P1Ast.requires_solve_0)
      id (fun _ => 1) := by
  exact ⟨256, 0, fun n _ _ => solvers.«1».runtimeCost.fixed_bound n⟩
