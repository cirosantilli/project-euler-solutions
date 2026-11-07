import solvers.«28.py.lean.d».State
import lang.py.FormalSW.CallsRuntimeCost

namespace solvers.«28».runtimeCost
open FormalSW.PyAst FormalSW.PyAst.Calls
set_option maxRecDepth 4096
set_option maxHeartbeats 800000

/-- Charge every arithmetic expression, the assertion, assignments and return. -/
theorem source_bound (n : Int) (positive : 1 ≤ n) (odd : n % 2 = 1) :
    Runtime.TimeBound .word program "solve" [.int n] 256 := by
  apply Exists.intro
  refine ⟨.ok (.int (formula ((n - 1) / 2))), ?_, ?_⟩
  · repeat first
      | exact RuntimeCost.Finishes.done rfl rfl
      | apply RuntimeCost.Finishes.next <;> first
        | rfl
        | simp [evalCmpOp, positive, odd]
        | skip
  · simp [PrimitiveWork.comparison, PrimitiveWork.scalarSize, PrimitiveWork.integer]

end solvers.«28».runtimeCost

theorem solvers.«28».solve.«BigO(1)» :
    FormalSW.PyAst.Calls.Runtime.IsBigO .word Solvers.P28Ast.module "solve"
      (fun n : Nat => [.int n])
      (fun n => FormalSW.PyAst.Calls.Condition Solvers.P28Ast.module
        (fun name => if name = "n" then some (.int n) else none) Solvers.P28Ast.requires_solve_0)
      id (fun _ => 1) := by
  refine ⟨256, 0, fun n requires _ => ?_⟩
  obtain ⟨positive, odd⟩ := solvers.«28».domain n requires
  exact solvers.«28».runtimeCost.source_bound n positive odd
