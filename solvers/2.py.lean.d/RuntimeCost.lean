import «.formalsw».solvers.«2.py.ast»
import lang.py.FormalSW.FibonacciRuntimeCost

namespace solvers.«2».runtimeCost
open FormalSW.PyAst FormalSW.PyAst.Calls
set_option maxRecDepth 4096
set_option maxHeartbeats 800000

private def initialEnv (n : Nat) : ScalarEnv :=
  let args := argumentEnv [⟨"limit", some (.name "int"), none⟩] [.int n]
  ((args.set "a" (.int 1)).set "b" (.int 2)).set "total" (.int 0)

/-- Connect the actual Python function to the shared Fibonacci loop proof. -/
theorem source_bound (n : Nat) :
    Runtime.TimeBound .word Solvers.P2Ast.module "solve" [.int n]
      (98 * (Nat.log2 n + 1) + 30) := by
  have initial : RuntimeCost.Reaches (step Solvers.P2Ast.module) (Runtime.charge .word Solvers.P2Ast.module)
      ⟨.invoke "solve" [.int n], []⟩ 15
      ⟨.block (Runtime.Fibonacci.loop true) (initialEnv n), []⟩ := by
    repeat first | exact RuntimeCost.Reaches.refl | runtime_step
  have valid : Runtime.Fibonacci.Locals true n 1 2 0 (initialEnv n) := ⟨rfl, rfl, rfl, rfl⟩
  obtain ⟨cost, value, run, bound⟩ := Runtime.Fibonacci.bound Solvers.P2Ast.module true n (initialEnv n) valid
  exact ⟨15 + cost, .ok value, initial.finishes run, by omega⟩

end solvers.«2».runtimeCost

theorem solvers.«2».solve.«BigO(Nat.log2 limit)» :
    FormalSW.PyAst.Calls.Runtime.IsBigO .word Solvers.P2Ast.module "solve"
      (fun n : Nat => [.int n])
      (fun n => FormalSW.PyAst.Calls.Condition Solvers.P2Ast.module
        (fun name => if name = "limit" then some (.int n) else none) Solvers.P2Ast.requires_solve_0)
      id Nat.log2 := by
  refine ⟨226, 2, fun n _ large => ?_⟩
  have positive : 1 ≤ Nat.log2 n := (Nat.le_log2 (by simp only [id_eq] at large; omega)).mpr
    (by simpa using large)
  exact (solvers.«2».runtimeCost.source_bound n).mono (by simp only [id_eq]; omega)
