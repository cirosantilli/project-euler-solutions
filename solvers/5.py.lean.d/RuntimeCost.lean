import «.formalsw».solvers.«5.py.ast»
import lang.py.FormalSW.LcmRuntimeCost

namespace solvers.«5».runtimeCost
open FormalSW.PyAst FormalSW.PyAst.Calls
set_option maxRecDepth 4096
set_option maxHeartbeats 800000

private def initialEnv (n : Nat) : ScalarEnv :=
  (argumentEnv [⟨"n", some (.name "int"), none⟩] [.int n]).set "ans" (.int 1)

private theorem binding : Runtime.Lcm.GcdBinding Solvers.P5Ast.module := ⟨rfl, rfl⟩

/-- Includes the range setup and the work charged inside each gcd call. -/
theorem source_bound (n : Nat) :
    Runtime.TimeBound .word Solvers.P5Ast.module "solve" [.int n]
      ((Nat.log2 n + 25) * n + 26) := by
  have initial : RuntimeCost.Reaches (step Solvers.P5Ast.module) (Runtime.charge .word Solvers.P5Ast.module)
      ⟨.invoke "solve" [.int n], []⟩ 19
      ⟨.block (Runtime.Lcm.loop true 2 (n + 1)) (initialEnv n), []⟩ := by
    repeat first | exact RuntimeCost.Reaches.refl | runtime_step
  cases n with
  | zero =>
      refine ⟨26, .ok (.int 1), ?_, by decide⟩
      apply RuntimeCost.Reaches.finishes (cb := 7) initial
      apply RuntimeCost.Finishes.next (amount := 4) (cost := 3)
      · rfl
      · rfl
      · apply RuntimeCost.Finishes.next (amount := 1) (cost := 2)
        · rfl
        · rfl
        · apply RuntimeCost.Finishes.next (amount := 1) (cost := 1)
          · rfl
          · rfl
          · exact .done rfl rfl
  | succ n =>
      obtain ⟨cost, answer, run, bound⟩ := Runtime.Lcm.loop_bound Solvers.P5Ast.module binding
        true (n + 1) n 2 (initialEnv (n + 1)) 1 (by decide) (by omega) rfl
      have enlarged := Nat.mul_le_mul_left (Nat.log2 (n + 1) + 25) (Nat.le_succ n)
      simp only [Nat.succ_eq_add_one] at enlarged
      exact ⟨19 + cost, .ok (.int answer), initial.finishes run, by omega⟩

end solvers.«5».runtimeCost

theorem solvers.«5».solve.«BigO(n * Nat.log2 n)» :
    FormalSW.PyAst.Calls.Runtime.IsBigO .word Solvers.P5Ast.module "solve"
      (fun n : Nat => [.int n])
      (fun n => FormalSW.PyAst.Calls.Condition Solvers.P5Ast.module
        (fun name => if name = "n" then some (.int n) else none) Solvers.P5Ast.requires_solve_0)
      id (fun n => n * Nat.log2 n) := by
  refine ⟨64, 2, fun n _ large => ?_⟩
  exact (solvers.«5».runtimeCost.source_bound n).mono (FormalSW.PyAst.Calls.Runtime.Lcm.budget_bound n large)
