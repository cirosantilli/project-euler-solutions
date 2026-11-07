import solvers.«12.py.lean.d».SearchBudget
import ProjectEulerStatements.Lib.NthPrimeBounds

namespace solvers.«12»
open FormalSW.PyAst

theorem runtimeCost.source_bound (k : Nat) :
    HeapCalls.Runtime.TimeBound .word Solvers.P12Ast.module "solve"
      [.scalar (.int k)] [] (55000*(ProjectEulerStatements.P12.firstIndex k+1)^2) := by
  obtain ⟨bound, prime, enough, upper⟩ :=
    ProjectEulerStatements.NthPrimeBounds.exists_prime_sqrt_bound
      (ProjectEulerStatements.P12.firstIndex k+1)
  exact (runtimeCost.source_bound_for_prime k bound prime enough).mono
    (runtimeCost.search_budget_le k bound upper)

theorem solve.«BigO((ProjectEulerStatements.P12.firstIndex n + 1)^2, n = k)» :
    HeapCalls.Runtime.IsBigO .word Solvers.P12Ast.module "solve"
      (fun k : Nat => [.scalar (.int k)]) (fun _ => [])
      (fun k : Nat => Calls.Condition Solvers.P12Ast.module
        (fun name => if name = "k" then some (.int k) else none)
        Solvers.P12Ast.requires_solve_0) id
      (fun k => (ProjectEulerStatements.P12.firstIndex k+1)^2) := by
  exact ⟨55000, 0, fun k _ _ => runtimeCost.source_bound k⟩

end solvers.«12»
