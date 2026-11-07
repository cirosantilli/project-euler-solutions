import solvers.«21.py.lean.d».Functional
import solvers.«21.py.lean.d».SourceRuntime

theorem solvers.«21».solve.«BigO(limit^2)» :
    FormalSW.PyAst.HeapCalls.Runtime.IsBigO .word Solvers.P21Ast.module "solve"
      (fun n : Nat => [.scalar (.int n)]) (fun _ => []) (fun _ => True) id (fun n => n^2) := by
  refine ⟨4000, 2, fun n _ large => ?_⟩
  change 2 ≤ n at large
  have bound := solvers.«21».runtimeCost.source_bound n large
    (by
      simpa only [Nat.sub_add_cancel (by omega : 1 ≤ n)] using
        solvers.«21».Sieve.result (n - 1) [])
    (fun x heap => solvers.«21».Factors.result x heap)
  apply bound.mono
  have doubled : n + 1 ≤ 2 * n := by omega
  have squared := Nat.pow_le_pow_left doubled 2
  change 1000 * (n + 1)^2 ≤ 4000 * n^2
  simpa only [Nat.mul_pow, ← Nat.mul_assoc] using Nat.mul_le_mul_left 1000 squared
