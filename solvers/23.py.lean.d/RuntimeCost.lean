import solvers.«23.py.lean.d».SourceRuntime

theorem solvers.«23».solve.«BigO(limit^2)» :
    FormalSW.PyAst.HeapCalls.Runtime.IsBigO .word Solvers.P23Ast.module "solve"
      (fun n : Nat => [.scalar (.int n)]) (fun _ => []) (fun _ => True) id (fun n => n^2) := by
  refine ⟨4000, 1, fun n _ large => ?_⟩
  change 1 ≤ n at large
  apply (solvers.«23».runtimeCost.source_bound n).mono
  have doubled : n + 1 ≤ 2 * n := by omega
  have squared := Nat.pow_le_pow_left doubled 2
  change 1000 * (n + 1)^2 ≤ 4000 * n^2
  simpa only [Nat.mul_pow, ← Nat.mul_assoc] using Nat.mul_le_mul_left 1000 squared
