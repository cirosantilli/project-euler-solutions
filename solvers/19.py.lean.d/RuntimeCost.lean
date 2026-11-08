import solvers.«19.py.lean.d».SourceRuntime

theorem solvers.«19».solve.«BigO(n, n = abs(start_year) + abs(end_year) + 1)» :
    FormalSW.PyAst.HeapCalls.Runtime.IsBigO .word Solvers.P19Ast.module "solve"
  (fun input : Int × Int => [.scalar (.int input.1), .scalar (.int input.2)]) (fun _ => [])
  (fun input : Int × Int => (FormalSW.PyAst.Calls.Condition Solvers.P19Ast.module (fun _arg => if _arg = "start_year" then some (.int input.1) else if _arg = "end_year" then some (.int input.2) else none) Solvers.P19Ast.requires_solve_0)) (fun input : Int × Int => input.1.natAbs + input.2.natAbs + 1) (fun n : Nat => n) := by
  refine ⟨1669, 1, fun input _ _ => ?_⟩
  apply (solvers.«19».runtimeCost.source_bound input.1 input.2).mono
  have magnitude : input.2 ≤ (input.2.natAbs : Int) := Int.le_natAbs
  have countBound : (input.2+1-1900).toNat ≤ input.1.natAbs + input.2.natAbs + 1 := by omega
  simp only
  omega
