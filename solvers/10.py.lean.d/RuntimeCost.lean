import solvers.«10.py.lean.d».SetupRuntime

namespace solvers.«10».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
open sumRuntime (solveBody)
private abbrev program := Solvers.P10Ast.module

theorem source_bound (limit : Nat) (large : 4 ≤ limit) :
    TimeBound .word program "solve" [.scalar (.int limit)] [] (512 * (limit * Nat.log2 limit)) := by
  let env := argumentEnv [⟨"limit", some (.name "int"), none⟩] [.scalar (.int limit)]
  have guard (bound : Nat) : EvaluatesIn .word program env
      (.compare (.name "limit") [(.ltE, .constant (.int bound))]) [] 7
      (.scalar (.bool (decide (limit ≤ bound)))) [] := by
    apply EvaluatesIn.compare (ca := 1) (cb := 1) (work := 2)
      (EvaluatesIn.lookup rfl) EvaluatesIn.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · rfl
  obtain ⟨finalEnv, finalHeap, bytes, prepared, getSieve, found, length⟩ :=
    setupRuntime.prepare limit large env [] rfl rfl
  have finished := sumRuntime.finish bytes finalEnv finalHeap 1 getSieve found
  have body : BlockReturnsIn .word program env solveBody []
      (6 * sieveSize limit * ((sieveSize limit).log2 + 2) + 344 * sieveSize limit + Nat.log2 (limit - 1) + 103)
      (.scalar (.int (2 + (oddSum 0 bytes : Int)))) finalHeap := by
    have rest := prepared.returns finished
    rw [length] at rest
    have second := BlockReturnsIn.branch (cc := 7) (flag := false) (value := .bool false)
      (yes := [.return (.constant (.int 2))]) (no := [])
      (by simpa [show ¬ limit ≤ 3 by omega] using guard 3) rfl rest
    have first := BlockReturnsIn.branch (cc := 7) (flag := false) (value := .bool false)
      (yes := [.return (.constant (.int 0))]) (no := [])
      (by simpa [show ¬ limit ≤ 2 by omega] using guard 2) rfl second
    exact first.mono (by omega)
  have called := BlockReturnsIn.function (module := program) (name := "solve")
    (values := [.scalar (.int limit)]) rfl rfl body
  obtain ⟨cost, trace, paid⟩ := called 0 []
  have terminal : RuntimeCost.Finishes (step program 0) (charge .word program)
      ⟨.value (.scalar (.int (2 + (oddSum 0 bytes : Int)))), [], finalHeap⟩ 1
      (.ok ⟨.ok (.scalar (.int (2 + (oddSum 0 bytes : Int)))), [], finalHeap, 0⟩) := .done rfl rfl
  refine ⟨cost + 1, _, trace.finishes terminal, ?_⟩
  have initialZero : potential ⟨.operation (.call "solve") [.scalar (.int limit)], [], []⟩ = 0 := rfl
  rw [initialZero] at paid
  have costBound : cost + 1 ≤
      6 * sieveSize limit * ((sieveSize limit).log2 + 2) + 344 * sieveSize limit + Nat.log2 (limit - 1) + 107 := by
    simp only [List.length_cons, List.length_nil] at paid
    omega
  have sizeBound : sieveSize limit ≤ limit := by unfold sieveSize; omega
  have logs : (sieveSize limit).log2 ≤ limit.log2 := by
    by_cases zero : sieveSize limit = 0
    · simp [zero]
    · exact (Nat.le_log2 (by omega : limit ≠ 0)).mpr ((Nat.log2_self_le zero).trans sizeBound)
  have product := Nat.mul_le_mul sizeBound (Nat.add_le_add_right logs 2)
  have overhead := (Nat.log2_le_self (limit - 1)).trans (Nat.sub_le limit 1)
  have positive : 1 ≤ limit.log2 := by
    exact (Nat.le_log2 (by omega : limit ≠ 0)).mpr (by omega)
  have scale := Nat.mul_le_mul_left limit positive
  nlinarith only [costBound, product, overhead, scale, sizeBound, large]

end solvers.«10».runtimeCost

theorem solvers.«10».solve.«BigO(limit * Nat.log2 limit)» :
    FormalSW.PyAst.HeapCalls.Runtime.IsBigO .word Solvers.P10Ast.module "solve"
      (fun limit : Nat => [.scalar (.int limit)]) (fun _ => [])
      (fun limit : Nat => FormalSW.PyAst.Calls.Condition Solvers.P10Ast.module
        (fun name => if name = "limit" then some (.int limit) else none) Solvers.P10Ast.requires_solve_0)
      id (fun limit => limit * Nat.log2 limit) := by
  exact ⟨512, 4, fun limit _ large => solvers.«10».runtimeCost.source_bound limit large⟩
