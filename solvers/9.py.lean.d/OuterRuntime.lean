import solvers.«9.py.lean.d».RangeRuntime

namespace solvers.«9».runtimeCost
open FormalSW.PyAst HeapCalls.Runtime solvers.«9».runtimeSource
private abbrev program := Solvers.P9Ast.module

theorem outerCondition_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (total m : Int)
    (ht : env "total" = some (.scalar (.int total))) (hm : env "m" = some (.scalar (.int m))) :
    EvaluatesIn .word program env outerCondition heap 22
      (.scalar (.bool (decide (2 * m * (m + 1) ≤ total)))) heap := by
  apply integer_compare
    (integer_binary
      (integer_binary EvaluatesIn.constant (EvaluatesIn.lookup hm) rfl)
      (integer_binary (EvaluatesIn.lookup hm) EvaluatesIn.constant rfl) rfl)
    (EvaluatesIn.lookup ht)
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

/-- A square potential pays for the triangular inner ranges, including all
primitive work in each candidate and both source loop guards. -/
theorem outerLoop_cost (total m : Nat) (best : Int) (mpositive : 2 ≤ m)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (ht : env "total" = some (.scalar (.int total))) (hm : env "m" = some (.scalar (.int m)))
    (hbest : env "best" = some (.scalar (.int best))) :
    ∃ newBest finalEnv finalHeap cost, cost ≤ 200 * (total - m * m) + 24 ∧
      BlockContinuesIn .word program env [outerLoop] heap cost finalEnv finalHeap ∧
      finalEnv "total" = some (.scalar (.int total)) ∧ finalEnv "best" = some (.scalar (.int newBest)) := by
  have guard : EvaluatesIn .word program env outerCondition heap 22
      (.scalar (.bool (decide (2 * m * (m + 1) ≤ total)))) heap := by
    have evaluated := outerCondition_cost env heap total m ht hm
    have castBound : (2 : Int) * m * ((m : Int) + 1) = (2 * m * (m + 1) : Nat) := rfl
    rw [castBound] at evaluated
    simpa only [Int.ofNat_le] using evaluated
  by_cases within : 2 * m * (m + 1) ≤ total
  · obtain ⟨rowBest, rowEnv, rowHeap, rowCost, rowBound, row, rowTotal, rowM, rowBestLookup⟩ :=
      innerRange_cost total best m 1 (by omega) env heap ht hm hbest
    have ranged : BlockContinuesIn .word program env
        [.forIn (.name "n") (.call (.name "range") [.constant (.int 1), .name "m"]) innerBody []]
        heap (1 + 1 + rowCost + 7) rowEnv rowHeap :=
      BlockContinuesIn.for_range_pair EvaluatesIn.constant (EvaluatesIn.lookup hm) row
    let nextEnv := HeapCalls.setLocal rowEnv "m" (.scalar (.int (m + 1 : Nat)))
    have incremented : BlockContinuesIn .word program rowEnv
        [.augAssign (.name "m") .add (.constant (.int 1))] rowHeap 7 nextEnv rowHeap := by
      apply BlockContinuesIn.augAssign (work := 1) rowM EvaluatesIn.constant
      · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
      · rfl
    have nextTotal : nextEnv "total" = some (.scalar (.int total)) := by
      simpa [nextEnv, HeapCalls.setLocal] using rowTotal
    have nextBest : nextEnv "best" = some (.scalar (.int rowBest)) := by
      simpa [nextEnv, HeapCalls.setLocal] using rowBestLookup
    have squareLower : m ≤ m * m := by
      have := Nat.mul_le_mul_left m (show 1 ≤ m by omega)
      simpa using this
    have nextSquare : (m + 1) * (m + 1) ≤ total := by
      simp only [Nat.mul_add, Nat.mul_one, Nat.mul_assoc] at within
      simp only [Nat.add_mul, Nat.mul_add, Nat.mul_one, Nat.one_mul]
      omega
    have potential : total - m * m = total - (m + 1) * (m + 1) + 2 * m + 1 := by
      simp only [Nat.add_mul, Nat.mul_add, Nat.mul_one, Nat.one_mul] at nextSquare ⊢
      omega
    obtain ⟨newBest, finalEnv, finalHeap, restCost, restBound, rest, keptTotal, answer⟩ :=
      outerLoop_cost total (m + 1) rowBest (by omega) nextEnv rowHeap nextTotal rfl nextBest
    refine ⟨newBest, finalEnv, finalHeap,
      22 + (((1 + 1 + rowCost + 7) + 7) + restCost) + 2, ?_, ?_, keptTotal, answer⟩
    · rw [potential]
      omega
    · apply BlockContinuesIn.while_loop (flag := true) (value := .bool true)
      · simpa [within] using guard
      · rfl
      · exact (ranged.append incremented).append rest
  · refine ⟨best, env, heap, 24, by omega, ?_, ht, hbest⟩
    apply BlockContinuesIn.while_loop (cc := 22) (cb := 0) (flag := false) (value := .bool false)
    · simpa [within] using guard
    · rfl
    · exact BlockContinuesIn.nil
termination_by total + 1 - m
decreasing_by
  have bound : m ≤ total := by
    simp only [Nat.mul_add, Nat.mul_one, Nat.mul_assoc] at within
    omega
  omega


end solvers.«9».runtimeCost
