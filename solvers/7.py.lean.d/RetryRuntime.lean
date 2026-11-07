import solvers.«7.py.lean.d».SetupRuntime
import lang.py.FormalSW.PrimitiveWorkBounds

namespace solvers.«7».retryRuntime
open FormalSW.PyAst HeapCalls.Runtime ProjectEulerSolutions.P7
open scanRuntime (attemptBody)
open setupRuntime (prepareWork finalHeap)
set_option maxRecDepth 4096
set_option maxHeartbeats 1000000
private abbrev program := Solvers.P7Ast.module

def attemptWork (limit : Nat) : Nat := prepareWork limit + 50*sieveSize limit + 23

theorem attempt_returns (k limit prime : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (following : List Stmt) (target : env "n" = some (.scalar (.int (k+2 : Nat))))
    (bound : env "limit" = some (.scalar (.int limit))) (mathUnbound : env "math" = none)
    (result : nthPrimeWithLimit (k+1) limit = some prime) :
    BlockReturnsIn .word program env (attemptBody ++ following) heap (attemptWork limit)
      (.scalar (.int prime)) (finalHeap heap limit) := by
  obtain ⟨ready, prepared, getN, getLimit, size, buffer, mathReady⟩ :=
    setupRuntime.prepare limit env heap bound mathUnbound
  have found : (finalHeap heap limit)[heap.length+1]? =
      some (Mutable.Object.bytearray (sieveBytes (oddsOnlySieveImpl limit))) := by simp [finalHeap]
  have scan := scanRuntime.source_returns (k+1) prime (oddsOnlySieveImpl limit) ready
    (finalHeap heap limit) (heap.length+1) (attemptBody[8]'(by decide) :: following)
    (getN.trans target) (by simpa [oddsOnlySieveImpl_size] using size) buffer found result
  have run := prepared.returns scan
  apply run.mono
  simp only [oddsOnlySieveImpl_size]
  unfold attemptWork
  omega

theorem attempt_continues (k limit : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (target : env "n" = some (.scalar (.int (k+2 : Nat))))
    (bound : env "limit" = some (.scalar (.int limit))) (mathUnbound : env "math" = none)
    (result : nthPrimeWithLimit (k+1) limit = none) :
    ∃ finalEnv, BlockContinuesIn .word program env attemptBody heap (attemptWork limit)
      finalEnv (finalHeap heap limit) ∧ finalEnv "n" = some (.scalar (.int (k+2 : Nat))) ∧
      finalEnv "limit" = some (.scalar (.int (limit*2 : Nat))) ∧ finalEnv "math" = none := by
  obtain ⟨ready, prepared, getN, getLimit, size, buffer, mathReady⟩ :=
    setupRuntime.prepare limit env heap bound mathUnbound
  have found : (finalHeap heap limit)[heap.length+1]? =
      some (Mutable.Object.bytearray (sieveBytes (oddsOnlySieveImpl limit))) := by simp [finalHeap]
  obtain ⟨scanned, scan, stable⟩ := scanRuntime.source_continues (k+1) (oddsOnlySieveImpl limit) ready
    (finalHeap heap limit) (heap.length+1) (getN.trans target)
    (by simpa [oddsOnlySieveImpl_size] using size) buffer found result
  let finalEnv := HeapCalls.setLocal scanned "limit" (.scalar (.int (limit*2 : Nat)))
  have grow : BlockContinuesIn .word program scanned [attemptBody[8]'(by decide)] (finalHeap heap limit)
      7 finalEnv (finalHeap heap limit) := by
    exact BlockContinuesIn.augAssign (left := .int limit) (right := .int 2) (work := 1)
      ((stable "limit" (by simp)).trans (getLimit.trans bound)) EvaluatesIn.constant
      (by simp [HeapCalls.inplace, Mutable.binary, scalarBinary]) rfl
  refine ⟨finalEnv, ?_, ?_, rfl, ?_⟩
  · have run := prepared.append (scan.append grow)
    apply run.mono
    simp only [oddsOnlySieveImpl_size]
    unfold attemptWork
    omega
  · simpa [finalEnv, HeapCalls.setLocal] using (stable "n" (by simp)).trans (getN.trans target)
  · simpa [finalEnv, HeapCalls.setLocal] using (stable "math" (by simp)).trans mathReady

theorem attemptWork_bound (limit ceiling : Nat) (positive : 1 ≤ limit) (bounded : limit ≤ ceiling) :
    attemptWork limit + 3 ≤ 1000*limit*(Nat.log2 ceiling+1) := by
  have size : sieveSize limit ≤ limit := by unfold sieveSize; omega
  have logSize := PrimitiveWork.log2_mono (size.trans bounded)
  have product := Nat.mul_le_mul size (Nat.add_le_add_right logSize 2)
  have logLimit := Nat.log2_le_self limit
  unfold attemptWork prepareWork
  nlinarith only [product, size, logLimit, positive]

/-- A failed attempt doubles a positive limit below the target prime. Charging
against the remaining geometric allowance avoids multiplying by retry count. -/
theorem retry (k ceiling limit : Nat) (positive : 1 ≤ limit) (bounded : limit ≤ ceiling)
    (enough : 2*Nat.nth Nat.Prime (k+1) ≤ ceiling)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (target : env "n" = some (.scalar (.int (k+2 : Nat))))
    (bound : env "limit" = some (.scalar (.int limit))) (mathUnbound : env "math" = none) :
    ∃ heap', BlockReturnsIn .word program env [.whileDo (.constant (.bool true)) attemptBody []] heap
      (1000*(2*ceiling-limit)*(Nat.log2 ceiling+1))
      (.scalar (.int (Nat.nth Nat.Prime (k+1)))) heap' := by
  have price := attemptWork_bound limit ceiling positive bounded
  cases result : nthPrimeWithLimit (k+1) limit with
  | some prime =>
      have correct := nthPrimeWithLimit_sound result
      subst prime
      have attempt := attempt_returns k limit _ env heap [.whileDo (.constant (.bool true)) attemptBody []] target bound mathUnbound result
      have entered := BlockReturnsIn.while_loop (body := attemptBody) (otherwise := []) (following := []) (cc := 1) (value := .bool true) (flag := true)
        EvaluatesIn.constant rfl attempt
      refine ⟨finalHeap heap limit, entered.mono ?_⟩
      have allowance : limit ≤ 2*ceiling-limit := by omega
      have grown := Nat.mul_le_mul_right (Nat.log2 ceiling+1) (Nat.mul_le_mul_left 1000 allowance)
      omega
  | none =>
      have below : limit < Nat.nth Nat.Prime (k+1) := by
        by_contra bad
        have succeeds := nthPrimeWithLimit_eq_nth (k+1) limit (by omega)
        rw [result] at succeeds
        contradiction
      obtain ⟨advanced, attempt, targetNext, boundNext, mathNext⟩ :=
        attempt_continues k limit env heap target bound mathUnbound result
      obtain ⟨heap', rest⟩ := retry k ceiling (limit*2) (by omega) (by omega) enough
        advanced (finalHeap heap limit) targetNext boundNext mathNext
      have entered := BlockReturnsIn.while_loop (body := attemptBody) (otherwise := []) (following := []) (cc := 1) (value := .bool true) (flag := true)
        EvaluatesIn.constant rfl (attempt.returns rest)
      refine ⟨heap', entered.mono ?_⟩
      have difference : 2*ceiling-limit = limit+(2*ceiling-limit*2) := by omega
      rw [difference]
      nlinarith only [price]
termination_by Nat.nth Nat.Prime (k+1)-limit
decreasing_by omega

end solvers.«7».retryRuntime
