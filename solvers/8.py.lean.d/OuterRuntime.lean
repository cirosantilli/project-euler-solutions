import solvers.«8.py.lean.d».SegmentRuntime
import lang.py.FormalSW.HeapRuntimeCostLoopControl

namespace solvers.«8».outerRuntime
open FormalSW.PyAst HeapCalls.Runtime
set_option maxRecDepth 4096
set_option maxHeartbeats 1200000
private abbrev program := Solvers.P8Ast.module

theorem process (digits : List Int) (heap : Mutable.Heap) (address start stop width : Nat)
    (env : HeapCalls.Env) (best : Int)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (size : env "k" = some (.scalar (.int width)))
    (left : env "i" = some (.scalar (.int start)))
    (right : env "j" = some (.scalar (.int stop)))
    (previous : env "best" = some (.scalar (.int best)))
    (ordered : start ≤ stop) (bounded : stop ≤ digits.length)
    (nonzero : ∀ index, start ≤ index → (bound : index < stop) → digits[index]'(by omega) ≠ 0) :
    ∃ cost finalEnv best', cost ≤ 52*(stop-start)+68 ∧
      BlockContinuesIn .word program env processSegment heap cost finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int best')) ∧
      ∀ name, name ≠ "prod" → name ≠ "t" → name ≠ "best" → name ≠ "seg_len" → finalEnv name = env name := by
  let measured := HeapCalls.setLocal env "seg_len" (.scalar (.int (stop-start : Nat)))
  have sizeRun : EvaluatesIn .word program env (.binOp (.name "j") .sub (.name "i")) heap 6
      (.scalar (.int (stop-start : Nat))) heap :=
    EvaluatesIn.binary (op := .sub) (value := .scalar (.int (stop-start : Nat)))
      (EvaluatesIn.lookup right) (EvaluatesIn.lookup left)
      (by simp [Mutable.binary, scalarBinary]; omega) rfl
  have test : EvaluatesIn .word program measured (.compare (.name "seg_len") [(.gtE, .name "k")]) heap 7
      (.scalar (.bool (decide (width ≤ stop-start)))) heap :=
    EvaluatesIn.compare (op := .gtE) (env := measured) (av := .scalar (.int (stop-start : Nat)))
      (bv := .scalar (.int width)) (middle := heap) (afterRight := heap)
      (value := .scalar (.bool (decide (width ≤ stop-start)))) (EvaluatesIn.lookup rfl)
      (EvaluatesIn.lookup (by simp [measured, HeapCalls.setLocal, size]))
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]) rfl
  by_cases enough : width ≤ stop-start
  · obtain ⟨cost, finalEnv, best', bound, segment, result, kept⟩ :=
      segmentRuntime.segment_bound digits heap address start stop width measured best found
        (by simp [measured, HeapCalls.setLocal, receiver]) (by simp [measured, HeapCalls.setLocal, size])
        (by simp [measured, HeapCalls.setLocal, left]) (by simp [measured, HeapCalls.setLocal, right])
        (by simp [measured, HeapCalls.setLocal, previous]) (by omega) bounded nonzero
    refine ⟨8+(7+cost+2), finalEnv, best', by omega, ?_, result, ?_⟩
    · apply BlockContinuesIn.append (BlockContinuesIn.assign sizeRun)
      exact BlockContinuesIn.branch (value := .bool true) (flag := true) (by simpa [enough] using test) rfl segment
    · intro name np nt nb ns
      rw [kept name np nt nb]
      simp [measured, HeapCalls.setLocal, ns]
  · refine ⟨17, measured, best, by omega, ?_, ?_, ?_⟩
    · apply BlockContinuesIn.append (cb := 9) (BlockContinuesIn.assign (name := "seg_len") sizeRun)
      exact BlockContinuesIn.branch (value := .bool false) (flag := false) (by simpa [enough] using test) rfl BlockContinuesIn.nil
    · simp [measured, HeapCalls.setLocal, previous]
    · intro name _ _ _ ns
      simp [measured, HeapCalls.setLocal, ns]

/-- Each nonzero segment advances the outer cursor. The linear budget pays
for both scans, the initial product, all rolling updates and loop control. -/
theorem outer (digits : List Int) (heap : Mutable.Heap) (address width start : Nat)
    (env : HeapCalls.Env) (best : Int)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (size : env "k" = some (.scalar (.int width)))
    (count : env "digit_count" = some (.scalar (.int digits.length)))
    (index : env "i" = some (.scalar (.int start)))
    (previous : env "best" = some (.scalar (.int best)))
    (bounded : start ≤ digits.length) :
    ∃ answer : Int, BlockReturnsIn .word program env [outerLoop, .return (.name "best")]
      heap (300*(digits.length-start)+60) (.scalar (.int answer)) heap := by
  by_cases valid : start < digits.length
  · obtain ⟨first, ordered, firstBound, zeroes, firstNonzero⟩ := scan_bounds digits start true bounded
    obtain ⟨skipCost, skipBound, skipped⟩ := scanRuntime.scan_bound digits heap address start first env "i" true
      found receiver count (by decide) (by decide) ordered firstBound zeroes firstNonzero
    rw [setLocal_current env "i" _ index] at skipped
    let zeroEnv := HeapCalls.setLocal env "i" (.scalar (.int first))
    by_cases exhausted : first = digits.length
    · refine ⟨best, ?_⟩
      apply BlockReturnsIn.mono (small := 7+(skipCost+(7+3+2))+2) _ (by omega)
      apply BlockReturnsIn.while_loop (cc := 7) (value := .bool true) (flag := true)
      · exact EvaluatesIn.compare (EvaluatesIn.lookup index) (EvaluatesIn.lookup count)
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, valid]) rfl
      · rfl
      · apply BlockContinuesIn.returns skipped
        apply BlockReturnsIn.branch (cc := 7) (cb := 3) (value := .bool true) (flag := true)
        · exact EvaluatesIn.compare (av := .scalar (.int first)) (bv := .scalar (.int digits.length))
            (EvaluatesIn.lookup (by simp [HeapCalls.setLocal]))
            (EvaluatesIn.lookup (by simp [HeapCalls.setLocal, count]))
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, exhausted]) rfl
        · rfl
        · apply BlockReturnsIn.break_loop rfl
          exact BlockReturnsIn.return_value (EvaluatesIn.lookup (show zeroEnv "best" = some (.scalar (.int best)) by
            simp [zeroEnv, HeapCalls.setLocal, previous]))
    · have firstValid : first < digits.length := by omega
      have firstNotZero : digits[first] ≠ 0 := by simpa using firstNonzero firstValid
      obtain ⟨stop, stopOrdered, stopBound, selected, boundary⟩ := scan_bounds digits first false firstBound
      have progress : first < stop := by
        by_cases equal : stop = first
        · subst stop
          have zero : digits[first] = 0 := by simpa using boundary firstValid
          exact False.elim (firstNotZero zero)
        · omega
      let segmentEnv := HeapCalls.setLocal zeroEnv "j" (.scalar (.int stop))
      obtain ⟨scanCost, scanBound, scanned⟩ := scanRuntime.scan_bound digits heap address first stop zeroEnv "j" false found
        (by simp [zeroEnv, HeapCalls.setLocal, receiver]) (by simp [zeroEnv, HeapCalls.setLocal, count])
        (by decide) (by decide) stopOrdered stopBound selected boundary
      obtain ⟨processCost, finalEnv, nextBest, processBound, processed, result, kept⟩ :=
        process digits heap address first stop width segmentEnv best found
          (by simp [segmentEnv, zeroEnv, HeapCalls.setLocal, receiver])
          (by simp [segmentEnv, zeroEnv, HeapCalls.setLocal, size])
          (by simp [segmentEnv, zeroEnv, HeapCalls.setLocal]) (by simp [segmentEnv, HeapCalls.setLocal])
          (by simp [segmentEnv, zeroEnv, HeapCalls.setLocal, previous]) stopOrdered stopBound
          (fun index lower upper => by simpa using selected index lower upper)
      let nextEnv := HeapCalls.setLocal finalEnv "i" (.scalar (.int stop))
      obtain ⟨answer, next⟩ := outer digits heap address width stop nextEnv nextBest found
        (by simp only [nextEnv, HeapCalls.setLocal]
            rw [kept "digits" (by decide) (by decide) (by decide) (by decide)]
            simp [segmentEnv, zeroEnv, HeapCalls.setLocal, receiver])
        (by simp only [nextEnv, HeapCalls.setLocal]
            rw [kept "k" (by decide) (by decide) (by decide) (by decide)]
            simp [segmentEnv, zeroEnv, HeapCalls.setLocal, size])
        (by simp only [nextEnv, HeapCalls.setLocal]
            rw [kept "digit_count" (by decide) (by decide) (by decide) (by decide)]
            simp [segmentEnv, zeroEnv, HeapCalls.setLocal, count])
        (by simp [nextEnv, HeapCalls.setLocal]) (by simpa [nextEnv, HeapCalls.setLocal] using result) stopBound
      refine ⟨answer, ?_⟩
      apply BlockReturnsIn.mono
        (small := 7+(skipCost+(7+(1+(scanCost+(processCost+(1+((300*(digits.length-stop)+60)+1)+2)))+2)+2))+2)
        _ (by omega)
      apply BlockReturnsIn.while_loop (cc := 7) (value := .bool true) (flag := true)
      · exact EvaluatesIn.compare (EvaluatesIn.lookup index) (EvaluatesIn.lookup count)
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, valid]) rfl
      · rfl
      · apply BlockContinuesIn.returns skipped
        apply BlockReturnsIn.branch (cc := 7) (value := .bool false) (flag := false)
        · exact EvaluatesIn.compare (av := .scalar (.int first)) (bv := .scalar (.int digits.length))
            (EvaluatesIn.lookup (by simp [HeapCalls.setLocal]))
            (EvaluatesIn.lookup (by simp [HeapCalls.setLocal, count]))
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega) rfl
        · rfl
        · apply BlockReturnsIn.assign (ca := 1) (value := .scalar (.int first))
            (EvaluatesIn.lookup (by simp [HeapCalls.setLocal]))
          apply BlockContinuesIn.returns scanned
          apply BlockContinuesIn.returns processed
          apply BlockReturnsIn.assign (ca := 1) (value := .scalar (.int stop))
          · exact EvaluatesIn.lookup (by
              rw [kept "j" (by decide) (by decide) (by decide) (by decide)]
              simp [segmentEnv, HeapCalls.setLocal])
          · exact BlockReturnsIn.loop_resume next
  · refine ⟨best, ?_⟩
    apply BlockReturnsIn.mono (small := 11) _ (by omega)
    apply BlockReturnsIn.while_loop (cc := 7) (cb := 2) (value := .bool false) (flag := false)
    · exact EvaluatesIn.compare (EvaluatesIn.lookup index) (EvaluatesIn.lookup count)
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, valid]) rfl
    · rfl
    · exact BlockReturnsIn.return_value (EvaluatesIn.lookup previous)
termination_by digits.length-start

end solvers.«8».outerRuntime
