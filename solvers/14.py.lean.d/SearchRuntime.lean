import solvers.«14.py.lean.d».LengthRuntime
import ProjectEulerStatements.Lib.CollatzWork

/-! Runtime of the source range loop, retaining cache growth across iterations. -/
namespace solvers.«14».runtimeCost
open FormalSW.PyAst FormalSW.PyAst.HeapCalls FormalSW.PyAst.HeapCalls.Runtime
open ProjectEulerStatements.P14 ProjectEulerStatements.Collatz

theorem search_one (limit n smallAddress bigAddress winner : Nat) (positive : 0 < n)
    (distinct : smallAddress ≠ bigAddress) (env : Mutable.Env) (heap : Mutable.Heap)
    (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big))
    (locals : SearchLocals env limit smallAddress bigAddress winner) :
    ∃ finalHeap finalSmall finalBig finalEnv,
      finalHeap[smallAddress]? = some (.list finalSmall) ∧
      finalHeap[bigAddress]? = some (.dict finalBig) ∧
      Cache.Valid limit finalSmall finalBig ∧
      SearchLocals finalEnv limit smallAddress bigAddress
        (ProjectEulerStatements.ArgMaximum.select collatzLen winner n) ∧
      finalBig.length ≤ big.length + collatzLen n ∧
      BlockContinuesIn .word program (setLocal env "n" (.scalar (.int n))) searchBody heap
        (1000 * (big.length + collatzLen n + 1) * (collatzLen n + 1) + 32) finalEnv finalHeap := by
  obtain ⟨finalHeap, finalSmall, finalBig, finalSmallFound, finalBigFound, finalValid, _, growth, called⟩ :=
    collatz_length_returns limit n smallAddress bigAddress positive distinct heap small big valid smallFound bigFound
  let entry := setLocal env "n" (.scalar (.int n))
  let scored := setLocal entry "ln" (.scalar (.int (collatzLen n)))
  have evaluated : EvaluatesIn .word program entry (.call (.name "collatz_length")
      [.name "n", .name "limit", .name "cache_small", .name "cache_big"])
      heap (1000 * (big.length + collatzLen n + 1) * (collatzLen n + 1) + 15)
      (.scalar (.int (collatzLen n))) finalHeap := by
    apply EvaluatesIn.mono (small := 14 + 1000 * (big.length + collatzLen n + 1) * (collatzLen n + 1) + 1) _ (by omega)
    apply EvaluatesIn.call (ca := 14) (values := [.scalar (.int n), .scalar (.int limit), .ref smallAddress, .ref bigAddress])
    · exact .cons (EvaluatesIn.lookup (by simp [entry, setLocal]))
        (.cons (EvaluatesIn.lookup (by simpa [entry, setLocal] using locals.bound))
        (.cons (EvaluatesIn.lookup (by simpa [entry, setLocal] using locals.small))
        (.cons (EvaluatesIn.lookup (by simpa [entry, setLocal] using locals.big)) .nil)))
    · exact called
  have test : EvaluatesIn .word program scored (.compare (.name "ln") [(.gt, .name "best_len")])
      finalHeap 7 (.scalar (.bool (decide (collatzLen winner < collatzLen n)))) finalHeap := by
    apply EvaluatesIn.compare (work := 2) (av := .scalar (.int (collatzLen n))) (EvaluatesIn.lookup (by simp [scored, entry, setLocal]))
      (EvaluatesIn.lookup (by simpa [scored, entry, setLocal] using locals.length))
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · rfl
  by_cases better : collatzLen winner < collatzLen n
  · let finalEnv := setLocal (setLocal scored "best_len" (.scalar (.int (collatzLen n)))) "best_n" (.scalar (.int n))
    refine ⟨finalHeap, finalSmall, finalBig, finalEnv, finalSmallFound, finalBigFound, finalValid, ?_, growth, ?_⟩
    · simp only [ProjectEulerStatements.ArgMaximum.select, if_pos better]
      constructor <;> simp [finalEnv, scored, entry, setLocal, locals.bound, locals.small, locals.big]
    · apply BlockContinuesIn.mono (small := (1000 * (big.length + collatzLen n + 1) * (collatzLen n + 1) + 17) + 15) _ (by omega)
      apply BlockContinuesIn.append (ca := 1000 * (big.length + collatzLen n + 1) * (collatzLen n + 1) + 17) (cb := 15) (body := [.assign [.name "ln"] (.call (.name "collatz_length")
          [.name "n", .name "limit", .name "cache_small", .name "cache_big"])])
        (middleEnv := scored) (middleHeap := finalHeap)
      · exact BlockContinuesIn.assign evaluated
      · apply BlockContinuesIn.branch (cc := 7) (cb := 6) test (flag := true) (by simp [Mutable.truth, scalarTruth, better])
        apply BlockContinuesIn.append (ca := 3) (cb := 3) (body := [.assign [.name "best_len"] (.name "ln")])
          (middleEnv := setLocal scored "best_len" (.scalar (.int (collatzLen n)))) (middleHeap := finalHeap)
        · exact BlockContinuesIn.assign (EvaluatesIn.lookup (by simp [scored, setLocal]))
        · exact BlockContinuesIn.assign (EvaluatesIn.lookup (by simp [scored, entry, setLocal]))
  · refine ⟨finalHeap, finalSmall, finalBig, scored, finalSmallFound, finalBigFound, finalValid, ?_, growth, ?_⟩
    · simp only [ProjectEulerStatements.ArgMaximum.select, if_neg better]
      constructor <;> simp [scored, entry, setLocal, locals.bound, locals.small, locals.big, locals.best, locals.length]
    · apply BlockContinuesIn.mono (small := (1000 * (big.length + collatzLen n + 1) * (collatzLen n + 1) + 17) + 15) _ (by omega)
      apply BlockContinuesIn.append (ca := 1000 * (big.length + collatzLen n + 1) * (collatzLen n + 1) + 17) (cb := 15) (body := [.assign [.name "ln"] (.call (.name "collatz_length")
          [.name "n", .name "limit", .name "cache_small", .name "cache_big"])])
        (middleEnv := scored) (middleHeap := finalHeap)
      · exact BlockContinuesIn.assign evaluated
      · exact (BlockContinuesIn.branch test (flag := false) (by simp [Mutable.truth, scalarTruth, better]) BlockContinuesIn.nil).mono (by omega)

theorem search_loop (count start limit smallAddress bigAddress winner : Nat)
    (positive : 0 < start) (stop : start + count = limit) (distinct : smallAddress ≠ bigAddress)
    (env : Mutable.Env) (heap : Mutable.Heap)
    (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big))
    (locals : SearchLocals env limit smallAddress bigAddress winner) :
    ∃ finalHeap finalSmall finalBig finalEnv,
      finalHeap[smallAddress]? = some (.list finalSmall) ∧
      finalHeap[bigAddress]? = some (.dict finalBig) ∧
      Cache.Valid limit finalSmall finalBig ∧
      SearchLocals finalEnv limit smallAddress bigAddress
        ((List.range' start count).foldl (ProjectEulerStatements.ArgMaximum.select collatzLen) winner) ∧
      finalBig.length ≤ big.length + rangeWork start count ∧
      BlockContinuesIn .word program env [.rangeStepLoop "n" start limit 1 searchBody []] heap
        (2000 * (big.length + rangeWork start count + 1) * (rangeWork start count + 1)) finalEnv finalHeap := by
  induction count generalizing start winner env heap small big with
  | zero =>
      refine ⟨heap, small, big, env, smallFound, bigFound, valid, locals, by simp, ?_⟩
      apply BlockContinuesIn.mono (small := 4) _ (by simp; omega)
      apply BlockContinuesIn.range_done (rest := BlockContinuesIn.nil)
      simp [Calls.rangeContinues, ← stop]
  | succ count ih =>
      obtain ⟨middleHeap, middleSmall, middleBig, middleEnv, middleSmallFound, middleBigFound, middleValid,
          middleLocals, middleGrowth, iteration⟩ := search_one limit start smallAddress bigAddress winner positive distinct
          env heap small big valid smallFound bigFound locals
      obtain ⟨finalHeap, finalSmall, finalBig, finalEnv, finalSmallFound, finalBigFound, finalValid, finalLocals, finalGrowth, recurse⟩ :=
        ih (start + 1) (ProjectEulerStatements.ArgMaximum.select collatzLen winner start) (by omega) (by omega)
          middleEnv middleHeap middleSmall middleBig middleValid middleSmallFound middleBigFound middleLocals
      refine ⟨finalHeap, finalSmall, finalBig, finalEnv, finalSmallFound, finalBigFound, finalValid, ?_, ?_, ?_⟩
      · simpa only [List.range'_succ, List.foldl_cons] using finalLocals
      · rw [rangeWork_succ]
        omega
      · have rest : BlockContinuesIn .word program middleEnv
            [Calls.loopContinuation searchBody (.rangeStepLoop "n" ((start : Int) + 1) limit 1 searchBody [])]
            middleHeap (2000 * (middleBig.length + rangeWork (start + 1) count + 1) *
              (rangeWork (start + 1) count + 1)) finalEnv finalHeap := by
          simpa [Calls.loopContinuation, searchBody] using recurse
        have run := BlockContinuesIn.range_step (next := (start : Int)) (stop := (limit : Int))
          (stride := 1) (by simp [Calls.rangeContinues]; omega) (iteration.append rest)
        apply run.mono
        rw [rangeWork_succ]
        let tail := rangeWork (start + 1) count
        let factor := big.length + (collatzLen start + 1 + tail) + 1
        have firstBound : (big.length + collatzLen start + 1) * (collatzLen start + 1) ≤
            factor * (collatzLen start + 1) := Nat.mul_le_mul_right _ (by dsimp [factor]; omega)
        have restBound : (middleBig.length + tail + 1) * (tail + 1) ≤ factor * (tail + 1) :=
          Nat.mul_le_mul_right _ (by dsimp [factor]; omega)
        have positivePart : 0 < factor * (collatzLen start + 1) :=
          Nat.mul_pos (by dsimp [factor]; omega) (by omega)
        change 1000 * (big.length + collatzLen start + 1) * (collatzLen start + 1) + 32 +
            2000 * (middleBig.length + tail + 1) * (tail + 1) + 4 ≤
            2000 * factor * ((collatzLen start + 1) + (tail + 1))
        rw [show 2000 * factor * ((collatzLen start + 1) + (tail + 1)) =
          2000 * factor * (collatzLen start + 1) + 2000 * factor * (tail + 1) from Nat.mul_add _ _ _]
        simp only [Nat.mul_assoc]
        omega

end solvers.«14».runtimeCost
