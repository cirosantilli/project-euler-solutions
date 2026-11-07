import solvers.«21.py.lean.d».FactorOddRuntime
import lang.py.FormalSW.HeapRuntimeCostResults
import lang.py.FormalSW.HeapRuntimeCostLoopControl

namespace solvers.«21».runtimeCost.Factors
open FormalSW.PyAst HeapCalls.Runtime

def tailBody : List Stmt := (executableBody definition.2).drop 6

theorem tailBody_eq : tailBody =
    [.ifThen (.compare (.name "n") [(.gt, .constant (.int 1))])
      [.augAssign (.name "sigma") .mult (.binOp (.constant (.int 1)) .add (.name "n"))] [],
     .return (.binOp (.name "sigma") .sub (.name "original"))] := rfl

theorem function_body_eq : executableBody definition.2 =
    [.ifThen (.compare (.name "n") [(.ltE, .constant (.int 1))])
        [.return (.constant (.int 0))] [],
     .assign [.name "original"] (.name "n"),
     .assign [.name "sigma"] (.constant (.int 1)),
     passStatement false,
     .assign [.name "p"] (.constant (.int 3)),
     oddLoop] ++ tailBody := rfl

theorem tail_cost (n original : Nat) (sigma : Int) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (current : env "n" = some (number n)) (saved : env "original" = some (number original))
    (total : env "sigma" = some (number sigma)) :
    ∃ value : Int, BlockReturnsIn .word program env tailBody heap 28 (number value) heap := by
  have test : EvaluatesIn .word program env
      (.compare (.name "n") [(.gt, .constant (.int 1))]) heap 7
      (.scalar (.bool (decide (1 < n)))) heap := by
    apply EvaluatesIn.compare (work := 2) (EvaluatesIn.lookup current) EvaluatesIn.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, number]
    · rfl
  by_cases large : 1 < n
  · let finalEnv := HeapCalls.setLocal env "sigma" (number (sigma * (1 + (n : Int))))
    have multiply : BlockContinuesIn .word program env
        [.augAssign (.name "sigma") .mult (.binOp (.constant (.int 1)) .add (.name "n"))]
        heap 12 finalEnv heap := by
      apply BlockContinuesIn.augAssign (work := 1) (right := .int (1 + (n : Int))) total
        (EvaluatesIn.binary (work := 1) EvaluatesIn.constant (EvaluatesIn.lookup current)
          (by simp [Mutable.binary, scalarBinary]) rfl)
      · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
      · rfl
    have returned : BlockReturnsIn .word program finalEnv
        [.return (.binOp (.name "sigma") .sub (.name "original"))] heap 7
        (number (sigma * (1 + (n : Int)) - original)) heap := by
      apply BlockReturnsIn.return_value
      apply EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1)
        (av := .int (sigma * (1 + (n : Int))))
      · exact EvaluatesIn.lookup (by simp [finalEnv, HeapCalls.setLocal, number])
      · exact EvaluatesIn.lookup (by simpa [finalEnv, HeapCalls.setLocal] using saved)
      · simp [Mutable.binary, scalarBinary, number]
      · rfl
    refine ⟨sigma * (1 + (n : Int)) - original, ?_⟩
    rw [tailBody_eq]
    exact BlockReturnsIn.branch (cc := 7) (cb := 19) (flag := true)
      (by simpa [large] using test) rfl (multiply.returns returned)
  · have returned : BlockReturnsIn .word program env
        [.return (.binOp (.name "sigma") .sub (.name "original"))] heap 7
        (number (sigma - original)) heap :=
      BlockReturnsIn.return_value (EvaluatesIn.binary (work := 1)
        (EvaluatesIn.lookup total) (EvaluatesIn.lookup saved)
        (by simp [Mutable.binary, scalarBinary, number]) rfl)
    refine ⟨sigma - original, ?_⟩
    rw [tailBody_eq]
    exact (BlockReturnsIn.branch (cc := 7) (cb := 7) (flag := false)
      (by simpa [large] using test) rfl returned).mono (by omega)

/-- Bound the original factorization helper on inputs below an arbitrary square.
The heap and every returned integer are accounted for by source execution. -/
theorem function_run (n bound : Nat) (square : n ≤ bound * bound) (heap : Mutable.Heap) :
    ∃ value : Int, ReturnsIn .word program "proper_divisor_sum_factorization" [number n] heap
      (62 * (bound + 1) + 35 * n.log2 + 112) (number value) heap := by
  let entry := HeapCalls.functionEnv definition.1 definition.2 [number n] []
  have current : entry "n" = some (number n) := rfl
  have test : EvaluatesIn .word program entry
      (.compare (.name "n") [(.ltE, .constant (.int 1))]) heap 7
      (.scalar (.bool (decide (n ≤ 1)))) heap := by
    apply EvaluatesIn.compare (work := 2) (EvaluatesIn.lookup current) EvaluatesIn.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, number]
    · rfl
  by_cases small : n ≤ 1
  · refine ⟨0, ?_⟩
    apply ReturnsIn.mono (small := 14)
    · apply BlockReturnsIn.function (args := definition.1) (body := definition.2)
        (values := [number n]) (completed := [number n]) (budget := 11) rfl rfl
      rw [function_body_eq]
      exact BlockReturnsIn.branch (cc := 7) (cb := 2) (flag := true)
        (by simpa [small] using test) rfl (BlockReturnsIn.return_value EvaluatesIn.constant)
    · omega
  · let savedEnv := HeapCalls.setLocal entry "original" (number n)
    let initialized := HeapCalls.setLocal savedEnv "sigma" (number 1)
    have save : BlockContinuesIn .word program entry [.assign [.name "original"] (.name "n")]
        heap 3 savedEnv heap := BlockContinuesIn.assign (EvaluatesIn.lookup current)
    have initialRun : BlockContinuesIn .word program savedEnv
        [.assign [.name "sigma"] (.constant (.int 1))] heap 3 initialized heap :=
      BlockContinuesIn.assign EvaluatesIn.constant
    have initialLocals : PassLocals initialized false n 2 1 := by
      refine ⟨?_, ?_, rfl⟩
      · simpa [initialized, savedEnv, HeapCalls.setLocal] using current
      · simp [initialized, HeapCalls.setLocal]
    obtain ⟨nextN, nextSigma, nextEnv, positive, nextBound, nextLocals, passPreserves, passRun⟩ :=
      pass_cost false n 2 1 (by omega) (by decide) heap initialized initialLocals
    let oddEnv := HeapCalls.setLocal nextEnv "p" (number 3)
    have setCandidate : BlockContinuesIn .word program nextEnv
        [.assign [.name "p"] (.constant (.int 3))] heap 3 oddEnv heap :=
      BlockContinuesIn.assign EvaluatesIn.constant
    have oddLocals : PassLocals oddEnv true nextN 3 nextSigma := by
      refine ⟨?_, ?_, ?_⟩
      · simpa [oddEnv, HeapCalls.setLocal] using nextLocals.current
      · simpa [oddEnv, HeapCalls.setLocal] using nextLocals.sigma
      · simp [FactorValue, oddEnv, HeapCalls.setLocal]
    obtain ⟨residue, finalP, finalSigma, finalEnv, _, residueBound, finalLocals,
        loopPreserves, loopRun⟩ := odd_loop_cost (bound + 1) bound nextN 3 nextSigma
          positive (by decide) (nextBound.trans square) (by omega) heap oddEnv oddLocals
    have saved : finalEnv "original" = some (number n) := by
      rw [loopPreserves "original" (by decide) (by decide) (by decide) (by decide) (by decide)]
      change nextEnv "original" = some (number n)
      rw [passPreserves "original" (by decide) (by decide) (by decide) (by decide)]
      simp [initialized, savedEnv, HeapCalls.setLocal]
    obtain ⟨value, tail⟩ := tail_cost residue n finalSigma heap finalEnv
      finalLocals.current saved finalLocals.sigma
    have body := save.returns (initialRun.returns (passRun.returns
      (setCandidate.returns (loopRun.returns tail))))
    have branch : BlockReturnsIn .word program entry (executableBody definition.2) heap
        (7 + (3 + (3 + (35 * (n.log2 - nextN.log2) + 41 +
          (3 + (62 * (bound + 1) + 35 * (nextN.log2 - residue.log2) + 14 + 28))))) + 2)
        (number value) heap := by
      rw [function_body_eq]
      exact BlockReturnsIn.branch (cc := 7) (flag := false)
        (by simpa [small] using test) rfl body
    refine ⟨value, ?_⟩
    apply ReturnsIn.mono
    · apply BlockReturnsIn.function (args := definition.1) (body := definition.2)
        (completed := [number n]) rfl rfl
      exact branch
    · have drop : nextN.log2 ≤ n.log2 := by
        simpa only [Nat.log2_eq_log_two] using Nat.log_mono_right (b := 2) nextBound
      simp only [List.length_cons, List.length_nil, show definition.1.length = 1 from rfl]
      omega

/-- A linear envelope in the square bound is convenient for divisor sums of
scan candidates: `d a ≤ a*a` makes the fallback cost at most `250*(a+1)`. -/
theorem function_bound (n bound : Nat) (square : n ≤ bound * bound) (heap : Mutable.Heap) :
    ∃ value : Int, ReturnsIn .word program "proper_divisor_sum_factorization" [number n] heap
      (250 * (bound + 1)) (number value) heap := by
  obtain ⟨value, run⟩ := function_run n bound square heap
  refine ⟨value, run.mono ?_⟩
  have logarithm : n.log2 ≤ (bound * bound).log2 := by
    simpa only [Nat.log2_eq_log_two] using Nat.log_mono_right (b := 2) square
  have boundLog := ProjectEulerStatements.TrialDivisionWork.log_square_linear bound
  omega

end solvers.«21».runtimeCost.Factors
