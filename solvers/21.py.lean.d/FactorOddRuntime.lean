import solvers.«21.py.lean.d».FactorPassRuntime
import Mathlib.Tactic.Linarith

namespace solvers.«21».runtimeCost.Factors
open FormalSW.PyAst HeapCalls.Runtime

theorem odd_test (n p : Nat) (sigma : Int) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (locals : PassLocals env true n p sigma) :
    EvaluatesIn .word program env oddCondition heap 12
      (.scalar (.bool (decide (p * p ≤ n)))) heap := by
  rw [oddCondition_eq]
  apply EvaluatesIn.compare (work := 2) (av := number (p * p : Nat)) (middle := heap)
    (EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup locals.factor)
      (EvaluatesIn.lookup locals.factor)
      (by simp [Mutable.binary, scalarBinary, number]) rfl)
    (EvaluatesIn.lookup locals.current)
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, number]
    simp only [← Int.natCast_mul]
    omega
  · rfl

theorem oddBody_continuation (next : Stmt) :
    Calls.loopContinuation oddBody next = next := rfl

/-- An arbitrary square envelope bounds the candidate count. Stripping costs
are paid once across all passes by the decreasing residue logarithm. -/
theorem odd_loop_cost (count bound n p : Nat) (sigma : Int)
    (positive : 0 < n) (large : 2 ≤ p) (square : n ≤ bound * bound)
    (covers : bound < p + count) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (locals : PassLocals env true n p sigma) :
    ∃ (residue finalP : Nat) (finalSigma : Int) (finalEnv : HeapCalls.Env),
      0 < residue ∧ residue ≤ n ∧ PassLocals finalEnv true residue finalP finalSigma ∧
      (∀ name, name ≠ "n" → name ≠ "p_pow" → name ≠ "term" → name ≠ "sigma" →
        name ≠ "p" → finalEnv name = env name) ∧
      BlockContinuesIn .word program env [oddLoop] heap
        (62 * count + 35 * (n.log2 - residue.log2) + 14) finalEnv heap := by
  induction count generalizing n p sigma env with
  | zero =>
      have stops : ¬ p * p ≤ n := by simp only [Nat.add_zero] at covers; nlinarith
      refine ⟨n, p, sigma, env, positive, le_rfl, locals, fun _ _ _ _ _ _ => rfl, ?_⟩
      simp only [Nat.mul_zero, Nat.sub_self, Nat.zero_add]
      rw [oddLoop_eq]
      exact BlockContinuesIn.while_loop (cc := 12) (cb := 0) (flag := false)
        (by simpa [stops] using odd_test n p sigma heap env locals) rfl BlockContinuesIn.nil
  | succ count ih =>
      by_cases continues : p * p ≤ n
      · obtain ⟨nextN, nextSigma, nextEnv, nextPositive, nextBound, nextLocals,
            passPreserves, passRun⟩ := pass_cost true n p sigma positive large heap env locals
        let advanced := HeapCalls.setLocal nextEnv "p" (number (p + 2 : Nat))
        have increment : BlockContinuesIn .word program nextEnv
            [.augAssign (.name "p") .add (.constant (.int 2))] heap 7 advanced heap := by
          apply BlockContinuesIn.augAssign (work := 1) (left := .int p) (right := .int 2)
            nextLocals.factor EvaluatesIn.constant
          · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
          · rfl
        have advancedLocals : PassLocals advanced true nextN (p + 2) nextSigma := by
          refine ⟨?_, ?_, ?_⟩
          · simpa [advanced, HeapCalls.setLocal] using nextLocals.current
          · simpa [advanced, HeapCalls.setLocal] using nextLocals.sigma
          · simp [FactorValue, advanced, HeapCalls.setLocal]
        obtain ⟨residue, finalP, finalSigma, finalEnv, finalPositive, finalBound,
            finalLocals, restPreserves, rest⟩ := ih nextN (p + 2) nextSigma nextPositive
              (by omega) (nextBound.trans square) (by omega) advanced advancedLocals
        refine ⟨residue, finalP, finalSigma, finalEnv, finalPositive,
          finalBound.trans nextBound, finalLocals, ?_, ?_⟩
        · intro name notN notPower notTerm notSigma notP
          calc
            finalEnv name = advanced name := restPreserves name notN notPower notTerm notSigma notP
            _ = nextEnv name := by simp [advanced, HeapCalls.setLocal, notP]
            _ = env name := passPreserves name notN notPower notTerm notSigma
        · have body := (passRun.append increment).append rest
          have paid : BlockContinuesIn .word program env [oddLoop] heap
              (12 + ((35 * (n.log2 - nextN.log2) + 41 + 7) +
                (62 * count + 35 * (nextN.log2 - residue.log2) + 14)) + 2) finalEnv heap := by
            rw [oddLoop_eq]
            apply BlockContinuesIn.while_loop (cc := 12) (flag := true)
              (by simpa [continues] using odd_test n p sigma heap env locals) rfl
            simpa only [oddBody_continuation, ← oddLoop_eq, oddBody_eq,
              List.cons_append, List.nil_append] using body
          apply paid.mono
          have firstDrop : nextN.log2 ≤ n.log2 := by
            simpa only [Nat.log2_eq_log_two] using Nat.log_mono_right (b := 2) nextBound
          have secondDrop : residue.log2 ≤ nextN.log2 := by
            simpa only [Nat.log2_eq_log_two] using Nat.log_mono_right (b := 2) finalBound
          omega
      · refine ⟨n, p, sigma, env, positive, le_rfl, locals, fun _ _ _ _ _ _ => rfl, ?_⟩
        apply BlockContinuesIn.mono (small := 14)
        · rw [oddLoop_eq]
          exact BlockContinuesIn.while_loop (cc := 12) (cb := 0) (flag := false)
            (by simpa [continues] using odd_test n p sigma heap env locals) rfl BlockContinuesIn.nil
        · omega

end solvers.«21».runtimeCost.Factors
