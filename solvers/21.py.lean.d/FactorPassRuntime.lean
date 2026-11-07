import solvers.«21.py.lean.d».FactorStripRuntime
import lang.py.FormalSW.HeapRuntimeCostRanges

/-! A source factor pass either skips the candidate or initializes, strips and
multiplies the accumulated term. Integer magnitudes do not affect word costs. -/
namespace solvers.«21».runtimeCost.Factors
open FormalSW.PyAst HeapCalls.Runtime

structure PassLocals (env : HeapCalls.Env) (named : Bool) (n p : Nat) (sigma : Int) : Prop where
  current : env "n" = some (number n)
  sigma : env "sigma" = some (number sigma)
  factor : FactorValue named p env

/-- The outer divisibility test does not read the uninitialized strip locals. -/
theorem pass_test (named : Bool) (n p : Nat) (sigma : Int) (positive : 0 < p)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (locals : PassLocals env named n p sigma) :
    EvaluatesIn .word program env (stripCondition named) heap 12
      (.scalar (.bool (decide (p ∣ n)))) heap := by
  rw [stripCondition_eq]
  apply EvaluatesIn.compare (work := 2) (av := number (n % p : Nat)) (middle := heap)
    (EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup locals.current)
      (factor_cost named p heap env locals.factor)
      (by simp [Mutable.binary, scalarBinary, number, positive, positive.ne']) rfl)
    EvaluatesIn.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, number,
      ← Int.dvd_iff_emod_eq_zero, Int.natCast_dvd_natCast]
  · rfl

theorem pass_skip (named : Bool) (n p : Nat) (sigma : Int) (positive : 0 < p)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (locals : PassLocals env named n p sigma)
    (absent : ¬ p ∣ n) :
    BlockContinuesIn .word program env [passStatement named] heap 14 env heap := by
  rw [passStatement_eq]
  exact BlockContinuesIn.branch (cc := 12) (cb := 0) (flag := false)
    (by simpa [absent] using pass_test named n p sigma positive heap env locals)
    rfl BlockContinuesIn.nil

/-- The log drop pays for all stripping iterations; the constant covers both
guards, accumulator initialization and the final sigma multiplication. -/
theorem pass_cost (named : Bool) (n p : Nat) (sigma : Int)
    (positive : 0 < n) (large : 2 ≤ p)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (locals : PassLocals env named n p sigma) :
    ∃ (residue : Nat) (finalSigma : Int) (finalEnv : HeapCalls.Env),
      0 < residue ∧ residue ≤ n ∧ PassLocals finalEnv named residue p finalSigma ∧
      (∀ name, name ≠ "n" → name ≠ "p_pow" → name ≠ "term" → name ≠ "sigma" →
        finalEnv name = env name) ∧
      BlockContinuesIn .word program env [passStatement named] heap
        (35 * (n.log2 - residue.log2) + 41) finalEnv heap := by
  have factorPositive : 0 < p := by omega
  by_cases divides : p ∣ n
  · let withTerm := HeapCalls.setLocal env "term" (number 1)
    let initialized := HeapCalls.setLocal withTerm "p_pow" (number 1)
    have initialRun : BlockContinuesIn .word program env
        [.assign [.name "term"] (.constant (.int 1)),
          .assign [.name "p_pow"] (.constant (.int 1))] heap 6 initialized heap := by
      exact (BlockContinuesIn.assign (name := "term") EvaluatesIn.constant).append
        (BlockContinuesIn.assign (name := "p_pow") EvaluatesIn.constant)
    have stripLocals : StripLocals initialized named n p 1 1 := by
      refine ⟨?_, ?_, ?_, locals.factor.preserve ?_⟩
      · simpa [initialized, withTerm, HeapCalls.setLocal] using locals.current
      · simp [initialized, HeapCalls.setLocal]
      · simp [initialized, withTerm, HeapCalls.setLocal]
      · simp [initialized, withTerm, HeapCalls.setLocal]
    obtain ⟨residue, finalPower, finalTerm, strippedEnv, residuePositive, residueBound,
        _, strippedLocals, stripPreserves, stripRun⟩ :=
      strip_cost named n p 1 1 positive large heap initialized stripLocals
    have oldSigma : strippedEnv "sigma" = some (number sigma) :=
      (stripPreserves "sigma" (by decide) (by decide) (by decide)).trans
        (by simpa [initialized, withTerm, HeapCalls.setLocal] using locals.sigma)
    let finalEnv := HeapCalls.setLocal strippedEnv "sigma" (number (sigma * finalTerm))
    have multiplied : BlockContinuesIn .word program strippedEnv
        [.augAssign (.name "sigma") .mult (.name "term")] heap 7 finalEnv heap := by
      apply BlockContinuesIn.augAssign (work := 1) (left := .int sigma) (right := .int finalTerm)
        oldSigma (EvaluatesIn.lookup strippedLocals.term)
      · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
      · rfl
    have finalLocals : PassLocals finalEnv named residue p (sigma * finalTerm) := by
      refine ⟨?_, ?_, strippedLocals.factor.preserve ?_⟩
      · simpa [finalEnv, HeapCalls.setLocal] using strippedLocals.current
      · simp [finalEnv, HeapCalls.setLocal]
      · simp [finalEnv, HeapCalls.setLocal]
    refine ⟨residue, sigma * finalTerm, finalEnv, residuePositive, residueBound, finalLocals, ?_, ?_⟩
    · intro name notN notPower notTerm notSigma
      calc
        finalEnv name = strippedEnv name := by simp [finalEnv, HeapCalls.setLocal, notSigma]
        _ = initialized name := stripPreserves name notN notPower notTerm
        _ = env name := by simp [initialized, withTerm, HeapCalls.setLocal, notPower, notTerm]
    · have body : BlockContinuesIn .word program env (passBody named) heap
          (35 * (n.log2 - residue.log2) + 27) finalEnv heap := by
        apply BlockContinuesIn.mono (small := (6 + (35 * (n.log2 - residue.log2) + 14)) + 7)
        · simpa only [passBody_eq, List.cons_append, List.nil_append] using
            (initialRun.append stripRun).append multiplied
        · omega
      rw [passStatement_eq]
      have run := BlockContinuesIn.branch (cc := 12) (flag := true) (no := [])
        (by simpa [divides] using pass_test named n p sigma factorPositive heap env locals)
        rfl body
      exact run.mono (by omega)
  · refine ⟨n, sigma, env, positive, le_rfl, locals, fun _ _ _ _ _ => rfl, ?_⟩
    exact (pass_skip named n p sigma factorPositive heap env locals divides).mono (by omega)

end solvers.«21».runtimeCost.Factors
