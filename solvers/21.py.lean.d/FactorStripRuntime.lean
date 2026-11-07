import solvers.«21.py.lean.d».FactorSource
import lang.py.FormalSW.HeapRuntimeCostSequences
import ProjectEulerStatements.Lib.TrialDivisionWork

/-! Runtime of the actual factorization stripping loops. The accumulators are
arbitrary integers; only the positive residue supplies logarithmic potential. -/
namespace solvers.«21».runtimeCost.Factors
open FormalSW.PyAst HeapCalls.Runtime

structure StripLocals (env : HeapCalls.Env) (named : Bool) (n p : Nat)
    (power term : Int) : Prop where
  current : env "n" = some (number n)
  power : env "p_pow" = some (number power)
  term : env "term" = some (number term)
  factor : FactorValue named p env

theorem factor_cost (named : Bool) (p : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (factor : FactorValue named p env) :
    EvaluatesIn .word program env (factorExpr named) heap 1 (number p) heap := by
  cases named with
  | false =>
      have equal : p = 2 := factor
      subst p
      exact EvaluatesIn.constant
  | true => exact EvaluatesIn.lookup factor

theorem strip_test (named : Bool) (n p : Nat) (power term : Int) (positive : 0 < p)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (locals : StripLocals env named n p power term) :
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

def stripStepEnv (env : HeapCalls.Env) (n p : Nat) (power term : Int) : HeapCalls.Env :=
  HeapCalls.setLocal (HeapCalls.setLocal (HeapCalls.setLocal env "n" (number (n / p : Nat)))
    "p_pow" (number (power * (p : Int)))) "term" (number (term + power * (p : Int)))

theorem strip_step_preserves (env : HeapCalls.Env) (n p : Nat) (power term : Int)
    (name : String) (notN : name ≠ "n") (notPower : name ≠ "p_pow") (notTerm : name ≠ "term") :
    stripStepEnv env n p power term name = env name := by
  simp [stripStepEnv, HeapCalls.setLocal, notN, notPower, notTerm]

theorem StripLocals.step {env : HeapCalls.Env} {named : Bool} {n p : Nat} {power term : Int}
    (locals : StripLocals env named n p power term) :
    StripLocals (stripStepEnv env n p power term) named (n / p) p
      (power * (p : Int)) (term + power * (p : Int)) := by
  refine ⟨?_, ?_, ?_, locals.factor.preserve ?_⟩
  · simp [stripStepEnv, HeapCalls.setLocal]
  · simp [stripStepEnv, HeapCalls.setLocal]
  · simp [stripStepEnv, HeapCalls.setLocal]
  · exact strip_step_preserves env n p power term "p" (by decide) (by decide) (by decide)

/-- The addition reads the updated `p_pow`, after multiplication has run. -/
theorem strip_one (named : Bool) (n p : Nat) (power term : Int) (positive : 0 < p)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (locals : StripLocals env named n p power term) :
    BlockContinuesIn .word program env (stripBody named) heap 21
      (stripStepEnv env n p power term) heap := by
  let divided := HeapCalls.setLocal env "n" (number (n / p : Nat))
  let powered := HeapCalls.setLocal divided "p_pow" (number (power * (p : Int)))
  have divide : BlockContinuesIn .word program env
      [.augAssign (.name "n") .floorDiv (factorExpr named)] heap 7 divided heap := by
    apply BlockContinuesIn.augAssign (work := 1) locals.current
      (factor_cost named p heap env locals.factor)
    · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number, positive, positive.ne']
    · rfl
  have multiply : BlockContinuesIn .word program divided
      [.augAssign (.name "p_pow") .mult (factorExpr named)] heap 7 powered heap := by
    apply BlockContinuesIn.augAssign (work := 1)
      (by simpa [divided, HeapCalls.setLocal] using locals.power)
      (factor_cost named p heap divided (locals.factor.preserve (by simp [divided, HeapCalls.setLocal])))
    · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
    · rfl
  have add : BlockContinuesIn .word program powered
      [.augAssign (.name "term") .add (.name "p_pow")] heap 7
      (stripStepEnv env n p power term) heap := by
    apply BlockContinuesIn.augAssign (work := 1)
      (left := .int term) (right := .int (power * (p : Int)))
      (by simpa [powered, divided, HeapCalls.setLocal] using locals.term)
      (EvaluatesIn.lookup (by simp [powered, HeapCalls.setLocal, number]))
    · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
    · rfl
  simpa only [stripBody_eq, List.cons_append, List.nil_append] using (divide.append multiply).append add

/-- Every successful division consumes at least one logarithmic unit. The
remaining potential can be reused by later candidate passes. -/
theorem strip_cost (named : Bool) (n p : Nat) (power term : Int)
    (positive : 0 < n) (large : 2 ≤ p)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (locals : StripLocals env named n p power term) :
    ∃ (residue : Nat) (finalPower finalTerm : Int) (finalEnv : HeapCalls.Env),
      0 < residue ∧ residue ≤ n ∧ ¬ p ∣ residue ∧
      StripLocals finalEnv named residue p finalPower finalTerm ∧
      (∀ name, name ≠ "n" → name ≠ "p_pow" → name ≠ "term" → finalEnv name = env name) ∧
      BlockContinuesIn .word program env [stripLoop named] heap
        (35 * (n.log2 - residue.log2) + 14) finalEnv heap := by
  have factorPositive : 0 < p := by omega
  have tested := strip_test named n p power term factorPositive heap env locals
  by_cases divides : p ∣ n
  · have factorBound : p ≤ n := Nat.le_of_dvd positive divides
    have quotientPositive : 0 < n / p := Nat.div_pos factorBound factorPositive
    obtain ⟨residue, finalPower, finalTerm, finalEnv, finalPositive, residueBound,
        removed, finalLocals, preserved, rest⟩ :=
      strip_cost named (n / p) p (power * (p : Int)) (term + power * (p : Int))
        quotientPositive large heap (stripStepEnv env n p power term) locals.step
    refine ⟨residue, finalPower, finalTerm, finalEnv, finalPositive,
      residueBound.trans (Nat.div_le_self n p), removed, finalLocals, ?_, ?_⟩
    · intro name notN notPower notTerm
      exact (preserved name notN notPower notTerm).trans
        (strip_step_preserves env n p power term name notN notPower notTerm)
    · have one := strip_one named n p power term factorPositive heap env locals
      have paid : BlockContinuesIn .word program env [stripLoop named] heap
          (12 + (21 + (35 * ((n / p).log2 - residue.log2) + 14)) + 2) finalEnv heap := by
        rw [stripLoop_eq]
        apply BlockContinuesIn.while_loop (cc := 12) (flag := true)
          (by simpa [divides] using tested) rfl
        simpa only [stripBody_continuation, ← stripLoop_eq] using one.append rest
      apply paid.mono
      have drop := ProjectEulerStatements.TrialDivisionWork.log_div_decrease n p (by omega) large
      have logarithm : residue.log2 ≤ (n / p).log2 := by
        simpa only [Nat.log2_eq_log_two] using Nat.log_mono_right (b := 2) residueBound
      omega
  · refine ⟨n, power, term, env, positive, le_rfl, divides, locals, fun _ _ _ _ => rfl, ?_⟩
    simp only [Nat.sub_self, Nat.mul_zero, Nat.zero_add]
    rw [stripLoop_eq]
    exact BlockContinuesIn.while_loop (cc := 12) (cb := 0) (flag := false)
      (by simpa [divides] using tested) rfl BlockContinuesIn.nil
termination_by n
decreasing_by exact Nat.div_lt_self positive (by omega)

end solvers.«21».runtimeCost.Factors
