import «.formalsw».solvers.«37.py.ast»
import lang.py.FormalSW.HeapCalls
import Mathlib.Data.Nat.Sqrt
import Mathlib.Order.Interval.Finset.Nat
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Int.Basic

set_option Elab.async false

namespace solvers.«37».is_prime
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P37Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)

private def Trial (n r p : Nat) : Prop :=
  ∀ q ∈ Finset.Icc p r, q % 2 = 1 → ¬ q ∣ n

private instance (n r p : Nat) : Decidable (Trial n r p) := by
  unfold Trial
  infer_instance

private theorem trial_step (n r p : Nat) (hle : p ≤ r) (odd : p % 2 = 1) :
    Trial n r p ↔ ¬p ∣ n ∧ Trial n r (p + 2) := by
  simp only [Trial, Finset.mem_Icc]
  constructor
  · intro h
    exact ⟨h p ⟨le_rfl, hle⟩ odd, fun q hq hodd => h q ⟨by omega, hq.2⟩ hodd⟩
  · rintro ⟨hp, h⟩ q hq hodd
    by_cases heq : q = p
    · simpa [heq] using hp
    · exact h q ⟨by omega, hq.2⟩ hodd

private theorem trial_prime (n : Nat) (hn : 2 ≤ n) (odd : n % 2 ≠ 0) :
    Trial n (Math.isqrt n) 3 ↔ Nat.Prime n := by
  have root : Math.isqrt n = Nat.sqrt n := Nat.eq_sqrt.mpr (Math.isqrt_spec n)
  rw [root, Nat.prime_def_le_sqrt]
  constructor
  · intro h
    refine ⟨hn, fun q hq hroot hdvd => ?_⟩
    by_cases even : q % 2 = 0
    · exact odd (Nat.dvd_iff_mod_eq_zero.mp (dvd_trans (Nat.dvd_iff_mod_eq_zero.mpr even) hdvd))
    · exact h q (Finset.mem_Icc.mpr ⟨by omega, hroot⟩) (by omega) hdvd
  · rintro ⟨_, h⟩ q hq _
    exact h q (by have := (Finset.mem_Icc.mp hq).1; omega) (Finset.mem_Icc.mp hq).2

private abbrev body : List Stmt :=
  [.ifThen (.compare (.binOp (.name "n") .mod (.name "divisor")) [(.eq, .constant (.int 0))])
    [.return (.constant (.bool false))] []]

private theorem loop (n r p : Nat) (positive : 0 < p) (odd : p % 2 = 1)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (getN : env "n" = some (number n)) :
    BlockReturns program env
      [.rangeStepLoop "divisor" p (r + 1) 2 body [], .return (.constant (.bool true))]
      heap (flag (decide (Trial n r p))) heap := by
  induction measure : r + 1 - p using Nat.strong_induction_on generalizing p env with
  | h size ih =>
      by_cases inside : p ≤ r
      · apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
        let bound := setLocal env "divisor" (number p)
        have divides : Evaluates program bound
            (.compare (.binOp (.name "n") .mod (.name "divisor")) [(.eq, .constant (.int 0))])
            heap (flag (decide (p ∣ n))) heap := by
          apply Evaluates.compare (av := number (n % p)) (bv := number 0)
            (middle := heap) (afterRight := heap)
          · exact Evaluates.binary (av := number n) (bv := number p)
              (middle := heap) (afterRight := heap) (finalHeap := heap)
              (Evaluates.lookup (by simpa [bound, setLocal] using getN))
              (Evaluates.lookup (by simp [bound, setLocal]))
              (by simp [Mutable.binary, scalarBinary, positive, positive.ne'])
          · exact Evaluates.constant
          · simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp,
              ← Int.dvd_iff_emod_eq_zero, Int.natCast_dvd_natCast]
        by_cases divisor : p ∣ n
        · have bad : ¬ Trial n r p := fun h => ((trial_step n r p inside odd).mp h).1 divisor
          simp only [bad, decide_false]
          apply BlockReturns.branch (middle := heap) (flag := true) (value := flag true)
            (by simpa [divisor] using divides) rfl
          exact BlockReturns.return_value Evaluates.constant
        · simp only [trial_step n r p inside odd, divisor, not_false_eq_true, true_and]
          apply BlockReturns.branch (middle := heap) (flag := false) (value := flag false)
            (by simpa [divisor] using divides) rfl
          have rest := ih (r + 1 - (p + 2)) (by omega) (p + 2) (by omega) (by omega) bound
            (by simpa [bound, setLocal] using getN) rfl
          simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, body] using rest
      · have good : Trial n r p := by
          intro q hq _
          have := Finset.mem_Icc.mp hq
          omega
        simp only [good, decide_true]
        apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
        exact BlockReturns.return_value Evaluates.constant

/-- Trial division through the integer square root is pure for every natural input. -/
theorem result (n : Nat) (heap : Mutable.Heap) :
    ReturnsValue program "is_prime" [number n] heap (flag (decide (Nat.Prime n))) heap := by
  apply BlockReturns.function rfl rfl
  by_cases small : n < 2
  · have bad : ¬ Nat.Prime n := fun prime => by have := prime.two_le; omega
    simp only [bad, decide_false]
    apply BlockReturns.branch (middle := heap) (flag := true) (value := flag true)
      (Evaluates.compare (av := number n) (bv := number 2) (middle := heap) (afterRight := heap)
        (Evaluates.lookup rfl) Evaluates.constant
        (by simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]; omega)) rfl
    exact BlockReturns.return_value Evaluates.constant
  · apply BlockReturns.branch (middle := heap) (flag := false) (value := flag false)
      (Evaluates.compare (av := number n) (bv := number 2) (middle := heap) (afterRight := heap)
        (Evaluates.lookup rfl) Evaluates.constant
        (by simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]; omega)) rfl
    by_cases two : n = 2
    · subst n
      simp only [Nat.prime_two, decide_true]
      apply BlockReturns.branch (middle := heap) (flag := true) (value := flag true)
        (Evaluates.compare (av := number 2) (bv := number 2) (middle := heap) (afterRight := heap)
          (Evaluates.lookup rfl) Evaluates.constant rfl) rfl
      exact BlockReturns.return_value Evaluates.constant
    · apply BlockReturns.branch (middle := heap) (flag := false) (value := flag false)
        (Evaluates.compare (av := number n) (bv := number 2) (middle := heap) (afterRight := heap)
          (Evaluates.lookup rfl) Evaluates.constant
          (by simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]; omega)) rfl
      let env := functionEnv ((findFunctionIn program.body "is_prime").getD ([], [])).1
        ((findFunctionIn program.body "is_prime").getD ([], [])).2 [number n] []
      have parity : Evaluates program env
          (.compare (.binOp (.name "n") .mod (.constant (.int 2))) [(.eq, .constant (.int 0))])
          heap (flag (decide (n % 2 = 0))) heap := by
        apply Evaluates.compare (av := number (n % 2)) (bv := number 0)
          (middle := heap) (afterRight := heap)
        · exact Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant
            (by simp [Mutable.binary, scalarBinary, number])
        · exact Evaluates.constant
        · simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp,
            ← Int.dvd_iff_emod_eq_zero, Int.ofNat_dvd_natCast, Nat.dvd_iff_mod_eq_zero]
      by_cases even : n % 2 = 0
      · have bad : ¬ Nat.Prime n := fun prime => by have := prime.eq_two_or_odd; omega
        simp only [bad, decide_false]
        apply BlockReturns.branch (middle := heap) (flag := true) (value := flag true)
          (by simpa [even] using parity) rfl
        exact BlockReturns.return_value Evaluates.constant
      · apply BlockReturns.branch (middle := heap) (flag := false) (value := flag false)
          (by simpa [even] using parity) rfl
        apply BlockReturns.for_range (values := [number 3, number (Math.isqrt n + 1), number 2])
          (middle := heap)
        · apply Evaluates.Arguments.cons (middle := heap) Evaluates.constant
          apply Evaluates.Arguments.cons (middle := heap)
          · apply Evaluates.binary (av := number (Math.isqrt n)) (bv := number 1)
              (middle := heap) (afterRight := heap)
            · apply Evaluates.library_call (.cons (Evaluates.lookup rfl) .nil) rfl
              change scalarResult (some (Math.isqrtCall [.int n])) heap = _
              simp [Math.isqrtCall, scalarResult, number, show ¬ (n : Int) < 0 by omega]
            · exact Evaluates.constant
            · simp [Mutable.binary, scalarBinary, number]
          · exact .cons Evaluates.constant .nil
        · rfl
        · simpa [trial_prime n (by omega) even] using
            loop n (Math.isqrt n) 3 (by decide) (by decide) heap env rfl

end solvers.«37».is_prime
