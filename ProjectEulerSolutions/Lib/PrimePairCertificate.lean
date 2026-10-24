import ProjectEulerSolutions.Lib.DecimalShift
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.TrialPrimeCertificate

namespace ProjectEulerSolutions.DecimalShift

/-- A finite arithmetic form for certificate inputs below 100,000. -/
def fiveDigits (n : Nat) : Nat :=
  if n < 1 then 1 else if n < 10 then 10 else if n < 100 then 100
  else if n < 1000 then 1000 else if n < 10000 then 10000 else 100000

theorem fiveDigits_eq (n : Nat) (bounded : n < 100000) : decimalShift n = fiveDigits n := by
  unfold fiveDigits decimalShift
  split_ifs with zero one two three four
  · exact from_done n 1 (by decide) (by omega)
  · rw [from_step n 1 (by decide) (by omega)]
    exact from_done n 10 (by decide) (by omega)
  · rw [from_step n 1 (by decide) (by omega), from_step n 10 (by decide) (by omega)]
    exact from_done n 100 (by decide) (by omega)
  · rw [from_step n 1 (by decide) (by omega), from_step n 10 (by decide) (by omega),
      from_step n 100 (by decide) (by omega)]
    exact from_done n 1000 (by decide) (by omega)
  · rw [from_step n 1 (by decide) (by omega), from_step n 10 (by decide) (by omega),
      from_step n 100 (by decide) (by omega), from_step n 1000 (by decide) (by omega)]
    exact from_done n 10000 (by decide) (by omega)
  · rw [from_step n 1 (by decide) (by omega), from_step n 10 (by decide) (by omega),
      from_step n 100 (by decide) (by omega), from_step n 1000 (by decide) (by omega),
      from_step n 10000 (by decide) (by omega)]
    exact from_done n 100000 (by decide) (by omega)

end ProjectEulerSolutions.DecimalShift

namespace ProjectEulerSolutions.PrimePairCertificate

/-- This finite certificate relation uses the source Miller–Rabin algorithm.
Only decimal shifting has been replaced by its proved five-digit arithmetic form. -/
def relation (p q : Nat) : Bool :=
  if (p + q) % 3 = 0 then false
  else MillerRabin.isPrimeModel (p * DecimalShift.fiveDigits q + q) &&
    MillerRabin.isPrimeModel (q * DecimalShift.fiveDigits p + p)

/-- Prime vertices are classified in short independently checked intervals. -/
def primeSegment (start count : Nat) : List Nat :=
  (List.range' start count).filter (fun n => decide (Nat.Prime n ∧ n ≠ 2 ∧ n ≠ 5))

theorem primeSegment_append (start left right : Nat) :
    primeSegment start (left + right) = primeSegment start left ++ primeSegment (start + left) right := by
  unfold primeSegment
  rw [← List.range'_append_1 (s := start) (m := left) (n := right), List.filter_append]

/-- The executable classifier keeps interval verification shallow. -/
def computedSegment (start count : Nat) : List Nat :=
  (List.range' start count).filter (fun n => TrialPrimeCertificate.smallPrime n && decide (n ≠ 2 ∧ n ≠ 5))

theorem computedSegment_eq_primeSegment (start count : Nat) (bounded : start + count ≤ 258 * 258) :
    computedSegment start count = primeSegment start count := by
  unfold computedSegment primeSegment
  apply List.filter_congr
  intro n member
  obtain ⟨i, inside, same⟩ := List.mem_range'.mp member
  rw [Bool.eq_iff_iff]
  simp only [Bool.and_eq_true, decide_eq_true_eq, TrialPrimeCertificate.smallPrime_iff n (by omega)]

end ProjectEulerSolutions.PrimePairCertificate
