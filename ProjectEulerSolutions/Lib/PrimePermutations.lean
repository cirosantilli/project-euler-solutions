import ProjectEulerStatements.P49
import lang.py.FormalSW.ScalarSort
import Mathlib.Data.List.Sort
import Mathlib.Data.List.Perm.Basic
import Mathlib.Data.Nat.Digits.Lemmas
import Mathlib.Data.List.Range
import Mathlib.Tactic.Ring
import Lean.Elab.Tactic.Omega

/-! Digit-multiset equality and exact arithmetic progressions for the prime-permutation search. -/
namespace ProjectEulerSolutions.PrimePermutations
open FormalSW.PyAst

set_option Elab.async false

def integerDigits (n : Nat) : List Int := (Nat.digits 10 n).map (fun (digit : Nat) => (digit : Int))
def sortedDigits (n : Nat) : List Int := ScalarSort.ints (integerDigits n)
def progression (a d : Int) (length : Nat) : List Int :=
  (List.range length).map (fun (i : Nat) => a+(i : Int)*d)

/-- The interpreter's stable integer sort equals the statement's sorted natural digits. -/
theorem sortedDigits_statement (n : Nat) :
    sortedDigits n=(ProjectEulerStatements.P49.digitsSorted n).map (fun (digit : Nat) => (digit : Int)) := by
  apply List.Perm.eq_of_pairwise (le := (· ≤ ·))
  · intro a b _ _ lower upper
    exact le_antisymm lower upper
  · exact ScalarSort.ints_ordered _
  · apply List.Pairwise.map (fun digit : Nat => (digit : Int)) (fun _ _ lower => Int.ofNat_le.mpr lower)
    exact List.pairwise_mergeSort' (· ≤ ·) (Nat.digits 10 n)
  · exact (ScalarSort.ints_perm (integerDigits n)).trans
      ((List.mergeSort_perm (Nat.digits 10 n) (fun a b => decide (a≤b))).map (fun digit : Nat => (digit : Int))).symm

/-- Equality of sorted digit lists is exactly equality of digit multisets. -/
theorem sortedDigits_eq_iff (a b : Nat) :
    sortedDigits a=sortedDigits b ↔ (Nat.digits 10 a).Perm (Nat.digits 10 b) := by
  constructor
  · intro equal
    have permutation := (ScalarSort.ints_perm (integerDigits a)).symm.trans
      ((by rw [equal] : (sortedDigits a).Perm (sortedDigits b)).trans (ScalarSort.ints_perm (integerDigits b)))
    exact (List.map_perm_map_iff (f := fun (x : Nat) => (x : Int))
      (by intro x y same; exact Int.natCast_inj.mp same)).mp permutation
  · intro permutation
    exact List.Perm.eq_of_pairwise (fun _ _ _ _ lower upper => le_antisymm lower upper)
      (ScalarSort.ints_ordered _) (ScalarSort.ints_ordered _)
      ((ScalarSort.ints_perm _).trans ((permutation.map (fun digit : Nat => (digit : Int))).trans (ScalarSort.ints_perm _).symm))

/-- The source's signed integer comparison matches the statement's natural digit predicate. -/
theorem permutation_statement (a b : Nat) :
    decide (sortedDigits a=sortedDigits b)=ProjectEulerStatements.P49.isPermutation a b := by
  apply Bool.eq_iff_iff.mpr
  simp only [decide_eq_true_eq, ProjectEulerStatements.P49.isPermutation, sortedDigits_statement]
  constructor
  · exact (List.map_inj_right (fun x y same => Int.natCast_inj.mp same)).mp
  · exact congrArg (List.map (fun (digit : Nat) => (digit : Int)))

@[simp] theorem progression_length (a d : Int) (length : Nat) : (progression a d length).length=length := by
  simp [progression]

theorem progression_get (a d : Int) (length index : Nat) (within : index<length) :
    (progression a d length)[index]?=some (a+(index : Int)*d) := by
  simp [progression, within]

/-- The signed source expression agrees with natural arithmetic when both parameters are natural. -/
theorem progression_natural (a d length : Nat) :
    progression a d length=(ProjectEulerStatements.P49.seqFrom a d length).map (fun (n : Nat) => (n : Int)) := by
  simp [progression, ProjectEulerStatements.P49.seqFrom, List.map_map, Function.comp_def]

/-- A nonempty sequence's last element has index length-1. -/
theorem progression_last (a d : Int) (length : Nat) (positive : 0<length) :
    (progression a d length).getLast?=some (a+((length-1 : Nat) : Int)*d) := by
  rw [List.getLast?_eq_getElem?]
  simpa only [progression_length] using progression_get a d length (length-1) (by omega)

theorem progression_first (a d : Int) (length : Nat) (positive : 0<length) :
    (progression a d length)[0]?=some a := by
  simpa using progression_get a d length 0 positive

/-- Increasing positive steps keep every earlier term at or below the last term. -/
theorem progression_upper (a d length index : Nat) (within : index<length) :
    a+index*d≤a+(length-1)*d := by
  have bound : index≤length-1 := by omega
  exact Nat.add_le_add_left (Nat.mul_le_mul_right d bound) a

end ProjectEulerSolutions.PrimePermutations
