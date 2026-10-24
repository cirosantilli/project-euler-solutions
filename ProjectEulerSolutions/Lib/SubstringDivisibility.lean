import ProjectEulerSolutions.Lib.ConstraintRows
import Mathlib.Data.Nat.Digits.Lemmas

/-! Complete right-to-left enumeration for the substring divisibility constraints. -/
namespace ProjectEulerSolutions.SubstringDivisibility
open FormalSW.PyAst
open ConstraintRows

set_option Elab.async false

def pool : List Nat := List.range 10
def primes : List Nat := [2, 3, 5, 7, 11, 13, 17]
def triple (a b c : Nat) : Nat := 100*a + 10*b + c
def window (p a : Nat) (suffix : List Nat) : Bool :=
  decide (triple a (suffix.headD 0) ((suffix.drop 1).headD 0) % p = 0)
def unconstrained : Nat → List Nat → Bool := fun _ _ => true
def tests : List (Nat → List Nat → Bool) :=
  unconstrained :: primes.map window ++ [unconstrained, unconstrained]
def rows : List (List Nat) := search pool tests
def referenceRows : List (List Nat) := (Permutations.rows 10 pool).filter (accepts tests)
def value (digits : List Nat) : Nat := Nat.ofDigits 10 digits.reverse
def nonzeroFirst (digits : List Nat) : Bool := decide (digits.headD 0 ≠ 0)

theorem pool_nodup : pool.Nodup := List.nodup_range

theorem rows_nodup : rows.Nodup := search_nodup pool tests pool_nodup

/-- The tests are precisely the seven overlapping three-digit divisibility conditions. -/
theorem accepts_digits (a b c d e f g h i j : Nat) :
    accepts tests [a,b,c,d,e,f,g,h,i,j] =
      decide (triple b c d % 2 = 0 ∧ triple c d e % 3 = 0 ∧
        triple d e f % 5 = 0 ∧ triple e f g % 7 = 0 ∧
        triple f g h % 11 = 0 ∧ triple g h i % 13 = 0 ∧ triple h i j % 17 = 0) := by
  simp [tests, primes, accepts, unconstrained, window]

theorem row_length (row : List Nat) (member : row ∈ rows) : row.length = 10 := by
  have accepted := (mem_search pool tests row).mp member
  simpa [tests, primes] using accepts_length tests row accepted.2.2

theorem row_perm (row : List Nat) (member : row ∈ rows) : row.Perm pool := by
  obtain ⟨nodup, subset, _⟩ := (mem_search pool tests row).mp member
  exact (nodup.subperm subset).perm_of_length_le (by simp [pool, row_length row member])

/-- Exhaustive permutations and backtracking have the same rows, with the same multiplicity. -/
theorem rows_perm_reference : rows.Perm referenceRows := by
  exact search_perm_filter pool tests pool_nodup (by decide)

/-- The source's exclusion of a zero first digit cannot remove a valid result. -/
theorem no_leading_zero : rows.all nonzeroFirst = true := by
  decide +kernel

theorem row_first_nonzero (row : List Nat) (member : row ∈ rows) : row.headD 0 ≠ 0 := by
  exact of_decide_eq_true (List.all_eq_true.mp no_leading_zero row member)

theorem reference_first_nonzero (row : List Nat) (member : row ∈ referenceRows) : row.headD 0 ≠ 0 :=
  row_first_nonzero row (rows_perm_reference.mem_iff.mpr member)

theorem filter_leading_zero : rows.filter nonzeroFirst = rows := by
  apply List.filter_eq_self.mpr
  intro row member
  exact List.all_eq_true.mp no_leading_zero row member

theorem sum_equiv : ((rows.filter nonzeroFirst).map value).sum = (referenceRows.map value).sum := by
  rw [filter_leading_zero]
  exact (rows_perm_reference.map value).sum_eq

/-- Leading zeros in a three-digit window do not affect decimal decoding. -/
theorem triple_decode (a b c : Nat) (ha : a < 10) (hb : b < 10) (hc : c < 10) :
    triple a b c < 1000 ∧ triple a b c / 100 = a ∧
      (triple a b c / 10) % 10 = b ∧ triple a b c % 10 = c := by
  dsimp [triple]
  omega

theorem triple_encode (x : Nat) :
    triple (x / 100) ((x / 10) % 10) (x % 10) = x := by
  dsimp [triple]
  omega

theorem triple_injective (a b c d e f : Nat)
    (ha : a < 10) (hb : b < 10) (hc : c < 10)
    (hd : d < 10) (he : e < 10) (hf : f < 10)
    (equal : triple a b c = triple d e f) : a = d ∧ b = e ∧ c = f := by
  have left := triple_decode a b c ha hb hc
  have right := triple_decode d e f hd he hf
  rw [equal] at left
  exact ⟨left.2.1.symm.trans right.2.1,
    left.2.2.1.symm.trans right.2.2.1, left.2.2.2.symm.trans right.2.2.2⟩

end ProjectEulerSolutions.SubstringDivisibility
