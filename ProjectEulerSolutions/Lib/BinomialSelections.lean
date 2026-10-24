import ProjectEulerStatements.Lib.Binomial
import lang.py.FormalSW.Math
import Mathlib.Order.Interval.Finset.Nat
import Mathlib.Data.Finset.Card
import Mathlib.Tactic.Linarith

/-! Exact binomial updates and the symmetric interval counted by the first threshold crossing. -/
namespace ProjectEulerSolutions.BinomialSelections
open FormalSW.PyAst

set_option Elab.async false

/-- The interpreter's standard-library model computes the mathematical binomial coefficient. -/
theorem comb_eq_choose (n k : Nat) : Math.comb n k = Nat.choose n k := by
  unfold Math.comb
  split_ifs with large
  · exact (Nat.choose_eq_zero_of_lt large).symm
  · have bound : min k (n - k) ≤ n := by omega
    rw [ProjectEulerStatements.Binomial.recurrence_eq
      (n - min k (n - k)) (Math.combProduct (n - min k (n - k))) rfl (fun _ => rfl),
      Nat.sub_add_cancel bound, ProjectEulerStatements.Binomial.choose_min n k (by omega)]

/-- Every division performed by the optimized row loop is exact. -/
theorem update (n r : Nat) (positive : 0 < r) (bounded : r ≤ n) :
    Nat.choose n (r - 1) * (n - r + 1) / r = Nat.choose n r := by
  have step := Nat.choose_succ_right_eq n (r - 1)
  have previous : r - 1 + 1 = r := by omega
  have difference : n - (r - 1) = n - r + 1 := by omega
  rw [previous, difference] at step
  rw [← step, Nat.mul_div_cancel _ positive]

/-- Binomial coefficients increase up to the middle of a row. -/
theorem monotone_left (n a b : Nat) (lower : a ≤ b) (upper : b ≤ n / 2) : Nat.choose n a ≤ Nat.choose n b := by
  induction b, lower using Nat.le_induction with
  | base => exact Nat.le_refl _
  | succ b before ih =>
      exact (ih (by omega)).trans (Nat.choose_le_succ_of_lt_half_left (by omega))

/-- Symmetry makes every coefficient between matching boundary positions at least the boundary value. -/
theorem between (n first index : Nat) (middle : first ≤ n / 2) (lower : first ≤ index) (upper : index ≤ n - first) :
    Nat.choose n first ≤ Nat.choose n index := by
  by_cases left : index ≤ n / 2
  · exact monotone_left n first index lower left
  · rw [← Nat.choose_symm (by omega : index ≤ n)]
    exact monotone_left n first (n - index) (by omega) (by omega)

def row (n : Nat) (threshold : Int) : Nat :=
  ((List.range (n + 1)).filter (fun r => decide (threshold < (Nat.choose n r : Int)))).length

/-- Before the first hit, the entire accepted row is exactly its symmetric central interval. -/
theorem first_hit (n first : Nat) (threshold : Int) (middle : first ≤ n / 2)
    (earlier : ∀ r < first, (Nat.choose n r : Int) ≤ threshold)
    (accepted : threshold < (Nat.choose n first : Int)) : row n threshold = n - 2 * first + 1 := by
  let items := (List.range (n + 1)).filter (fun r => decide (threshold < (Nat.choose n r : Int)))
  have equal : items.toFinset = Finset.Icc first (n - first) := by
    ext index
    simp only [items, List.mem_toFinset, List.mem_filter, List.mem_range, decide_eq_true_eq, Finset.mem_Icc]
    constructor
    · rintro ⟨bounded, good⟩
      constructor
      · by_contra small
        have bad := earlier index (by omega)
        omega
      · by_contra large
        have bad := earlier (n - index) (by omega)
        rw [Nat.choose_symm (by omega)] at bad
        omega
    · rintro ⟨lower, upper⟩
      have increase : (Nat.choose n first : Int) ≤ Nat.choose n index := by
        exact Int.ofNat_le.mpr (between n first index middle lower upper)
      exact ⟨by omega, by omega⟩
  have size := List.toFinset_card_of_nodup ((List.nodup_range (n := n + 1)).filter (fun r => decide (threshold < (Nat.choose n r : Int))))
  change items.toFinset.card = row n threshold at size
  rw [equal, Nat.card_Icc] at size
  omega

/-- If the middle was rejected, symmetry excludes all remaining coefficients. -/
theorem exhausted (n first : Nat) (threshold : Int) (past : n / 2 < first)
    (earlier : ∀ r < first, (Nat.choose n r : Int) ≤ threshold) : row n threshold = 0 := by
  have middle := earlier (n / 2) past
  have empty : (List.range (n + 1)).filter (fun r => decide (threshold < (Nat.choose n r : Int))) = [] := by
    apply List.filter_eq_nil_iff.mpr
    intro r _
    have maximum : (Nat.choose n r : Int) ≤ Nat.choose n (n / 2) := Int.ofNat_le.mpr (Nat.choose_le_middle r n)
    simp only [decide_eq_true_eq]
    omega
  simp [row, empty]

/-- For nonpositive thresholds every coefficient, including both endpoints, contributes. -/
theorem row_low (n : Nat) (threshold : Int) (low : threshold < 1) : row n threshold = n + 1 := by
  have selected : (List.range (n + 1)).filter (fun r => decide (threshold < (Nat.choose n r : Int))) = List.range (n + 1) := by
    apply List.filter_eq_self.mpr
    intro r member
    have positive : 0 < Nat.choose n r := Nat.choose_pos (by have bound := List.mem_range.mp member; omega)
    simp only [decide_eq_true_eq]
    omega
  simp only [row, selected, List.length_range]

def series (first count : Nat) (threshold : Int) : Nat :=
  ((List.range' first count).map (fun n => row n threshold)).sum

def answer (limit threshold : Int) : Nat := series 1 limit.toNat threshold

@[simp] theorem series_zero (first : Nat) (threshold : Int) : series first 0 threshold = 0 := rfl

theorem series_succ (first count : Nat) (threshold : Int) :
    series first (count + 1) threshold = row first threshold + series (first + 1) count threshold := by
  simp only [series, List.range'_succ, List.map_cons, List.sum_cons]

/-- Counting every entry of the positive rows gives the exact low-threshold shortcut. -/
theorem answer_low (limit threshold : Int) (low : threshold < 1) :
    answer limit threshold = limit.toNat * (limit.toNat + 3) / 2 := by
  have doubled : ∀ count, 2 * series 1 count threshold = count * (count + 3) := by
    intro count
    induction count with
    | zero => simp
    | succ count ih =>
        have extended : series 1 (count + 1) threshold = series 1 count threshold + row (count + 1) threshold := by
          simp [series, List.range'_concat, List.map_append, List.sum_append, Nat.add_comm]
        rw [extended, row_low _ _ low]
        nlinarith
  unfold answer
  rw [← doubled limit.toNat]
  omega

end ProjectEulerSolutions.BinomialSelections
