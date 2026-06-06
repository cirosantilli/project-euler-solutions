import ProjectEulerSolutions.P8
import Mathlib.Tactic

namespace ProjectEulerSolutions.P8

lemma statements_windowProducts_eq_nil_of_length_lt (k : Nat) :
    ∀ l : List Nat, l.length < k → ProjectEulerStatements.P8.windowProducts k l = []
  | [], _ => by simp [ProjectEulerStatements.P8.windowProducts]
  | _ :: xs, h => by
      rw [ProjectEulerStatements.P8.windowProducts]
      simp [show xs.length + 1 < k by simpa using h]

lemma statements_digits1000_length : ProjectEulerStatements.P8.digits1000.length = 1000 := by
  native_decide

lemma naive_eq_zero_of_gt1000 {k : Nat} (h : 1000 < k) :
    ProjectEulerStatements.P8.naive k = 0 := by
  unfold ProjectEulerStatements.P8.naive
  rw [statements_windowProducts_eq_nil_of_length_lt]
  · simp [ProjectEulerStatements.P8.listMax]
  · rw [statements_digits1000_length]
    omega

lemma findNextZero_le_n (digits : Array Nat) (n i : Nat) (hi : i ≤ n) :
    findNextZero digits n i ≤ n := by
  induction i using findNextZero.induct digits n with
  | case1 x h ih =>
      rw [findNextZero.eq_def]
      simp [h]
      apply ih
      simp at h
      omega
  | case2 x h =>
      rw [findNextZero.eq_def]
      simp [h, hi]

lemma findNextNonZero_le_n (digits : Array Nat) (n i : Nat) (hi : i ≤ n) :
    findNextNonZero digits n i ≤ n := by
  induction i using findNextNonZero.induct digits n with
  | case1 x h ih =>
      rw [findNextNonZero.eq_def]
      simp [h]
      apply ih
      simp at h
      omega
  | case2 x h =>
      rw [findNextNonZero.eq_def]
      simp [h, hi]

lemma loop_eq_best_of_lt_k (digits : Array Nat) (n k i best : Nat) (hk : n < k)
    (hi : i ≤ n) :
    loop digits n k i best = best := by
  induction i, best using loop.induct digits n k with
  | case1 i best hge =>
      rw [loop.eq_def]
      simp only [hge, ↓reduceDIte]
  | case2 i best hge i0 hige =>
      subst i0
      rw [loop.eq_def]
      simp only [hge, hige, ↓reduceDIte, if_true]
  | case3 i best hge i0 hi0_lt j hj_gt =>
      have hi0_le : i0 ≤ n := findNextNonZero_le_n digits n i hi
      have hj_le : j ≤ n := findNextZero_le_n digits n i0 hi0_le
      omega
  | case4 i best hge i0 hi0_lt j hj_ngt hj_le_i =>
      subst i0
      subst j
      have hi0_le : findNextNonZero digits n i ≤ n := findNextNonZero_le_n digits n i hi
      have hj_le : findNextZero digits n (findNextNonZero digits n i) ≤ n :=
        findNextZero_le_n digits n (findNextNonZero digits n i) hi0_le
      have hseg : ¬ findNextZero digits n (findNextNonZero digits n i) -
          findNextNonZero digits n i ≥ k := by omega
      rw [loop.eq_def]
      simp only [hge, hi0_lt, hseg, hj_ngt, hj_le_i, ↓reduceDIte, if_false]
  | case5 i best hge i0 hi0_lt j segLen best0 hj_ngt hj_nle_i ih =>
      subst i0
      subst j
      subst segLen
      subst best0
      have hi0_le : findNextNonZero digits n i ≤ n := findNextNonZero_le_n digits n i hi
      have hj_le : findNextZero digits n (findNextNonZero digits n i) ≤ n :=
        findNextZero_le_n digits n (findNextNonZero digits n i) hi0_le
      have hseg : ¬ findNextZero digits n (findNextNonZero digits n i) -
          findNextNonZero digits n i ≥ k := by omega
      rw [loop.eq_def]
      simp only [hge, hi0_lt, hseg, hj_ngt, hj_nle_i, ↓reduceDIte, if_false]
      simpa [hseg] using ih hj_le

lemma digitsArray_numberStr_size : (digitsArray numberStr).size = 1000 := by
  native_decide

lemma solve_eq_zero_of_gt1000 {k : Nat} (h : 1000 < k) : solve k = 0 := by
  unfold solve maxAdjacentProduct
  change loop (digitsArray numberStr) (digitsArray numberStr).size k 0 0 = 0
  rw [loop_eq_best_of_lt_k]
  · rw [digitsArray_numberStr_size]
    omega
  · exact Nat.zero_le _

theorem equiv (n : Nat) : ProjectEulerStatements.P8.naive n = solve n := by
  by_cases h : n ≤ 1000
  · interval_cases n <;> native_decide
  · have hgt : 1000 < n := by omega
    rw [naive_eq_zero_of_gt1000 hgt, solve_eq_zero_of_gt1000 hgt]

end ProjectEulerSolutions.P8
