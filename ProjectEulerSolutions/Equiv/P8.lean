import ProjectEulerSolutions.P8
import Mathlib.Tactic

namespace ProjectEulerSolutions.P8
open ProjectEulerStatements.P8

lemma windowProducts_eq_nil_of_length_lt (k : Nat) :
    ∀ l : List Nat, l.length < k → windowProducts k l = []
  | [], _ => by simp [windowProducts]
  | _ :: xs, h => by
      rw [windowProducts]
      simp [show xs.length + 1 < k by simpa using h]

lemma listMax_zero_cons (xs : List Nat) : listMax (0 :: xs) = listMax xs := by
  simp [listMax]

lemma listProduct_eq_zero_of_hasZero : ∀ xs : List Nat, hasZero xs = true → listProduct xs = 0
  | [], h => by simp [hasZero] at h
  | x :: xs, h => by
      rw [hasZero] at h
      simp only [Bool.or_eq_true] at h
      rcases h with hx | hxs
      · have hx0 : x = 0 := by simpa using hx
        simp [listProduct, hx0]
      · simp [listProduct, listProduct_eq_zero_of_hasZero xs hxs]

lemma naive_zero (xs : List Nat) : listMax (windowProducts 0 xs) = if xs.isEmpty then 0 else 1 := by
  induction xs with
  | nil => simp [windowProducts, listMax]
  | cons x xs ih =>
      rw [windowProducts]
      simp [listProduct, listMax]
      rw [ih]
      split <;> simp

lemma productLoop_correct_of_pos (k : Nat) (hk : 0 < k) :
    ∀ xs : List Nat, listMax (productLoop k xs) = listMax (windowProducts k xs)
  | [] => by simp [productLoop, windowProducts, listMax]
  | x :: xs => by
      rw [productLoop]
      rw [windowProducts]
      by_cases hlen : xs.length + 1 < k
      · simp [hlen]
      · simp [hlen]
        by_cases hz : hasZero (List.take k (x :: xs)) = true
        · simp [hz, productLoop_correct_of_pos k hk xs]
          rw [listProduct_eq_zero_of_hasZero _ hz]
          simp [listMax]
        · have hzfalse : hasZero (List.take k (x :: xs)) = false := by
            cases hhz : hasZero (List.take k (x :: xs)) <;> simp [hhz] at hz ⊢
          simp [hzfalse, productLoop_correct_of_pos k hk xs, listMax]

theorem equiv (n k : Nat) : ProjectEulerStatements.P8.naive n k = solve n k := by
  unfold ProjectEulerStatements.P8.naive solve maxAdjacentProduct digitsList
  by_cases hk : k = 0
  · subst k
    simp [naive_zero]
  · have hkpos : 0 < k := Nat.pos_of_ne_zero hk
    simp [show (k == 0) = false by simp [hk]]
    exact (productLoop_correct_of_pos k hkpos _).symm

end ProjectEulerSolutions.P8
