import ProjectEulerSolutions.P16
import Batteries.Data.Nat.Digits
import Batteries.Data.String.Lemmas
import Mathlib.Data.Nat.Digits.Defs
import Mathlib.Tactic
namespace ProjectEulerSolutions.P16

lemma digitChar_val (d : Nat) (h : d < 10) : (Nat.digitChar d).toNat - '0'.toNat = d := by
  interval_cases d <;> native_decide

lemma foldl_digitChars_sum_aux (xs : List Nat) (acc : Nat) (hxs : ∀ d ∈ xs, d < 10) :
    (xs.map Nat.digitChar).foldl (fun acc c => acc + (c.toNat - '0'.toNat)) acc =
      acc + xs.sum := by
  induction xs generalizing acc with
  | nil => simp
  | cons x xs ih =>
      have hx : x < 10 := hxs x (by simp)
      have htail : ∀ d ∈ xs, d < 10 := by intro d hd; exact hxs d (by simp [hd])
      rw [List.map_cons, List.foldl_cons,
        ih (acc + ((Nat.digitChar x).toNat - '0'.toNat)) htail]
      rw [digitChar_val x hx]
      simp [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

lemma foldl_digitChars_sum (xs : List Nat) (hxs : ∀ d ∈ xs, d < 10) :
    (xs.map Nat.digitChar).foldl (fun acc c => acc + (c.toNat - '0'.toNat)) 0 =
      xs.sum := by
  simpa using foldl_digitChars_sum_aux xs 0 hxs

lemma toDigits_eq_digitChars (n : Nat) (hn : 0 < n) :
    Nat.toDigits 10 n = (Nat.digits 10 n).reverse.map Nat.digitChar := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      by_cases hlt : n < 10
      · rw [Nat.toDigits_of_lt_base hlt, Nat.digits_of_lt 10 n (Nat.ne_of_gt hn) hlt]
        simp
      · have hn10 : 10 ≤ n := le_of_not_gt hlt
        have hdivpos : 0 < n / 10 := Nat.div_pos hn10 (by norm_num)
        have hdivlt : n / 10 < n := Nat.div_lt_self hn (by norm_num)
        have hmodlt : n % 10 < 10 := Nat.mod_lt n (by norm_num)
        have hn_eq : n = 10 * (n / 10) + n % 10 := by
          conv_lhs => rw [← Nat.div_add_mod n 10]
        calc
          Nat.toDigits 10 n = Nat.toDigits 10 (10 * (n / 10) + n % 10) := by
            exact congrArg (Nat.toDigits 10) hn_eq
          _ = Nat.toDigits 10 (n / 10) ++ Nat.toDigits 10 (n % 10) := by
            exact (Nat.toDigits_append_toDigits (b := 10) (n := n / 10) (d := n % 10)
              (by norm_num) hdivpos hmodlt).symm
          _ = (Nat.digits 10 (n / 10)).reverse.map Nat.digitChar ++
                [Nat.digitChar (n % 10)] := by
            rw [ih (n / 10) hdivlt hdivpos, Nat.toDigits_of_lt_base hmodlt]
          _ = (Nat.digits 10 n).reverse.map Nat.digitChar := by
            rw [Nat.digits_of_two_le_of_pos (b := 10) (n := n) (by norm_num) hn]
            simp [List.map_append]

lemma digitSum_eq_digits_sum (n : Nat) : digitSum n = (Nat.digits 10 n).sum := by
  unfold digitSum
  by_cases hn : n = 0
  · subst n
    change (toString 0).data.foldl (fun acc c => acc + (c.toNat - '0'.toNat)) 0 =
      (Nat.digits 10 0).sum
    rw [show (toString 0).data = Nat.toDigits 10 0 by rfl, Nat.toDigits_zero]
    native_decide
  · have hnpos : 0 < n := Nat.pos_of_ne_zero hn
    have hdigits : ∀ d ∈ (Nat.digits 10 n).reverse, d < 10 := by
      intro d hd
      simpa using (Nat.digits_lt_base (b := 10) (m := n) (d := d) (by norm_num)
        (by simpa using hd))
    change (toString n).data.foldl (fun acc c => acc + (c.toNat - '0'.toNat)) 0 =
      (Nat.digits 10 n).sum
    rw [show (toString n).data = Nat.toDigits 10 n by rfl]
    rw [toDigits_eq_digitChars n hnpos]
    rw [foldl_digitChars_sum (Nat.digits 10 n).reverse hdigits]
    exact List.sum_reverse (Nat.digits 10 n)

theorem equiv (n : Nat) : ProjectEulerStatements.P16.naive n = solve n := by
  unfold ProjectEulerStatements.P16.naive ProjectEulerStatements.P16.digitSum solve
  exact (digitSum_eq_digits_sum (2 ^ n)).symm
end ProjectEulerSolutions.P16
