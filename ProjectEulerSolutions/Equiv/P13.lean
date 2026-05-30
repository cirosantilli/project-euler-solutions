import ProjectEulerSolutions.P13
import Batteries.Data.Nat.Digits
import Batteries.Data.String.Lemmas
import Mathlib.Data.Nat.Digits.Defs
import Mathlib.Tactic
namespace ProjectEulerSolutions.P13

lemma digitChar_isDigit (d : Nat) (h : d < 10) : (Nat.digitChar d).isDigit = true := by
  simp [Nat.isDigit_digitChar, h]

lemma digitChar_val (d : Nat) (h : d < 10) : (Nat.digitChar d).toNat - '0'.toNat = d := by
  interval_cases d <;> native_decide

lemma all_digitChars (xs : List Nat) (hxs : ∀ d ∈ xs, d < 10) :
    ((xs.map Nat.digitChar).all (fun c => c.isDigit)) = true := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      have hx : x < 10 := hxs x (by simp)
      have htail : ∀ d ∈ xs, d < 10 := by intro d hd; exact hxs d (by simp [hd])
      simp [digitChar_isDigit x hx, ih htail]

lemma string_mk_cons_isEmpty_false (c : Char) (cs : List Char) :
    (String.mk (c :: cs)).isEmpty = false := by
  have hnot : ¬ (String.mk (c :: cs)).isEmpty := by
    rw [String.isEmpty_iff]
    intro h
    have hd := congrArg String.data h
    simp at hd
  exact Bool.eq_false_iff.mpr hnot

lemma isNat_digitChars_cons (x : Nat) (xs : List Nat) (hx : x < 10)
    (hxs : ∀ d ∈ xs, d < 10) :
    (String.mk ((x :: xs).map Nat.digitChar)).isNat = true := by
  have hall : (String.mk ((x :: xs).map Nat.digitChar)).all (·.isDigit) = true := by
    simpa [String.all_eq] using all_digitChars (x :: xs) (by
      intro d hd
      simp at hd
      rcases hd with rfl | hd
      · exact hx
      · exact hxs d hd)
  rw [String.isNat]
  simp [string_mk_cons_isEmpty_false (Nat.digitChar x) (xs.map Nat.digitChar)]
  change (String.mk ((x :: xs).map Nat.digitChar)).all (fun x => x.isDigit) = true
  exact hall

lemma foldl_digitChars_aux (xs : List Nat) (acc : Nat) (hxs : ∀ d ∈ xs, d < 10) :
    (xs.map Nat.digitChar).foldl (fun n c => n * 10 + (c.toNat - '0'.toNat)) acc =
      acc * 10 ^ xs.length + Nat.ofDigits 10 xs.reverse := by
  induction xs generalizing acc with
  | nil => simp
  | cons x xs ih =>
      have hx : x < 10 := hxs x (by simp)
      have htail : ∀ d ∈ xs, d < 10 := by intro d hd; exact hxs d (by simp [hd])
      rw [List.map_cons, List.foldl_cons,
        ih (acc * 10 + ((Nat.digitChar x).toNat - '0'.toNat)) htail]
      rw [digitChar_val x hx]
      rw [List.reverse_cons, Nat.ofDigits_append]
      simp [Nat.ofDigits]
      ring

lemma foldl_digitChars (xs : List Nat) (hxs : ∀ d ∈ xs, d < 10) :
    (xs.map Nat.digitChar).foldl (fun n c => n * 10 + (c.toNat - '0'.toNat)) 0 =
      Nat.ofDigits 10 xs.reverse := by
  simpa using foldl_digitChars_aux xs 0 hxs

lemma toNat!_digitChars (xs : List Nat) (hxs : ∀ d ∈ xs, d < 10) :
    (String.mk (xs.map Nat.digitChar)).toNat! = Nat.ofDigits 10 xs.reverse := by
  cases xs with
  | nil => rfl
  | cons x xs =>
      have hx : x < 10 := hxs x (by simp)
      have htail : ∀ d ∈ xs, d < 10 := by intro d hd; exact hxs d (by simp [hd])
      unfold String.toNat!
      rw [isNat_digitChars_cons x xs hx htail]
      rw [String.foldl_eq]
      change ((x :: xs).map Nat.digitChar).foldl
        (fun n c => n * 10 + (c.toNat - '0'.toNat)) 0 = Nat.ofDigits 10 (x :: xs).reverse
      exact foldl_digitChars (x :: xs) hxs

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

lemma firstDigits_nat (n k : Nat) :
    (String.mk ((toString n).data.take k)).toNat! =
      Nat.ofDigits 10 (((Nat.digits 10 n).reverse.take k).reverse) := by
  by_cases hn : n = 0
  · subst n
    rw [show (toString 0).data = Nat.toDigits 10 0 by rfl, Nat.toDigits_zero]
    cases k with
    | zero => rfl
    | succ k =>
        rw [List.take_succ_cons]
        simp
        native_decide
  · have hnpos : 0 < n := Nat.pos_of_ne_zero hn
    have hdigits : ∀ d ∈ (Nat.digits 10 n).reverse.take k, d < 10 := by
      intro d hd
      have hdrev : d ∈ (Nat.digits 10 n).reverse := List.mem_of_mem_take hd
      simpa using (Nat.digits_lt_base (b := 10) (m := n) (d := d) (by norm_num)
        (by simpa using hdrev))
    rw [show (toString n).data = Nat.toDigits 10 n by rfl]
    rw [toDigits_eq_digitChars n hnpos]
    rw [← List.map_take]
    exact toNat!_digitChars ((Nat.digits 10 n).reverse.take k) hdigits

lemma foldl_add_eq_sum_aux (nums : List Nat) (acc : Nat) :
    nums.foldl (fun acc n => acc + n) acc = acc + nums.sum := by
  induction nums generalizing acc with
  | nil => simp
  | cons x xs ih =>
      rw [List.foldl_cons, ih (acc + x)]
      simp [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

lemma foldl_add_eq_sum (nums : List Nat) : nums.foldl (fun acc n => acc + n) 0 = nums.sum := by
  simpa using foldl_add_eq_sum_aux nums 0

theorem equiv (nums : List Nat) (k : Nat) :
    ProjectEulerStatements.P13.naive nums k = solve nums k := by
  unfold ProjectEulerStatements.P13.naive solve firstDigitsOfSum
  rw [foldl_add_eq_sum nums]
  exact (firstDigits_nat nums.sum k).symm
end ProjectEulerSolutions.P13
