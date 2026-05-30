import ProjectEulerSolutions.P1
import Mathlib.Tactic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Algebra.Ring.Parity
namespace ProjectEulerSolutions.P1

lemma triangular_step (q : Nat) :
    ((q + 1) * (q + 2) / 2) = (q * (q + 1) / 2) + (q + 1) := by
  have h := Nat.choose_succ_succ' (q + 1) 1
  simp [Nat.choose_two_right, Nat.choose_one_right] at h
  simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm,
    Nat.mul_assoc] using h

lemma scaled_triangular_step (m q : Nat) :
    m * (q + 1) * (q + 2) / 2 = m * q * (q + 1) / 2 + m * (q + 1) := by
  rw [show m * (q + 1) * (q + 2) = m * ((q + 1) * (q + 2)) by ring]
  rw [Nat.mul_div_assoc m (Nat.two_dvd_mul_add_one (q + 1))]
  rw [show m * q * (q + 1) = m * (q * (q + 1)) by ring]
  rw [Nat.mul_div_assoc m (Nat.two_dvd_mul_add_one q)]
  rw [triangular_step]
  ring

lemma div_mul_succ_sub_one (m q : Nat) (hm : 0 < m) :
    (m * (q + 1) - 1) / m = q := by
  apply Nat.div_eq_of_lt_le
  · apply Nat.le_sub_one_of_lt
    nlinarith [show m * (q + 1) = q * m + m by ring]
  · have hpos : 0 < m * (q + 1) := Nat.mul_pos hm (Nat.succ_pos q)
    have hlt : m * (q + 1) - 1 < m * (q + 1) := Nat.sub_one_lt_of_lt hpos
    simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using hlt

lemma div_mul_add_sub_one (m q r : Nat) (hrpos : 0 < r) (hrlt : r < m) :
    (m * q + r - 1) / m = q := by
  apply Nat.div_eq_of_lt_le
  · apply Nat.le_sub_one_of_lt
    nlinarith [show m * q = q * m by ring]
  · have hle : m * q + r - 1 ≤ m * q + r := Nat.sub_le _ _
    have hlt : m * q + r < (q + 1) * m := by
      nlinarith [show (q + 1) * m = m * q + m by ring]
    exact Nat.lt_of_le_of_lt hle hlt

lemma div_mul_add (m q r : Nat) (hrlt : r < m) :
    (m * q + r) / m = q := by
  apply Nat.div_eq_of_lt_le
  · nlinarith [show m * q = q * m by ring]
  · nlinarith [show (q + 1) * m = m * q + m by ring]

lemma mod_mul_add (m q r : Nat) (hrlt : r < m) : (m * q + r) % m = r := by
  simp [Nat.add_mod, Nat.mul_mod_right, Nat.mod_eq_of_lt hrlt]

lemma sumOfMultiples_step (m n : Nat) (hm : 0 < m) :
    sumOfMultiples m n = sumOfMultiples m (n - 1) + (if m ∣ n then n else 0) := by
  let q := n / m
  let r := n % m
  have hn : n = m * q + r := by
    simpa [q, r, Nat.mul_comm, Nat.add_comm] using (Nat.div_add_mod n m).symm
  have hrlt : r < m := by simpa [r] using Nat.mod_lt n hm
  by_cases hr0 : r = 0
  · have hn0 : n = m * q := by omega
    rw [hn0]
    cases q with
    | zero => simp [sumOfMultiples]
    | succ q =>
        have hdiv : (m * (q + 1)) / m = q + 1 := Nat.mul_div_right (q + 1) hm
        have hdiv_prev : (m * (q + 1) - 1) / m = q := div_mul_succ_sub_one m q hm
        have hdvd : m ∣ m * (q + 1) := dvd_mul_right m (q + 1)
        unfold sumOfMultiples
        rw [hdiv, hdiv_prev]
        simp [hdvd, scaled_triangular_step]
  · have hrpos : 0 < r := Nat.pos_of_ne_zero hr0
    rw [hn]
    have hdiv : (m * q + r) / m = q := div_mul_add m q r hrlt
    have hdiv_prev : (m * q + r - 1) / m = q := div_mul_add_sub_one m q r hrpos hrlt
    have hnotdvd : ¬ m ∣ m * q + r := by
      intro hd
      have hmod : (m * q + r) % m = r := mod_mul_add m q r hrlt
      have hzero : (m * q + r) % m = 0 := Nat.mod_eq_zero_of_dvd hd
      omega
    unfold sumOfMultiples
    rw [hdiv, hdiv_prev]
    simp [hnotdvd]

lemma sumOfMultiples_le_of_dvd {a b : Nat} (ha : 0 < a) (hb : 0 < b) (hab : a ∣ b)
    (n : Nat) :
    sumOfMultiples b n ≤ sumOfMultiples a n := by
  induction n with
  | zero => simp [sumOfMultiples]
  | succ n ih =>
      rw [sumOfMultiples_step b (n + 1) hb, sumOfMultiples_step a (n + 1) ha]
      by_cases hbdiv : b ∣ n + 1
      · have hadiv : a ∣ n + 1 := dvd_trans hab hbdiv
        simp [hbdiv, hadiv]
        exact ih
      · by_cases hadiv : a ∣ n + 1
        · simp [hbdiv, hadiv]
          exact le_trans ih (Nat.le_add_right _ _)
        · simp [hbdiv, hadiv, ih]

lemma dvd_fifteen_iff (n : Nat) : 15 ∣ n ↔ 3 ∣ n ∧ 5 ∣ n := by
  constructor
  · intro h
    exact ⟨dvd_trans (by norm_num : 3 ∣ 15) h, dvd_trans (by norm_num : 5 ∣ 15) h⟩
  · intro h
    simpa [Nat.mul_comm] using
      (Nat.Coprime.mul_dvd_of_dvd_of_dvd (by norm_num : Nat.Coprime 3 5) h.1 h.2)

lemma solve_step (n : Nat) :
    solve (n + 1) = solve n + (if 3 ∣ n ∨ 5 ∣ n then n else 0) := by
  unfold solve
  simp only [Nat.add_sub_cancel_right]
  rw [sumOfMultiples_step 3 n (by norm_num), sumOfMultiples_step 5 n (by norm_num),
    sumOfMultiples_step 15 n (by norm_num)]
  have hC3 : sumOfMultiples 15 (n - 1) ≤ sumOfMultiples 3 (n - 1) :=
    sumOfMultiples_le_of_dvd (by norm_num) (by norm_num) (by norm_num : 3 ∣ 15) (n - 1)
  have hC : sumOfMultiples 15 (n - 1) ≤ sumOfMultiples 3 (n - 1) + sumOfMultiples 5 (n - 1) :=
    le_trans hC3 (Nat.le_add_right _ _)
  by_cases h3 : 3 ∣ n <;> by_cases h5 : 5 ∣ n <;>
    simp [h3, h5, dvd_fifteen_iff, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
  all_goals omega

theorem equiv (n : Nat) : ProjectEulerStatements.P1.naive n = solve n := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [ProjectEulerStatements.P1.naive, ih, solve_step]
      by_cases h : 3 ∣ n ∨ 5 ∣ n <;> simp [h, Nat.add_comm]
end ProjectEulerSolutions.P1
