import ProjectEulerSolutions.P5
import Mathlib.Algebra.GCDMonoid.Finset
import Mathlib.Data.Finset.Interval
import Mathlib.Tactic
namespace ProjectEulerSolutions.P5

lemma lcm_eq_nat_lcm (a b : Nat) : lcm a b = Nat.lcm a b := by
  unfold lcm Nat.lcm
  calc
    a / Nat.gcd a b * b = b * (a / Nat.gcd a b) := by rw [Nat.mul_comm]
    _ = b * a / Nat.gcd a b := by rw [Nat.mul_div_assoc b (Nat.gcd_dvd_left a b)]
    _ = a * b / Nat.gcd a b := by rw [Nat.mul_comm b a]

lemma Ioc_eq_Icc_succ (x n : Nat) : Finset.Ioc x n = Finset.Icc (x + 1) n := by
  ext y
  simp
  omega

lemma go_eq_lcm_Icc (n x acc : Nat) :
    go n x acc = Nat.lcm acc ((Finset.Icc x n).lcm (fun y => y)) := by
  induction h : n + 1 - x using Nat.strong_induction_on generalizing x acc with
  | h m ih =>
      rw [go.eq_1]
      by_cases hx : x > n
      · rw [if_pos hx]
        have hempty : Finset.Icc x n = ∅ := Finset.Icc_eq_empty (by omega)
        simp [hempty]
      · rw [if_neg hx]
        have hxle : x ≤ n := by omega
        have hm : n + 1 - (x + 1) < m := by
          rw [← h]
          exact Nat.sub_lt_sub_left (by omega : x < n + 1) (Nat.lt_succ_self x)
        rw [ih (n + 1 - (x + 1)) hm (x + 1) (lcm acc x) rfl]
        rw [lcm_eq_nat_lcm]
        rw [Finset.Icc_eq_cons_Ioc hxle]
        simp [Finset.lcm_insert, Ioc_eq_Icc_succ, Nat.lcm_assoc]
        rfl

theorem equiv (n : Nat) : ProjectEulerStatements.P5.naive n = solve n := by
  unfold ProjectEulerStatements.P5.naive solve
  rw [go_eq_lcm_Icc]
  by_cases hn : 1 ≤ n
  · rw [Finset.Icc_eq_cons_Ioc hn]
    simp [Finset.lcm_insert, Ioc_eq_Icc_succ]
  · have h1 : Finset.Icc 1 n = ∅ := Finset.Icc_eq_empty hn
    have h2 : Finset.Icc 2 n = ∅ := Finset.Icc_eq_empty (by omega)
    simp [h1, h2]
end ProjectEulerSolutions.P5
