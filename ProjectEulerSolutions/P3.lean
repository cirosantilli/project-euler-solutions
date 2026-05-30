import ProjectEulerStatements.P3
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Nat.Find
import Mathlib.Data.Prod.Basic
import Mathlib.Tactic

namespace ProjectEulerSolutions.P3

open ProjectEulerStatements.P3

def lpf (n : Nat) : Nat :=
  Nat.findGreatest (isPrimeFactor n) n

def stripTwos (n : Nat) (largest : Nat) : Nat × Nat :=
  if h0 : n = 0 then
    (0, largest)
  else if n % 2 == 0 then
    stripTwos (n / 2) 2
  else
    (n, largest)
termination_by n
decreasing_by
  have hnpos : 0 < n := Nat.pos_of_ne_zero h0
  have hlt : n / 2 < n := Nat.div_lt_self hnpos (by decide : 1 < 2)
  simpa using hlt

def oddLoop (n f largest : Nat) : Nat :=
  if hf : f ≤ 1 then
    if n > 1 then n else largest
  else if f * f <= n then
    if n % f == 0 then
      oddLoop (n / f) f f
    else
      oddLoop n (f + 2) largest
  else
    if n > 1 then n else largest
termination_by (n, n - f)
decreasing_by
  · have hf1 : 1 < f := lt_of_not_ge hf
    have hfpos : 0 < f := lt_trans (by decide : 0 < 1) hf1
    have hffpos : 0 < f * f := Nat.mul_pos hfpos hfpos
    have hnpos : 0 < n := lt_of_lt_of_le hffpos (by
      have : f * f ≤ n := by
        simpa using ‹f * f <= n›
      exact this)
    have hlt : n / f < n := Nat.div_lt_self hnpos hf1
    exact (Prod.lex_iff).2 (Or.inl hlt)
  · have hf1 : 1 < f := lt_of_not_ge hf
    have hf2 : 2 ≤ f := Nat.succ_le_iff.mp hf1
    have hff : f * f ≤ n := by
      simpa using ‹f * f <= n›
    have hf2le : f + 2 ≤ f * f := by
      calc
        f + 2 ≤ f + f := by
          exact Nat.add_le_add_left hf2 f
        _ ≤ f * f := by
          have hff : 2 * f ≤ f * f := Nat.mul_le_mul_right f hf2
          have hff_eq : 2 * f = f + f := by
            simp [Nat.two_mul]
          simpa [hff_eq] using hff
    have hf2le' : f + 2 ≤ n := le_trans hf2le hff
    have hlt : n - (f + 2) < n - f := by
      omega
    exact (Prod.lex_iff).2 (Or.inr ⟨rfl, hlt⟩)

def solve (n : ProjectEulerStatements.P3.NatGE2) : Nat :=
  let (n', largest') := stripTwos n.1 1
  oddLoop n' 3 largest'

example : solve ⟨13195, by decide⟩ = 29 := by
  native_decide

theorem equiv (n : ProjectEulerStatements.P3.NatGE2) :
    ProjectEulerStatements.P3.naive n = solve n := sorry

end ProjectEulerSolutions.P3
