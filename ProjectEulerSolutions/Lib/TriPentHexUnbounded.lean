import ProjectEulerSolutions.Lib.TriPentHex
import Mathlib.Data.Nat.Find

/-! Unbounded common polygonal values and their least successor above any signed threshold. -/
namespace ProjectEulerSolutions.TriPentHexUnbounded
open TriPentHex
set_option Elab.async false

def nextPent (pent hex : Nat) : Nat := 97 * pent + 112 * hex - 44
def nextHex (pent hex : Nat) : Nat := 84 * pent + 97 * hex - 38

/-- A Pell recurrence preserves equality of the pentagonal and hexagonal polynomials. -/
theorem next_common (pent hex : Nat) (positivePent : 0 < pent) (positiveHex : 0 < hex)
    (same : pentagonal pent = hexagonal hex) :
    pentagonal (nextPent pent hex) = hexagonal (nextHex pent hex) := by
  cases pent with
  | zero => omega
  | succ pent =>
      cases hex with
      | zero => omega
      | succ hex =>
          have p := PentagonalDiscriminant.twice (pent + 1)
          change 2 * pentagonal (pent + 1) = _ at p
          rw [same] at p
          have next := PentagonalDiscriminant.twice (nextPent (pent + 1) (hex + 1))
          change 2 * pentagonal (nextPent (pent + 1) (hex + 1)) = _ at next
          have pEq : nextPent (pent + 1) (hex + 1) = 97 * pent + 112 * hex + 165 := by
            simp only [nextPent]; omega
          have hEq : nextHex (pent + 1) (hex + 1) = 84 * pent + 97 * hex + 143 := by
            simp only [nextHex]; omega
          have subP : 3 * (pent + 1) - 1 = 3 * pent + 2 := by omega
          have subH : 2 * (hex + 1) - 1 = 2 * hex + 1 := by omega
          have subNextP : 3 * nextPent (pent + 1) (hex + 1) - 1 =
              3 * (97 * pent + 112 * hex + 165) - 1 := by rw [pEq]
          have subNextP' : 3 * (97 * pent + 112 * hex + 165) - 1 = 291 * pent + 336 * hex + 494 := by omega
          have subNextH : 2 * nextHex (pent + 1) (hex + 1) - 1 = 168 * pent + 194 * hex + 285 := by
            rw [hEq]; omega
          simp only [hexagonal, subH, subP] at p
          rw [subNextP, subNextP', pEq] at next
          rw [pEq, hexagonal, subNextH, hEq]
          nlinarith only [p, next]

theorem next_pent_positive (pent hex : Nat) (positivePent : 0 < pent) (positiveHex : 0 < hex) :
    0 < nextPent pent hex := by simp only [nextPent]; omega

theorem next_hex_grows (pent hex : Nat) (positivePent : 0 < pent) (positiveHex : 0 < hex) :
    hex < nextHex pent hex := by simp only [nextHex]; omega

def sequence : Nat → Nat × Nat
  | 0 => (1, 1)
  | n + 1 => let previous := sequence n; (nextPent previous.1 previous.2, nextHex previous.1 previous.2)

theorem sequence_spec (n : Nat) : 0 < (sequence n).1 ∧ n < (sequence n).2 ∧
    pentagonal (sequence n).1 = hexagonal (sequence n).2 := by
  induction n with
  | zero => exact ⟨by decide, by decide, rfl⟩
  | succ n ih =>
      have grows := next_hex_grows (sequence n).1 (sequence n).2 ih.1 (by omega)
      exact ⟨next_pent_positive _ _ ih.1 (by omega), by simpa only [sequence] using (show n + 1 < nextHex (sequence n).1 (sequence n).2 by omega),
        next_common _ _ ih.1 (by omega) ih.2.2⟩

theorem pent_witness (pent hex : Nat) (positivePent : 0 < pent) (same : pentagonal pent = hexagonal hex) :
    hexTest hex = true := by
  unfold hexTest
  rw [PentagonalDiscriminant.correct, PentagonalDiscriminant.exhaustive_iff]
  exact ⟨pent, positivePent, same ▸ PentagonalDiscriminant.index_le pent, same⟩

abbrev Eligible (after : Int) (hex : Nat) : Prop :=
  0 < hex ∧ after < (hexagonal hex : Int) ∧ hexTest hex = true

/-- The recurrence supplies a common value above every integer threshold. -/
theorem exists_eligible (after : Int) : ∃ hex, Eligible after hex := by
  let n := after.toNat + 1
  obtain ⟨positivePent, grows, same⟩ := sequence_spec n
  have positiveHex : 0 < (sequence n).2 := by omega
  have valueBound := hex_index_le (sequence n).2
  exact ⟨(sequence n).2, positiveHex, by omega, pent_witness _ _ positivePent same⟩

noncomputable def winner (after : Int) : Nat := Nat.find (exists_eligible after)
noncomputable def answer (after : Int) : Nat := hexagonal (winner after)

theorem winner_spec (after : Int) : Eligible after (winner after) := Nat.find_spec (exists_eligible after)

theorem winner_minimal (after : Int) (hex : Nat) (eligible : Eligible after hex) : winner after ≤ hex :=
  Nat.find_min' (exists_eligible after) eligible

theorem hex_strictMono : StrictMono hexagonal := by
  intro left right less
  by_cases zero : left = 0
  · subst left
    simpa [hexagonal] using hex_positive right (by omega)
  · rw [← hex_triangular left (by omega), ← hex_triangular right (by omega)]
    exact triangular_strictMono (by omega)

theorem answer_positive (after : Int) : 0 < answer after := hex_positive _ (winner_spec after).1

theorem answer_after (after : Int) : after < (answer after : Int) := (winner_spec after).2.1

theorem answer_le_common (after : Int) (pent hex : Nat) (positivePent : 0 < pent) (positiveHex : 0 < hex)
    (same : pentagonal pent = hexagonal hex) (later : after < (hexagonal hex : Int)) :
    answer after ≤ hexagonal hex :=
  hex_strictMono.monotone (winner_minimal after hex ⟨positiveHex, later, pent_witness _ _ positivePent same⟩)

noncomputable def triangleIndex (after : Int) : Nat := 2 * winner after - 1

private theorem exists_pent_index (after : Int) : ∃ pent : Nat, 0 < pent ∧ pentagonal pent = answer after := by
  have accepted := (winner_spec after).2.2
  unfold hexTest at accepted
  rw [PentagonalDiscriminant.correct, PentagonalDiscriminant.exhaustive_iff] at accepted
  obtain ⟨pent, positive, _, same⟩ := accepted
  exact ⟨pent, positive, same⟩

noncomputable def pentIndex (after : Int) : Nat := Classical.choose (exists_pent_index after)

theorem pent_index_spec (after : Int) : 0 < pentIndex after ∧ pentagonal (pentIndex after) = answer after :=
  Classical.choose_spec (exists_pent_index after)

theorem triangle_index_spec (after : Int) : 0 < triangleIndex after ∧ triangular (triangleIndex after) = answer after := by
  have positive := (winner_spec after).1
  exact ⟨by unfold triangleIndex; omega, hex_triangular _ positive⟩

theorem rejected_before (after : Int) (hex : Nat) (positive : 0 < hex) (before : hex < winner after) :
    ¬ (after < (hexagonal hex : Int) ∧ hexTest hex = true) := by
  intro eligible
  have least := winner_minimal after hex ⟨positive, eligible⟩
  omega

/-- A cursor at the minimum value cannot be at the answer unless all three cursors agree. -/
theorem minimum_before (after : Int) (t p h : Nat)
    (tBound : t ≤ answer after) (pBound : p ≤ answer after) (hBound : h ≤ answer after)
    (rejected : ¬ (t = p ∧ p = h ∧ after < (t : Int)))
    (first : t ≤ p) (second : t ≤ h) : t < answer after := by
  by_contra full
  have sameT : t = answer after := by omega
  have sameP : p = answer after := by omega
  have sameH : h = answer after := by omega
  exact rejected ⟨by omega, by omega, by simpa only [sameT] using answer_after after⟩

end ProjectEulerSolutions.TriPentHexUnbounded
