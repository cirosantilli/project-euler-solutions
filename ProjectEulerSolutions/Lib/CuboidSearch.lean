import ProjectEulerSolutions.Lib.CuboidRoute
import Mathlib.Tactic.Ring

/-! Unbounded cuboid counts and the least side bound exceeding a signed threshold. -/
namespace ProjectEulerSolutions.CuboidRoute

set_option Elab.async false

def total : Nat → Nat
  | 0 => 0
  | m + 1 => total m + exhaustiveLayer (m + 1)

def groupedTotal : Nat → Nat
  | 0 => 0
  | m + 1 => groupedTotal m + groupedLayer (m + 1)

theorem total_equivalence (m : Nat) : total m = groupedTotal m := by
  induction m with
  | zero => rfl
  | succ m ih => simp only [total, groupedTotal, ih, layer_equivalence]

theorem total_monotone : Monotone total :=
  monotone_nat_of_le_succ (fun m => by rw [total]; omega)

/-- Each positive scale of the 3-4-5 triangle supplies another valid cuboid. -/
theorem scaled_cuboid (k : Nat) (positive : 0 < k) : (2 * k, 2 * k) ∈ cuboids (3 * k) := by
  rw [mem_cuboids]
  refine ⟨by omega, le_rfl, by omega, ?_⟩
  apply (square_iff _).mpr
  exact ⟨5 * k, by ring⟩

theorem scaled_layer_positive (k : Nat) (positive : 0 < k) : 1 ≤ exhaustiveLayer (3 * k) :=
  Finset.card_pos.mpr ⟨(2 * k, 2 * k), scaled_cuboid k positive⟩

theorem total_three_mul (k : Nat) : k ≤ total (3 * k) := by
  induction k with
  | zero => simp [total]
  | succ k ih =>
      have earlier := total_monotone (show 3 * k ≤ 3 * k + 2 by omega)
      have positive : 1 ≤ exhaustiveLayer (3 * k + 2 + 1) := by
        simpa [Nat.mul_add, Nat.add_assoc] using scaled_layer_positive (k + 1) (by omega)
      have bound : k + 1 ≤ total (3 * k + 2 + 1) := by rw [total]; omega
      simpa [Nat.mul_add, Nat.add_assoc] using bound

/-- This bound is a termination witness, never a cutoff in the executable search. -/
def searchBound (target : Int) : Nat := 3 * (target.toNat + 1)

theorem bound_exceeds (target : Int) : target < (total (searchBound target) : Int) := by
  have enough := total_three_mul (target.toNat + 1)
  unfold searchBound
  omega

theorem active_below_bound (target : Int) (m : Nat) (active : (total m : Int) ≤ target) :
    m < searchBound target := by
  have enough := bound_exceeds target
  by_contra h
  have counts := total_monotone (show searchBound target ≤ m by omega)
  omega

def scan (target : Int) (m : Nat) : Nat :=
  if active : (total m : Int) ≤ target then scan target (m + 1) else m
termination_by searchBound target - m
decreasing_by
  have := active_below_bound target m (by assumption)
  omega

theorem scan_spec (target : Int) (m : Nat) :
    m ≤ scan target m ∧ target < (total (scan target m) : Int) ∧
      ∀ earlier, m ≤ earlier → earlier < scan target m → (total earlier : Int) ≤ target := by
  rw [scan]
  split
  · rename_i active
    obtain ⟨later, enough, minimal⟩ := scan_spec target (m + 1)
    refine ⟨by omega, enough, ?_⟩
    intro earlier lower upper
    by_cases same : earlier = m
    · simpa only [same] using active
    · exact minimal earlier (by omega) upper
  · rename_i finished
    exact ⟨le_rfl, by omega, by intro earlier lower upper; omega⟩
termination_by searchBound target - m
decreasing_by
  have := active_below_bound target m (by assumption)
  omega

def answer (target : Int) : Nat := scan target 0

theorem answer_spec (target : Int) : target < (total (answer target) : Int) ∧
    ∀ earlier < answer target, (total earlier : Int) ≤ target := by
  have checked := scan_spec target 0
  exact ⟨checked.2.1, fun earlier less => checked.2.2 earlier (by omega) less⟩

theorem answer_negative (target : Int) (negative : target < 0) : answer target = 0 := by
  rw [answer, scan, dif_neg (by simpa [total] using not_le.mpr negative)]

end ProjectEulerSolutions.CuboidRoute
