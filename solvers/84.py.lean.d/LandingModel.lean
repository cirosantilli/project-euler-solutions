import solvers.«84.py.lean.d».LandingChance
import solvers.«84.py.lean.d».LandingCommunityResult
import solvers.«84.py.lean.d».LandingSingletonResult
import ProjectEulerSolutions.Lib.MonopolyStationary

namespace solvers.«84».resolve_landing
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

def modelData (square : Nat) : List weights.CardData :=
  (Monopoly.landingWeights square).map fun entry => (((entry.1.1 : Int), entry.1.2), (entry.2 : Int))

private theorem mem_int (square : Nat) (values : List Nat) :
    (square : Int) ∈ values.map Int.ofNat ↔ square ∈ values := by
  constructor
  · intro member
    obtain ⟨value, allowed, same⟩ := List.mem_map.mp member
    have equal : value = square := Int.ofNat.inj same
    simpa only [equal] using allowed
  · intro member
    exact List.mem_map.mpr ⟨square, member, rfl⟩

theorem model_bound (square : Nat) (bound : square < 40) :
    ∀ item ∈ modelData square, 0 ≤ item.1.1 ∧ item.1.1 < 40 := by
  intro item member
  obtain ⟨entry, allowed, rfl⟩ := List.mem_map.mp member
  have inside := Monopoly.landingWeights_bound square bound entry allowed
  exact ⟨Int.natCast_nonneg _, by dsimp; omega⟩

/-- The actual landing source returns a fresh dictionary with the exact model weights and distinct positive entries. -/
theorem result (square : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.ReturnsRel program pool.spec "resolve_landing" [.scalar (.int square)] heap
      (weights.Post heap (modelData square)) globals := by
  by_cases jail : square = 30
  · subst square
    simpa [modelData, Monopoly.landingWeights] using jail_weights_result heap globals
  · by_cases cc : square ∈ [2, 17, 33]
    · have data : community.data square = modelData square := by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at cc
        rcases cc with rfl | rfl | rfl <;> rfl
      simpa only [data] using community.weights_result square ((mem_int square [2, 17, 33]).mpr cc) heap globals
    · by_cases ch : square ∈ [7, 22, 36]
      · have data : chance.data square = modelData square := chance.data_model square ch
        simpa only [chance.Post, data] using chance.result square ((mem_int square [7, 22, 36]).mpr ch) heap globals
      · simpa [modelData, Monopoly.landingWeights, jail, cc, ch] using
          ordinary_weights_result square heap globals (by exact_mod_cast jail)
            (fun member => cc ((mem_int square [2, 17, 33]).mp member))
            (fun member => ch ((mem_int square [7, 22, 36]).mp member))

end solvers.«84».resolve_landing
