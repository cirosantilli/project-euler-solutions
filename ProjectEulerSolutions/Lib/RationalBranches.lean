import ProjectEulerSolutions.Lib.RationalReduction
import ProjectEulerSolutions.Lib.RationalResultSpec

/-! Result predicates for accumulating the increasing-index pair search. -/
namespace ProjectEulerSolutions.RationalPair
open IndexedFilter ArithmeticExpressions
set_option Elab.async false

def PairExpected (pairs : List Pair) (i j : Nat) (result : Pair) : Prop :=
  ∃ (ordered : i < j) (upper : j < pairs.length),
    ∃ combined ∈ choices (pairs[i]'(by omega)) pairs[j], Expected (combined :: select (exceptTwo i j) 0 pairs) result

theorem pairExpected_iff (pairs : List Pair) (i j : Nat) (ordered : i < j) (upper : j < pairs.length) (result : Pair) :
    PairExpected pairs i j result ↔
      ∃ combined ∈ choices (pairs[i]'(by omega)) pairs[j], Expected (combined :: select (exceptTwo i j) 0 pairs) result := by
  constructor
  · rintro ⟨_, _, combined, chosen, expected⟩
    exact ⟨combined, chosen, expected⟩
  · intro expected
    exact ⟨ordered, upper, expected⟩

theorem expected_iff_pair (pairs : List Pair) (result : Pair)
    (positive : ∀ pair ∈ pairs, 0 < pair.2) (multiple : 1 < pairs.length) :
    Expected pairs result ↔ ∃ i j, PairExpected pairs i j result := by
  constructor
  · rintro ⟨canonical, derived⟩
    obtain ⟨i, j, ordered, upper, combined, chosen, continued⟩ :=
      (reduces_iff_index_pair pairs (value result) positive multiple).mp derived
    exact ⟨i, j, ordered, upper, combined, chosen, canonical, continued⟩
  · rintro ⟨i, j, ordered, upper, combined, chosen, canonical, continued⟩
    exact ⟨canonical, (reduces_iff_index_pair pairs (value result) positive multiple).mpr
      ⟨i, j, ordered, upper, combined, chosen, continued⟩⟩

end ProjectEulerSolutions.RationalPair
