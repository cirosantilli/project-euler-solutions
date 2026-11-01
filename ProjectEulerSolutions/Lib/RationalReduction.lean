import ProjectEulerSolutions.Lib.ArithmeticReduction
import ProjectEulerSolutions.Lib.RationalChoices
import ProjectEulerSolutions.Lib.PairSelection

/-! Index-pair enumeration implements unrestricted arithmetic pair reduction. -/
namespace ProjectEulerSolutions.RationalPair
open ArithmeticExpressions IndexedFilter
set_option Elab.async false

/-- Every reduction of a nonterminal state starts with one of the source's increasing index pairs. -/
theorem reduces_iff_index_pair (pairs : List Pair) (answer : Rat)
    (positive : ∀ pair ∈ pairs, 0 < pair.2) (multiple : 1 < pairs.length) :
    Reduces (pairs.map value) answer ↔
      ∃ (i j : Nat) (ordered : i < j) (upper : j < pairs.length) (combined : Pair),
        combined ∈ choices (pairs[i]'(by omega)) pairs[j] ∧
        Reduces ((combined :: select (exceptTwo i j) 0 pairs).map value) answer := by
  constructor
  · intro derived
    generalize original : pairs.map value = inputs at derived
    cases derived with
    | leaf a =>
      have size := congrArg List.length original
      simp only [List.length_map, List.length_singleton] at size
      omega
    | @step inputs rest a b combined answer selected operation continued =>
      rw [← original] at selected
      obtain ⟨i, j, ordered, upper, picked, remainder⟩ := pair_indices selected
      have jInside : j < pairs.length := by simpa using upper
      have iInside : i < pairs.length := by omega
      have positiveI := positive pairs[i] (List.getElem_mem iInside)
      have positiveJ := positive pairs[j] (List.getElem_mem jInside)
      have foundI : (pairs.map value)[i]? = some (value pairs[i]) := by simp [List.getElem?_map, List.getElem?_eq_getElem iInside]
      have foundJ : (pairs.map value)[j]? = some (value pairs[j]) := by simp [List.getElem?_map, List.getElem?_eq_getElem jInside]
      have chosen : combined ∈ (choices pairs[i] pairs[j]).map value := by
        rw [mem_choices_values _ _ positiveI positiveJ]
        rcases picked with ⟨first, second⟩ | ⟨first, second⟩
        · have equalI := Option.some.inj (foundI.symm.trans first)
          have equalJ := Option.some.inj (foundJ.symm.trans second)
          exact .inl (by simpa only [equalI, equalJ] using operation)
        · have equalI := Option.some.inj (foundI.symm.trans first)
          have equalJ := Option.some.inj (foundJ.symm.trans second)
          exact .inr (by simpa only [equalI, equalJ] using operation)
      obtain ⟨pair, member, equal⟩ := List.mem_map.mp chosen
      rw [select_map] at remainder
      refine ⟨i, j, ordered, jInside, pair, member, ?_⟩
      simpa only [List.map_cons, equal] using continued.perm (remainder.symm.cons combined)
  · rintro ⟨i, j, ordered, upper, combined, member, continued⟩
    have inside : i < pairs.length := by omega
    have selected := (select_two_perm pairs i j ordered upper).map value
    have operations := (mem_choices_values pairs[i] pairs[j]
      (positive pairs[i] (List.getElem_mem inside)) (positive pairs[j] (List.getElem_mem upper)) (value combined)).mp
        (List.mem_map.mpr ⟨combined, member, rfl⟩)
    rcases operations with forward | backward
    · exact Reduces.step selected forward continued
    · exact Reduces.step (selected.trans (List.Perm.swap _ _ _)) backward continued

end ProjectEulerSolutions.RationalPair
