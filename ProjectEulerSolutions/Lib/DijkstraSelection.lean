import ProjectEulerSolutions.P83
import Mathlib.Data.Int.Order.Basic

/-! The scan-based Dijkstra model selects a least unvisited finite estimate. -/
namespace ProjectEulerSolutions.DijkstraSelection
open ProjectEulerSolutions.P83

def choose (state : SearchState) (best : Nat × Int) (index : Nat) : Nat × Int :=
  if !state.visited[index]! && state.distances[index]! < best.2
  then (index, state.distances[index]!) else best

private theorem fold_le (state : SearchState) (indices : List Nat) (best : Nat × Int) :
    (indices.foldl (choose state) best).2 ≤ best.2 := by
  induction indices generalizing best with
  | nil => exact le_refl _
  | cons index rest ih =>
      simp only [List.foldl_cons]
      have lower : (choose state best index).2 ≤ best.2 := by
        unfold choose
        split
        · rename_i selected
          simp only [Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq] at selected
          exact le_of_lt selected.2
        · exact le_refl _
      exact Int.le_trans (ih _) lower

private theorem fold_le_member (state : SearchState) (indices : List Nat)
    (best : Nat × Int) (index : Nat) (member : index ∈ indices)
    (unvisited : state.visited[index]! = false) :
    (indices.foldl (choose state) best).2 ≤ state.distances[index]! := by
  induction indices generalizing best with
  | nil => simp at member
  | cons first rest ih =>
      simp only [List.foldl_cons]
      rcases List.mem_cons.mp member with equal | member
      · subst index
        have lower : (choose state best first).2 ≤ state.distances[first]! := by
          simp only [choose, unvisited, Bool.not_false, Bool.true_and, decide_eq_true_eq]
          split
          · exact le_refl _
          · omega
        exact Int.le_trans (fold_le state rest _) lower
      · exact ih _ member

private theorem fold_selected (state : SearchState) (indices : List Nat) (best : Nat × Int) :
    indices.foldl (choose state) best = best ∨
      ∃ index ∈ indices, state.visited[index]! = false ∧
        indices.foldl (choose state) best = (index, state.distances[index]!) := by
  induction indices generalizing best with
  | nil => exact Or.inl rfl
  | cons first rest ih =>
      rcases ih (choose state best first) with unchanged | ⟨index, member, unvisited, selected⟩
      · simp only [List.foldl_cons, unchanged]
        unfold choose
        split
        · rename_i chosen
          right
          refine ⟨first, by simp, ?_, rfl⟩
          simp only [Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq] at chosen
          exact chosen.1
        · exact Or.inl rfl
      · exact Or.inr ⟨index, by simp [member], unvisited, selected⟩

theorem findMin_le (total : Nat) (infinity : Int) (state : SearchState) :
    (findMin total infinity state).2 ≤ infinity :=
  fold_le state (List.range total) (0, infinity)

theorem findMin_le_unvisited (total : Nat) (infinity : Int) (state : SearchState)
    (index : Nat) (inside : index < total) (unvisited : state.visited[index]! = false) :
    (findMin total infinity state).2 ≤ state.distances[index]! :=
  fold_le_member state (List.range total) (0, infinity) index (List.mem_range.mpr inside) unvisited

/-- A finite remaining estimate prevents the sentinel from being selected. -/
theorem findMin_selected (total : Nat) (infinity : Int) (state : SearchState)
    (available : ∃ index < total, state.visited[index]! = false ∧ state.distances[index]! < infinity) :
    let selected := findMin total infinity state
    selected.1 < total ∧ state.visited[selected.1]! = false ∧
      selected.2 = state.distances[selected.1]! ∧ selected.2 < infinity := by
  obtain ⟨index, inside, unvisited, finite⟩ := available
  have lower := findMin_le_unvisited total infinity state index inside unvisited
  have below : (findMin total infinity state).2 < infinity := Int.lt_of_le_of_lt lower finite
  rcases fold_selected state (List.range total) (0, infinity) with unchanged |
      ⟨selected, member, unvisited, value⟩
  · have impossible : findMin total infinity state = (0, infinity) := unchanged
    rw [impossible] at below
    exact False.elim ((lt_irrefl _) below)
  · have result : findMin total infinity state = (selected, state.distances[selected]!) := value
    dsimp only
    rw [result]
    exact ⟨List.mem_range.mp member, unvisited, rfl, by simpa only [result] using below⟩

/-- If the scan returns its sentinel, no unvisited estimate lies below that bound. -/
theorem findMin_sentinel (total : Nat) (infinity : Int) (state : SearchState)
    (sentinel : (findMin total infinity state).2 = infinity)
    (index : Nat) (inside : index < total) (unvisited : state.visited[index]! = false) :
    infinity ≤ state.distances[index]! := by
  simpa only [sentinel] using findMin_le_unvisited total infinity state index inside unvisited

end ProjectEulerSolutions.DijkstraSelection
