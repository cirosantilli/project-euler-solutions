import ProjectEulerSolutions.P83
import Mathlib.Data.Int.Order.Basic

/-! Exact array effects of the relaxation operation in the Dijkstra model. -/
namespace ProjectEulerSolutions.DijkstraArrayRelaxation
open ProjectEulerSolutions.P83

private theorem set_read (values : Array Int) (target position : Nat) (replacement : Int)
    (inside : position < values.size) :
    (values.set! target replacement)[position]! =
      if target = position then replacement else values[position]! := by
  simp [Array.set!_eq_setIfInBounds, getElem!_pos, inside, Array.getElem_setIfInBounds]

theorem relax_size (matrix : List (List Int)) (width : Nat) (visited : Array Bool)
    (distance : Int) (distances : Array Int) (cell : Int × Int) :
    (relax matrix width visited distance distances cell).size = distances.size := by
  dsimp only [relax]
  split <;> simp

/-- Only an unvisited target can change, and its new value is exactly the smaller candidate. -/
theorem relax_read (matrix : List (List Int)) (width : Nat) (visited : Array Bool)
    (distance : Int) (distances : Array Int) (cell : Int × Int) (position : Nat)
    (inside : position < distances.size) :
    (relax matrix width visited distance distances cell)[position]! =
      if cell.1.toNat * width + cell.2.toNat = position ∧ visited[position]! = false
      then min distances[position]! (distance + ProjectEulerStatements.P83.getD2 matrix cell.1 cell.2)
      else distances[position]! := by
  dsimp only [relax]
  simp only [Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq]
  by_cases update : visited[cell.1.toNat * width + cell.2.toNat]! = false ∧
      distance + ProjectEulerStatements.P83.getD2 matrix cell.1 cell.2 <
        distances[cell.1.toNat * width + cell.2.toNat]!
  · rw [if_pos update]
    rw [set_read distances _ position _ inside]
    by_cases same : cell.1.toNat * width + cell.2.toNat = position
    · rw [same] at update
      simp only [same, update.1, and_self, if_true]
      exact (min_eq_right (le_of_lt update.2)).symm
    · simp [same]
  · rw [if_neg update]
    by_cases affected : cell.1.toNat * width + cell.2.toNat = position ∧ visited[position]! = false
    · obtain ⟨same, pending⟩ := affected
      simp only [same, pending, true_and] at update
      simp only [same, pending, and_self, if_true]
      exact (min_eq_left (by omega)).symm
    · simp [affected]

theorem relax_le (matrix : List (List Int)) (width : Nat) (visited : Array Bool)
    (distance : Int) (distances : Array Int) (cell : Int × Int) (position : Nat)
    (inside : position < distances.size) :
    (relax matrix width visited distance distances cell)[position]! ≤ distances[position]! := by
  rw [relax_read matrix width visited distance distances cell position inside]
  split
  · exact Int.min_le_left _ _
  · exact le_refl _

theorem relax_settled (matrix : List (List Int)) (width : Nat) (visited : Array Bool)
    (distance : Int) (distances : Array Int) (cell : Int × Int) (position : Nat)
    (inside : position < distances.size) (settled : visited[position]! = true) :
    (relax matrix width visited distance distances cell)[position]! = distances[position]! := by
  rw [relax_read matrix width visited distance distances cell position inside]
  simp [settled]

theorem fold_size (matrix : List (List Int)) (width : Nat) (visited : Array Bool)
    (distance : Int) (distances : Array Int) (cells : List (Int × Int)) :
    (cells.foldl (relax matrix width visited distance) distances).size = distances.size := by
  induction cells generalizing distances with
  | nil => rfl
  | cons cell rest ih => simp only [List.foldl_cons, ih, relax_size]

theorem fold_le (matrix : List (List Int)) (width : Nat) (visited : Array Bool)
    (distance : Int) (distances : Array Int) (cells : List (Int × Int)) (position : Nat)
    (inside : position < distances.size) :
    (cells.foldl (relax matrix width visited distance) distances)[position]! ≤ distances[position]! := by
  induction cells generalizing distances with
  | nil => exact le_refl _
  | cons cell rest ih =>
      exact Int.le_trans (ih _ (by simpa only [relax_size] using inside))
        (relax_le matrix width visited distance distances cell position inside)

theorem fold_settled (matrix : List (List Int)) (width : Nat) (visited : Array Bool)
    (distance : Int) (distances : Array Int) (cells : List (Int × Int)) (position : Nat)
    (inside : position < distances.size) (settled : visited[position]! = true) :
    (cells.foldl (relax matrix width visited distance) distances)[position]! = distances[position]! := by
  induction cells generalizing distances with
  | nil => rfl
  | cons cell rest ih =>
      rw [List.foldl_cons, ih _ (by simpa only [relax_size] using inside)]
      exact relax_settled matrix width visited distance distances cell position inside settled

/-- Traversing all neighbors establishes the relaxation inequality at each unvisited target. -/
theorem fold_neighbor_bound (matrix : List (List Int)) (width : Nat) (visited : Array Bool)
    (distance : Int) (distances : Array Int) (cells : List (Int × Int)) (cell : Int × Int)
    (member : cell ∈ cells) (position : Nat)
    (target : cell.1.toNat * width + cell.2.toNat = position)
    (inside : position < distances.size) (pending : visited[position]! = false) :
    (cells.foldl (relax matrix width visited distance) distances)[position]! ≤
      distance + ProjectEulerStatements.P83.getD2 matrix cell.1 cell.2 := by
  induction cells generalizing distances with
  | nil => simp at member
  | cons first rest ih =>
      simp only [List.foldl_cons]
      rcases List.mem_cons.mp member with same | member
      · subst first
        have lower := fold_le matrix width visited distance
          (relax matrix width visited distance distances cell) rest position
          (by simpa only [relax_size] using inside)
        rw [relax_read matrix width visited distance distances cell position inside] at lower
        simp only [target, pending, and_self, if_true] at lower
        exact Int.le_trans lower (Int.min_le_right _ _)
      · exact ih _ member (by simpa only [relax_size] using inside)

end ProjectEulerSolutions.DijkstraArrayRelaxation
