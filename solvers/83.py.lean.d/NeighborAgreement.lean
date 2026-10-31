import solvers.«83.py.lean.d».Neighbors
import solvers.«83.py.lean.d».NeighborCells
import ProjectEulerSolutions.Lib.DijkstraListRelaxation

namespace solvers.«83»
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
open ProjectEulerSolutions.DijkstraArrayRelaxation
open ProjectEulerSolutions.P83 (relax)
set_option Elab.async false

theorem neighborUpdate_le (rows : MatrixRows) (width r c : Nat) (distance : Int)
    (direction : NeighborDirection) (values : List Int) (i : Nat) (inside : i < values.length) :
    (neighborUpdate rows width r c distance direction values).getD i 0 ≤ values.getD i 0 := by
  unfold neighborUpdate
  split
  · have updatedInside : i < (values.set (direction.targetRow r * width + direction.targetColumn c)
        (min (values.getD (direction.targetRow r * width + direction.targetColumn c) 0)
          (distance + ProjectEulerStatements.P83.getD2 (rows.map Prod.snd)
            (direction.targetRow r) (direction.targetColumn c)))).length := by simpa using inside
    rw [List.getD_eq_getElem _ _ updatedInside, List.getD_eq_getElem _ _ inside]
    by_cases same : direction.targetRow r * width + direction.targetColumn c = i
    · simp only [same, List.getElem_set_self]
      rw [List.getD_eq_getElem _ _ inside]
      exact Int.min_le_left _ _
    · rw [List.getElem_set_ne same]
  · exact le_refl _

theorem neighbor_update_agreement (rows : MatrixRows) (width r c : Nat) (distance : Int)
    (direction : NeighborDirection) (values : List Int) (visited : Array Bool)
    (size : values.length = rows.length * width) (rowInside : r < rows.length) (columnInside : c < width)
    (active : direction.enabled r c rows.length width)
    (settled : visited[direction.targetRow r * width + direction.targetColumn c]! = true →
      values.getD (direction.targetRow r * width + direction.targetColumn c) 0 ≤
        distance + ProjectEulerStatements.P83.getD2 (rows.map Prod.snd)
          (direction.targetRow r) (direction.targetColumn c)) :
    (neighborUpdate rows width r c distance direction values).toArray =
      relax (rows.map Prod.snd) width visited distance values.toArray
        ((direction.targetRow r : Int), (direction.targetColumn c : Int)) := by
  have inside : direction.targetRow r * width + direction.targetColumn c < values.length := by
    rw [size]
    exact (direction.bounds r c rows.length width rowInside columnInside active).2.2
  have agreement := list_relax_agreement (rows.map Prod.snd) width visited distance values
    ((direction.targetRow r : Int), (direction.targetColumn c : Int)) (by simpa using inside)
    (by simpa [List.getD, List.getElem?_eq_getElem inside] using settled)
  simpa [neighborUpdate, active, List.getD, List.getElem?_eq_getElem inside] using agreement

theorem neighbor_fold_agreement (directions : List NeighborDirection) (rows : MatrixRows)
    (width r c : Nat) (distance : Int) (values : List Int) (visited : Array Bool)
    (size : values.length = rows.length * width) (rowInside : r < rows.length) (columnInside : c < width)
    (settled : ∀ direction ∈ directions, direction.enabled r c rows.length width →
      visited[direction.targetRow r * width + direction.targetColumn c]! = true →
      values.getD (direction.targetRow r * width + direction.targetColumn c) 0 ≤
        distance + ProjectEulerStatements.P83.getD2 (rows.map Prod.snd)
          (direction.targetRow r) (direction.targetColumn c)) :
    (directions.foldl (fun values direction => neighborUpdate rows width r c distance direction values) values).toArray =
      (neighborCells directions r c rows.length width).foldl
        (relax (rows.map Prod.snd) width visited distance) values.toArray := by
  induction directions generalizing values with
  | nil => rfl
  | cons direction rest ih =>
      have nextSize : (neighborUpdate rows width r c distance direction values).length = rows.length * width := by
        rw [neighborUpdate_length, size]
      have nextSettled : ∀ other ∈ rest, other.enabled r c rows.length width →
          visited[other.targetRow r * width + other.targetColumn c]! = true →
          (neighborUpdate rows width r c distance direction values).getD
            (other.targetRow r * width + other.targetColumn c) 0 ≤
            distance + ProjectEulerStatements.P83.getD2 (rows.map Prod.snd)
              (other.targetRow r) (other.targetColumn c) := by
        intro other member active done
        have inside : other.targetRow r * width + other.targetColumn c < values.length := by
          rw [size]
          exact (other.bounds r c rows.length width rowInside columnInside active).2.2
        exact Int.le_trans (neighborUpdate_le rows width r c distance direction values _ inside)
          (settled other (List.mem_cons_of_mem _ member) active done)
      rw [List.foldl_cons, ih _ nextSize nextSettled]
      by_cases active : direction.enabled r c rows.length width
      · rw [neighbor_update_agreement rows width r c distance direction values visited size rowInside columnInside
          active (settled direction (by simp) active)]
        simp only [neighborCells, List.filterMap_cons, if_pos active, List.foldl_cons]
      · simp only [neighborCells, List.filterMap_cons, if_neg active, neighborUpdate]

/-- The actual four-branch distance buffer agrees with the model's neighbor fold. -/
theorem neighbors_agreement (rows : MatrixRows) (width r c : Nat) (distance : Int)
    (values : List Int) (visited : Array Bool)
    (size : values.length = rows.length * width) (rowInside : r < rows.length) (columnInside : c < width)
    (settled : ∀ direction : NeighborDirection, direction.enabled r c rows.length width →
      visited[direction.targetRow r * width + direction.targetColumn c]! = true →
      values.getD (direction.targetRow r * width + direction.targetColumn c) 0 ≤
        distance + ProjectEulerStatements.P83.getD2 (rows.map Prod.snd)
          (direction.targetRow r) (direction.targetColumn c)) :
    (neighborsDistances rows width r c distance values).toArray =
      (ProjectEulerStatements.P83.neighbors r c rows.length width).foldl
        (relax (rows.map Prod.snd) width visited distance) values.toArray := by
  rw [neighborsDistances, neighbor_fold_agreement _ rows width r c distance values visited
    size rowInside columnInside (fun direction _ => settled direction)]
  exact fold_perm _ width visited distance values.toArray _ _
    (neighborCells_perm r c rows.length width rowInside columnInside)

end solvers.«83»
