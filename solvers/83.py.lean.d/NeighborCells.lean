import solvers.«83.py.lean.d».NeighborArithmetic
import ProjectEulerSolutions.Lib.DijkstraRelaxationOrder

namespace solvers.«83»
set_option Elab.async false

namespace NeighborDirection

def cell (direction : NeighborDirection) (r c : Nat) : Int × Int := match direction with
  | .up => (r - 1, c)
  | .down => (r + 1, c)
  | .left => (r, c - 1)
  | .right => (r, c + 1)

theorem enabled_iff_valid (direction : NeighborDirection) (r c height width : Nat)
    (rowInside : r < height) (columnInside : c < width) :
    direction.enabled r c height width ↔
      0 ≤ (direction.cell r c).1 ∧ 0 ≤ (direction.cell r c).2 ∧
        (direction.cell r c).1 < height ∧ (direction.cell r c).2 < width := by
  cases direction <;> simp only [enabled, cell] <;> omega

theorem cell_eq_target (direction : NeighborDirection) (r c height width : Nat)
    (active : direction.enabled r c height width) :
    direction.cell r c = ((direction.targetRow r : Int), (direction.targetColumn c : Int)) := by
  cases direction <;> simp only [cell, targetRow, targetColumn, enabled] at * <;>
    apply Prod.ext <;> omega

end NeighborDirection

def neighborCells (directions : List NeighborDirection) (r c height width : Nat) : List (Int × Int) :=
  directions.filterMap fun direction =>
    if direction.enabled r c height width then
      some (direction.targetRow r, direction.targetColumn c)
    else none

theorem neighborCells_filter (directions : List NeighborDirection) (r c height width : Nat)
    (rowInside : r < height) (columnInside : c < width) :
    neighborCells directions r c height width =
      (directions.map fun direction => direction.cell r c).filter
        (fun p => decide (0 ≤ p.1 ∧ 0 ≤ p.2 ∧ p.1 < (height : Int) ∧ p.2 < (width : Int))) := by
  induction directions with
  | nil => rfl
  | cons direction rest ih =>
      have valid := direction.enabled_iff_valid r c height width rowInside columnInside
      by_cases active : direction.enabled r c height width
      · have coordinates := direction.cell_eq_target r c height width active
        simp only [neighborCells, List.filterMap_cons, if_pos active, List.map_cons,
          List.filter_cons, valid.mp active]
        rw [← coordinates]
        exact congrArg (List.cons _) ih
      · simp only [neighborCells, List.filterMap_cons, if_neg active, List.map_cons,
          List.filter_cons, show ¬ (0 ≤ (direction.cell r c).1 ∧ 0 ≤ (direction.cell r c).2 ∧
            (direction.cell r c).1 < height ∧ (direction.cell r c).2 < width) from
              fun h => active (valid.mpr h), decide_false, Bool.false_eq_true, if_false]
        exact ih

/-- The source order and model order enumerate the same valid neighbors. -/
theorem neighborCells_perm (r c height width : Nat)
    (rowInside : r < height) (columnInside : c < width) :
    (neighborCells [.up, .down, .left, .right] r c height width).Perm
      (ProjectEulerStatements.P83.neighbors r c height width) := by
  have order : ([.up, .down, .left, .right] : List NeighborDirection).Perm
      [.down, .right, .left, .up] :=
    (List.Perm.swap NeighborDirection.down .up [.left, .right]).trans
      (List.Perm.cons NeighborDirection.down (List.reverse_perm ([.up, .left, .right] : List NeighborDirection)).symm)
  have mapped := order.filterMap (fun direction =>
    if direction.enabled r c height width then
      some ((direction.targetRow r : Int), (direction.targetColumn c : Int)) else none)
  change (neighborCells _ r c height width).Perm (neighborCells _ r c height width) at mapped
  have model : neighborCells [.down, .right, .left, .up] r c height width =
      ProjectEulerStatements.P83.neighbors r c height width :=
    neighborCells_filter _ r c height width rowInside columnInside
  rw [model] at mapped
  exact mapped

end solvers.«83»
