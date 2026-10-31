import solvers.«83.py.lean.d».NeighborAgreement
import ProjectEulerSolutions.Lib.DijkstraMarkedNeighbors

namespace solvers.«83»
open ProjectEulerSolutions
open RectangularMatrixCalls (MatrixRows)
open GridShortestPaths (Cell cost)
open DijkstraGrid (Invariant distance mark update grid)
open FiniteGridIndex (index)
set_option Elab.async false

/-- The settled-neighbor premise follows from the grid invariant, including after marking the selected vertex. -/
theorem neighbors_grid_agreement (rows : MatrixRows) (width : Nat)
    (matrixWidth : ((rows.map Prod.snd).headD []).length = width)
    (source selected : Cell (rows.map Prod.snd)) (state : P83.SearchState)
    (invariant : Invariant (rows.map Prod.snd) source state)
    (finite : distance state selected < ProjectEulerStatements.P83.bigVal (rows.map Prod.snd))
    (values : List Int) (represented : values.toArray = state.distances) :
    (neighborsDistances rows width selected.1.val selected.2.val (distance state selected) values).toArray =
      (update (rows.map Prod.snd) state selected).distances := by
  have rowInside : selected.1.val < rows.length := by simpa using selected.1.isLt
  have columnInside : selected.2.val < width := by simpa only [matrixWidth] using selected.2.isLt
  have size : values.length = rows.length * width := by
    have sizes := invariant.distances_size
    rw [← represented] at sizes
    simpa only [List.size_toArray, DijkstraGrid.total, List.length_map, matrixWidth] using sizes
  have agreement := neighbors_agreement rows width selected.1.val selected.2.val (distance state selected)
    values (mark state selected) size rowInside columnInside (by
      intro direction active done
      have bounds := direction.bounds selected.1.val selected.2.val rows.length width rowInside columnInside active
      let neighbor : Cell (rows.map Prod.snd) :=
        (⟨direction.targetRow selected.1.val, by simpa only [List.length_map] using bounds.1⟩,
         ⟨direction.targetColumn selected.2.val, by simpa only [matrixWidth] using bounds.2.1⟩)
      have readIndex : index neighbor = direction.targetRow selected.1.val * width +
          direction.targetColumn selected.2.val := by simp only [index, neighbor, matrixWidth]
      have edge : (grid (rows.map Prod.snd)).Adj neighbor selected := by
        change FiniteGridGraph.near
          ((direction.targetRow selected.1.val : Int), (direction.targetColumn selected.2.val : Int))
          (selected.1.val, selected.2.val)
        rw [← direction.cell_eq_target selected.1.val selected.2.val rows.length width active]
        cases direction <;> simp [NeighborDirection.cell, FiniteGridGraph.near]
      have bound := DijkstraGrid.marked_neighbor_le_candidate (rows.map Prod.snd) source state invariant
        selected neighbor finite edge (by simpa only [readIndex] using done)
      have inside : direction.targetRow selected.1.val * width + direction.targetColumn selected.2.val <
          values.length := by rw [size]; exact bounds.2.2
      rw [distance, ← represented, readIndex] at bound
      simpa [getElem!_pos, inside, List.getD_eq_getElem _ _ inside,
        cost, GridRelaxationAgreement.cost, FiniteGridGraph.encode, neighbor] using bound)
  simpa only [update, FiniteGridGraph.encode, List.length_map, matrixWidth, represented] using agreement

/-- Replacing the distance buffer by the source result preserves the complete mathematical invariant. -/
theorem neighbors_grid_invariant (rows : MatrixRows) (width : Nat)
    (matrixWidth : ((rows.map Prod.snd).headD []).length = width)
    (nonnegative : ∀ row ∈ rows.map Prod.snd, ∀ value ∈ row, 0 ≤ value)
    (source selected : Cell (rows.map Prod.snd)) (state : P83.SearchState)
    (invariant : Invariant (rows.map Prod.snd) source state)
    (pending : DijkstraGrid.visited state selected = false)
    (finite : distance state selected < ProjectEulerStatements.P83.bigVal (rows.map Prod.snd))
    (least : ∀ v : Cell (rows.map Prod.snd), DijkstraGrid.visited state v = false →
      distance state selected ≤ distance state v)
    (values : List Int) (represented : values.toArray = state.distances) :
    Invariant (rows.map Prod.snd) source
      ⟨mark state selected,
       (neighborsDistances rows width selected.1.val selected.2.val (distance state selected) values).toArray⟩ := by
  rw [neighbors_grid_agreement rows width matrixWidth source selected state invariant finite values represented]
  exact DijkstraGrid.update_invariant (rows.map Prod.snd) nonnegative source state invariant selected pending finite least

end solvers.«83»
