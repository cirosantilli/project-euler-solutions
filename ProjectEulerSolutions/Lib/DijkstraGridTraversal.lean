import ProjectEulerSolutions.Lib.DijkstraGridProgress
import ProjectEulerSolutions.Lib.DijkstraGridInitial

/-! The finite Dijkstra traversal settles every grid cell and attains a shortest walk. -/
namespace ProjectEulerSolutions.DijkstraGrid
open ProjectEulerSolutions.P83
open ProjectEulerStatements.P83 (bigVal)
open FiniteGridGraph (origin vertices)
open FiniteGridIndex (index)
open GridShortestPaths (Cell cost finish)

def run (matrix : List (List Int)) (state : SearchState) : Nat → SearchState
  | 0 => state
  | n + 1 => searchStep matrix (run matrix state n)

theorem run_invariant (matrix : List (List Int))
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (source : Cell matrix) (state : SearchState) (invariant : Invariant matrix source state) (n : Nat) :
    Invariant matrix source (run matrix state n) := by
  induction n with
  | zero => exact invariant
  | succ n ih => exact searchStep_invariant matrix nonnegative source _ ih

theorem pending_run (matrix : List (List Int))
    (shape : ∀ row ∈ matrix, row.length = (matrix.headD []).length)
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (source : Cell matrix) (reachable : ∀ v : Cell matrix, Nonempty ((grid matrix).Walk v source))
    (state : SearchState) (invariant : Invariant matrix source state) (n : Nat) :
    (pendingCells matrix (run matrix state n)).card ≤ (pendingCells matrix state).card - n := by
  induction n with
  | zero => exact le_refl _
  | succ n ih =>
      have current := run_invariant matrix nonnegative source state invariant n
      by_cases empty : (pendingCells matrix (run matrix state n)).card = 0
      · have bound := pending_step_le matrix source _ current
        change (pendingCells matrix (searchStep matrix (run matrix state n))).card ≤ _
        omega
      · have advance := pending_progress matrix shape nonnegative source reachable _ current (by omega)
        change (pendingCells matrix (searchStep matrix (run matrix state n))).card ≤ _
        omega

theorem run_settles_all (matrix : List (List Int))
    (shape : ∀ row ∈ matrix, row.length = (matrix.headD []).length)
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (source : Cell matrix) (reachable : ∀ v : Cell matrix, Nonempty ((grid matrix).Walk v source))
    (state : SearchState) (invariant : Invariant matrix source state) :
    ∀ v : Cell matrix, visited (run matrix state (total matrix)) v = true := by
  have remaining := pending_run matrix shape nonnegative source reachable state invariant (total matrix)
  have initialBound := pending_le_total matrix state
  have empty : (pendingCells matrix (run matrix state (total matrix))).card = 0 := by omega
  intro v
  cases flag : visited (run matrix state (total matrix)) v with
  | true => rfl
  | false =>
      have positive := Finset.card_pos.mpr
        ⟨v, (mem_pendingCells matrix (run matrix state (total matrix)) v).mpr flag⟩
      omega

theorem fold_eq_run (matrix : List (List Int)) (state : SearchState) (n : Nat) :
    (List.range n).foldl (fun current _ => searchStep matrix current) state = run matrix state n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, ih, run]

theorem settled_minimum (matrix : List (List Int))
    (shape : ∀ row ∈ matrix, row.length = (matrix.headD []).length)
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (source v : Cell matrix) (state : SearchState) (invariant : Invariant matrix source state)
    (done : visited state v = true) (reachable : (grid matrix).Walk v source) :
    ∃ shortest : (grid matrix).Walk v source,
      (shortest.support.map (cost matrix)).sum = distance state v ∧
      ∀ walk : (grid matrix).Walk v source,
        (shortest.support.map (cost matrix)).sum ≤ (walk.support.map (cost matrix)).sum := by
  obtain ⟨path, _, _, _, bound⟩ := NonnegativeWalkBounds.bounded_path_enumerated
    (vertices matrix.length (matrix.headD []).length) FiniteGridGraph.mem_vertices
    (grid matrix) (cost matrix) (GridShortestPaths.cost_nonnegative matrix nonnegative) reachable
  rw [GridShortestPaths.vertices_cost matrix shape, ← GridPathBounds.bigVal_eq_sum] at bound
  have finite := (invariant.settled v done path).trans_lt bound
  rcases (invariant.witness v).2 with sentinel | ⟨shortest, value⟩
  · omega
  · exact ⟨shortest, value, fun walk => value ▸ invariant.settled v done walk⟩

theorem finish_index (matrix : List (List Int))
    (rowsPositive : 0 < matrix.length) (colsPositive : 0 < (matrix.headD []).length) :
    index (finish matrix rowsPositive colsPositive) = total matrix - 1 := by
  have lower : (matrix.headD []).length ≤ total matrix := by
    simpa only [Nat.one_mul] using Nat.mul_le_mul_right (matrix.headD []).length rowsPositive
  simp only [index, finish, Nat.sub_mul, Nat.one_mul]
  unfold total at *
  omega

theorem solve_eq_final_distance (matrix : List (List Int))
    (rowsPositive : 0 < matrix.length) (colsPositive : 0 < (matrix.headD []).length) :
    solve matrix = distance (run matrix (initial matrix) (total matrix))
      (finish matrix rowsPositive colsPositive) := by
  have positive : 0 < total matrix := Nat.mul_pos rowsPositive colsPositive
  simp only [solve, dijkstraMinPathSum, ne_of_gt positive, if_false, fold_eq_run,
    distance, finish_index, initial]

/-- The array Dijkstra model returns the attained minimum between opposite grid corners. -/
theorem solve_minimum (matrix : List (List Int))
    (shape : ∀ row ∈ matrix, row.length = (matrix.headD []).length)
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (rowsPositive : 0 < matrix.length) (colsPositive : 0 < (matrix.headD []).length) :
    ∃ shortest : (grid matrix).Walk (finish matrix rowsPositive colsPositive) (origin rowsPositive colsPositive),
      (shortest.support.map (cost matrix)).sum = solve matrix ∧
      ∀ walk : (grid matrix).Walk (finish matrix rowsPositive colsPositive) (origin rowsPositive colsPositive),
        (shortest.support.map (cost matrix)).sum ≤ (walk.support.map (cost matrix)).sum := by
  have initialized := initial_invariant matrix shape nonnegative rowsPositive colsPositive
  have reachable := FiniteGridGraph.reachable_origin rowsPositive colsPositive
  have invariant := run_invariant matrix nonnegative _ _ initialized (total matrix)
  have done := run_settles_all matrix shape nonnegative _ reachable _ initialized
  obtain ⟨walk⟩ := reachable (finish matrix rowsPositive colsPositive)
  have result := settled_minimum matrix shape nonnegative _ _ _ invariant
    (done (finish matrix rowsPositive colsPositive)) walk
  rw [← solve_eq_final_distance matrix rowsPositive colsPositive] at result
  exact result

end ProjectEulerSolutions.DijkstraGrid
