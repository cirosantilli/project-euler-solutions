import ProjectEulerSolutions.Lib.GridRelaxation
import ProjectEulerSolutions.Lib.VertexRelaxation

/-! Relating the reference's nested matrix updates to graph relaxation. -/
namespace ProjectEulerSolutions.GridRelaxationAgreement
open ProjectEulerStatements.P83

abbrev Position := Int × Int

def valid (matrix : List (List Int)) (p : Position) : Prop :=
  0 ≤ p.1 ∧ 0 ≤ p.2 ∧ p.1 < matrix.length ∧ p.2 < (matrix.headD []).length

def adjacent (matrix : List (List Int)) (p : Position) : List Position :=
  neighbors p.1 p.2 matrix.length (matrix.headD []).length

def cost (matrix : List (List Int)) (p : Position) : Int := getD2 matrix p.1 p.2

theorem adjacent_valid (matrix : List (List Int)) (p q : Position)
    (member : q ∈ adjacent matrix p) : valid matrix q := by
  simpa [adjacent, neighbors, valid] using (List.mem_filter.mp member).2

theorem getD2_nonnegative (matrix : List (List Int))
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value) (r c : Int) :
    0 ≤ getD2 matrix r c := by
  unfold getD2
  split
  · simp only [List.getD_eq_getElem?_getD]
    cases rowEq : matrix[r.toNat]? with
    | none => simp
    | some row =>
        simp only [Option.getD_some]
        cases cellEq : row[c.toNat]? with
        | none => simp
        | some value =>
            exact nonnegative row (List.mem_of_getElem? rowEq) value (List.mem_of_getElem? cellEq)
  · exact le_refl _

theorem cell_eq_step (matrix distances : List (List Int)) (p : Position)
    (nonnegative : 0 ≤ cost matrix p) :
    GridRelaxation.cell matrix distances p.1 p.2 =
      VertexRelaxation.step (cost matrix) (adjacent matrix) (cost distances) p := by
  exact (VertexRelaxation.step_eq_cost_fold (cost matrix) (adjacent matrix)
    (cost distances) p nonnegative).symm

theorem read_relaxStep (matrix distances : List (List Int)) (r c : Nat)
    (rowBound : r < matrix.length) (colBound : c < (matrix.headD []).length) :
    getD2 (relaxStep matrix distances) r c = GridRelaxation.cell matrix distances r c := by
  have colBound' : c < (matrix.head?.getD []).length := by simpa using colBound
  simp [getD2, GridRelaxation.relaxStep_eq, List.getD_eq_getElem?_getD, rowBound, colBound']

theorem read_initial (matrix : List (List Int)) (r c : Nat)
    (rowBound : r < matrix.length) (colBound : c < (matrix.headD []).length) :
    getD2 (initDist matrix) r c =
      VertexRelaxation.initial (cost matrix) (bigVal matrix) (0, 0) (r, c) := by
  have colBound' : c < (matrix.head?.getD []).length := by simpa using colBound
  simp [getD2, initDist, dims, List.getD_eq_getElem?_getD, rowBound, colBound',
    VertexRelaxation.initial, cost, Prod.ext_iff]

private theorem fold_congr (values : List Position) (left right : Position → Int)
    (agree : ∀ p ∈ values, left p = right p) (amount best : Int) :
    values.foldl (fun acc p => min acc (amount + left p)) best =
      values.foldl (fun acc p => min acc (amount + right p)) best := by
  induction values generalizing best with
  | nil => rfl
  | cons p rest ih =>
      simp only [List.foldl_cons, agree p (by simp)]
      exact ih (fun p member => agree p (by simp [member])) _

def rounds (matrix : List (List Int)) : Nat → List (List Int)
  | 0 => initDist matrix
  | n + 1 => relaxStep matrix (rounds matrix n)

/-- Each matrix iteration agrees with the scalar graph recurrence at every valid cell. -/
theorem rounds_eq_run (matrix : List (List Int))
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value) (n : Nat) :
    ∀ p, valid matrix p → cost (rounds matrix n) p =
      VertexRelaxation.run (cost matrix) (adjacent matrix) (bigVal matrix) (0, 0) n p := by
  induction n with
  | zero =>
      intro p inside
      have rowBound : p.1.toNat < matrix.length := by obtain ⟨_, _, _, _⟩ := inside; omega
      have colBound : p.2.toNat < (matrix.headD []).length := by obtain ⟨_, _, _, _⟩ := inside; omega
      have result := read_initial matrix p.1.toNat p.2.toNat rowBound colBound
      simpa [rounds, cost, VertexRelaxation.run, Int.toNat_of_nonneg inside.1,
        Int.toNat_of_nonneg inside.2.1] using result
  | succ n ih =>
      intro p inside
      have rowBound : p.1.toNat < matrix.length := by obtain ⟨_, _, _, _⟩ := inside; omega
      have colBound : p.2.toNat < (matrix.headD []).length := by obtain ⟨_, _, _, _⟩ := inside; omega
      have read := read_relaxStep matrix (rounds matrix n) p.1.toNat p.2.toNat rowBound colBound
      have read' : cost (rounds matrix (n + 1)) p =
          GridRelaxation.cell matrix (rounds matrix n) p.1 p.2 := by
        simpa [cost, rounds, Int.toNat_of_nonneg inside.1,
          Int.toNat_of_nonneg inside.2.1] using read
      rw [read', cell_eq_step _ _ _ (getD2_nonnegative matrix nonnegative _ _)]
      change (adjacent matrix p).foldl _ (cost (rounds matrix n) p) =
        (adjacent matrix p).foldl _ _
      rw [ih p inside]
      exact fold_congr _ _ _ (fun q member => ih q (adjacent_valid matrix p q member)) _ _

theorem iterate_eq_rounds (matrix : List (List Int)) :
    iterate matrix (initDist matrix) =
      rounds matrix (matrix.length * (matrix.headD []).length) := by
  have fold (n : Nat) :
      (List.range n).foldl (fun current _ => relaxStep matrix current) (initDist matrix) =
        rounds matrix n := by
    induction n with
    | zero => rfl
    | succ n ih => simp [List.range_succ, List.foldl_append, ih, rounds]
  exact fold _

theorem naive_eq_run (matrix : List (List Int))
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (rowsPositive : 0 < matrix.length) (colsPositive : 0 < (matrix.headD []).length) :
    naive matrix = VertexRelaxation.run (cost matrix) (adjacent matrix) (bigVal matrix)
      (0, 0) (matrix.length * (matrix.headD []).length)
      ((matrix.length : Int) - 1, ((matrix.headD []).length : Int) - 1) := by
  unfold naive minPath dims
  rw [iterate_eq_rounds]
  exact rounds_eq_run matrix nonnegative _
    ((matrix.length : Int) - 1, ((matrix.headD []).length : Int) - 1)
    (by unfold valid; dsimp only; omega)

theorem naive_empty : naive [] = 0 := by
  simp [naive, minPath, dims, getD2]

theorem naive_zero_width (matrix : List (List Int))
    (emptyWidth : (matrix.headD []).length = 0) : naive matrix = 0 := by
  have emptyWidth' : (matrix.head?.getD []).length = 0 := by simpa using emptyWidth
  simp [naive, minPath, dims, emptyWidth', getD2]

end ProjectEulerSolutions.GridRelaxationAgreement
