import ProjectEulerSolutions.P18
import ProjectEulerStatements.Lib.TrianglePaths

/-! Equivalence of the two bottom-up triangle reductions. -/
namespace ProjectEulerSolutions.P18

private theorem combineRow_eq (row below : List Nat) :
    combineRow row below = ProjectEulerStatements.P18.maxRow row below := by
  induction row generalizing below with
  | nil => rfl
  | cons x xs ih =>
      cases below with
      | nil => rfl
      | cons y ys =>
          cases ys with
          | nil => rfl
          | cons z zs =>
              simp only [combineRow, ProjectEulerStatements.P18.maxRow, ih]
              congr 2
              rw [Nat.max_eq_max]
              by_cases larger : y > z
              · rw [if_pos larger, Nat.max_eq_left (by omega)]
              · rw [if_neg larger, Nat.max_eq_right (by omega)]

theorem equiv (tri : ProjectEulerStatements.P18.Triangle) :
    ProjectEulerStatements.P18.naive tri = solve tri := by
  have size := ProjectEulerStatements.P18.reduceRows_triangle_length tri
  rw [ProjectEulerStatements.P18.naive_eq_reduceRows]
  have optimized : solve tri = match ProjectEulerStatements.P18.reduceRows tri.1 with
      | [x] => x
      | _ => 0 := by
    cases reversed : tri.1.reverse with
    | nil => simp [solve, maxPathSum, ProjectEulerStatements.P18.reduceRows, reversed]
    | cons bottom rest =>
        simp only [solve, maxPathSum, ProjectEulerStatements.P18.reduceRows, combineRow_eq, reversed]
        rfl
  rw [optimized]
  cases reduced : ProjectEulerStatements.P18.reduceRows tri.1 with
  | nil => rfl
  | cons x xs =>
      cases xs with
      | nil => rfl
      | cons y ys => simp [reduced] at size

end ProjectEulerSolutions.P18
