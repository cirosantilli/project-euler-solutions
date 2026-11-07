import ProjectEulerSolutions.Lib.PowerDigitSum
import ProjectEulerStatements.P56

namespace ProjectEulerSolutions.PowerDigitSum

private theorem row_nat (a : Nat) (bs : List Nat) (initial : Nat) :
    row a bs initial = (bs.foldl (fun total b => if digitSum (a ^ b) > total then
      digitSum (a ^ b) else total) initial : Nat) := by
  induction bs generalizing initial with
  | nil => rfl
  | cons b bs ih =>
      rw [row_cons, List.foldl_cons]
      unfold score
      by_cases larger : initial < digitSum (a ^ b)
      · rw [if_pos larger, max_eq_right (by exact_mod_cast Nat.le_of_lt larger)]
        exact ih _
      · rw [if_neg larger, max_eq_left (by exact_mod_cast Nat.le_of_not_gt larger)]
        exact ih _

private theorem scan_nat (bs as : List Nat) (initial : Nat) :
    scan bs as initial = (as.foldl (fun total a => bs.foldl (fun total b =>
      if digitSum (a ^ b) > total then digitSum (a ^ b) else total) total) initial : Nat) := by
  induction as generalizing initial with
  | nil => rfl
  | cons a as ih =>
      rw [scan_cons, row_nat, List.foldl_cons, ih]

theorem maximum_eq_naive (n : Nat) : maximum n = (ProjectEulerStatements.P56.naive n : Int) := by
  exact scan_nat (List.range n) (List.range n) 0

end ProjectEulerSolutions.PowerDigitSum
