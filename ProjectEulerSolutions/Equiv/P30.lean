import ProjectEulerSolutions.P30
namespace ProjectEulerSolutions.P30

theorem digitPowerSum_correct (power n : Nat) :
    digitPowerSum (digitPowers power) n = ProjectEulerStatements.P30.digitPowerSum power n := by
  have go (digits : List Nat) (bounded : ∀ d ∈ digits, d < 10) (initial : Nat) :
      digits.foldl (fun acc d => acc + (digitPowers power)[d]!) initial =
      digits.foldl (fun acc d => acc + d^power) initial := by
    induction digits generalizing initial with
    | nil => rfl
    | cons head tail ih =>
      have hh := bounded head (by simp)
      simp only [List.foldl_cons]
      have get : (digitPowers power)[head]! = head^power := by simp [digitPowers, hh]
      rw [get]
      exact ih (fun d h => bounded d (by simp [h])) _
  exact go _ (fun d h => Nat.digits_lt_base (by decide) h) 0

theorem equiv (n : Nat) : ProjectEulerStatements.P30.naive n = solve n := by
  simp only [solve, digitPowerSum_correct, ProjectEulerStatements.P30.naive,
    ProjectEulerStatements.P30.sumPowerDigits]
end ProjectEulerSolutions.P30
