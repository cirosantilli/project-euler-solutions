import Init.Data.List.Pairwise
import Lean.Elab.Tactic.Omega

/-! Positive descending frequencies of a five-card hand have seven possible profiles.
The all-equal profile remains present until a separate card-validity argument excludes it. -/
namespace ProjectEulerSolutions.PokerCountPatterns

set_option Elab.async false

def profiles : List (List Int) :=
  [[5], [4, 1], [3, 2], [3, 1, 1], [2, 2, 1], [2, 1, 1, 1], [1, 1, 1, 1, 1]]

/-- Positive integer counts require at least one unit of mass per list element. -/
theorem length_le_sum (counts : List Int) (positive : ∀ count ∈ counts, 0 < count) :
    (counts.length : Int) ≤ counts.sum := by
  revert positive
  induction counts with
  | nil => intro _; simp
  | cons count counts ih =>
      intro positive
      have first := positive count List.mem_cons_self
      have rest := ih (fun value member => positive value (List.mem_cons_of_mem count member))
      simp only [List.length_cons, List.sum_cons]
      omega

theorem length_le_five (counts : List Int) (positive : ∀ count ∈ counts, 0 < count)
    (total : counts.sum = 5) : counts.length ≤ 5 := by
  have bounded := length_le_sum counts positive
  omega

/-- No frequency pattern is omitted, including five equal ranks. -/
theorem classify (counts : List Int) (positive : ∀ count ∈ counts, 0 < count)
    (descending : counts.Pairwise (fun a b : Int => b ≤ a)) (total : counts.sum = 5) :
    counts ∈ profiles := by
  have bounded := length_le_five counts positive total
  cases counts with
  | nil => simp at total
  | cons a rest =>
      cases rest with
      | nil => simp_all [profiles] <;> omega
      | cons b rest =>
          cases rest with
          | nil => simp_all [profiles] <;> omega
          | cons c rest =>
              cases rest with
              | nil => simp_all [profiles] <;> omega
              | cons d rest =>
                  cases rest with
                  | nil => simp_all [profiles] <;> omega
                  | cons e rest =>
                      cases rest with
                      | nil => simp_all [profiles] <;> omega
                      | cons f rest =>
                          simp only [List.length_cons] at bounded
                          omega

theorem head_five (rest : List Int) (positive : ∀ count ∈ (5 : Int) :: rest, 0 < count)
    (descending : ((5 : Int) :: rest).Pairwise (fun a b : Int => b ≤ a))
    (total : ((5 : Int) :: rest).sum = 5) : (5 : Int) :: rest = [5] := by
  simpa [profiles] using classify ((5 : Int) :: rest) positive descending total

/-- A largest frequency of four leaves exactly one kicker. -/
theorem head_four (rest : List Int) (positive : ∀ count ∈ (4 : Int) :: rest, 0 < count)
    (descending : ((4 : Int) :: rest).Pairwise (fun a b : Int => b ≤ a))
    (total : ((4 : Int) :: rest).sum = 5) : (4 : Int) :: rest = [4, 1] := by
  simpa [profiles] using classify ((4 : Int) :: rest) positive descending total

/-- Three equal ranks form either a full house or a trip with two singleton kickers. -/
theorem head_three (rest : List Int) (positive : ∀ count ∈ (3 : Int) :: rest, 0 < count)
    (descending : ((3 : Int) :: rest).Pairwise (fun a b : Int => b ≤ a))
    (total : ((3 : Int) :: rest).sum = 5) :
    (3 : Int) :: rest = [3, 2] ∨ (3 : Int) :: rest = [3, 1, 1] := by
  simpa [profiles] using classify ((3 : Int) :: rest) positive descending total

/-- A largest frequency of two gives two pairs or one pair and three kickers. -/
theorem head_two (rest : List Int) (positive : ∀ count ∈ (2 : Int) :: rest, 0 < count)
    (descending : ((2 : Int) :: rest).Pairwise (fun a b : Int => b ≤ a))
    (total : ((2 : Int) :: rest).sum = 5) :
    (2 : Int) :: rest = [2, 2, 1] ∨ (2 : Int) :: rest = [2, 1, 1, 1] := by
  simpa [profiles] using classify ((2 : Int) :: rest) positive descending total

/-- A largest frequency of one means all five ranks are distinct. -/
theorem head_one (rest : List Int) (positive : ∀ count ∈ (1 : Int) :: rest, 0 < count)
    (descending : ((1 : Int) :: rest).Pairwise (fun a b : Int => b ≤ a))
    (total : ((1 : Int) :: rest).sum = 5) : (1 : Int) :: rest = [1, 1, 1, 1, 1] := by
  simpa [profiles] using classify ((1 : Int) :: rest) positive descending total

end ProjectEulerSolutions.PokerCountPatterns
