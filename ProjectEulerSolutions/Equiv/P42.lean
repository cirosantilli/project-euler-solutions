import ProjectEulerSolutions.P42
namespace ProjectEulerSolutions.P42

private theorem value_lower (letters : List Char) (initial : Nat) :
    initial + letters.length ≤ letters.foldl (fun acc c => acc + (c.toNat - 'A'.toNat + 1)) initial := by
  induction letters generalizing initial with
  | nil => simp
  | cons head tail ih =>
    simp only [List.foldl_cons, List.length_cons]
    have := ih (initial + (head.toNat - 'A'.toNat + 1))
    omega

/-- Python accepts nonempty uppercase words; the proof needs only nonemptiness. -/
theorem equiv (words : List String) (nonempty : ∀ w ∈ words, w.toList ≠ []) :
    ProjectEulerStatements.P42.naive words = solve words := by
  have go (items : List String) (hne : ∀ w ∈ items, w.toList ≠ []) (initial : Nat) :
      items.foldl (fun acc w => if ProjectEulerStatements.P42.isTriangle
        (ProjectEulerStatements.P42.wordValue w) then acc+1 else acc) initial =
      items.foldl (fun acc w => if isTriangle (wordValue w) then acc+1 else acc) initial := by
    induction items generalizing initial with
    | nil => rfl
    | cons head tail ih =>
      have length := List.length_pos_iff.mpr (hne head (by simp))
      have lower := value_lower head.toList 0
      have positive : 0 < wordValue head := by unfold wordValue; omega
      have same := TriangularDiscriminant.correct (wordValue head) positive
      simp only [List.foldl_cons, isTriangle, same]
      exact ih (fun w h => hne w (by simp [h])) _
  exact go words nonempty 0
end ProjectEulerSolutions.P42
