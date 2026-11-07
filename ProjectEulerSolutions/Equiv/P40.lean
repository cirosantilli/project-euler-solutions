import ProjectEulerSolutions.P40
namespace ProjectEulerSolutions.P40

/-- Every finite reference stream reaching the requested positions gives the same digits. -/
theorem equiv (positions : List Nat) (limit : Nat)
    (positive : ∀ p ∈ positions, 0 < p) (bounded : ∀ p ∈ positions, p ≤ limit) :
    ProjectEulerStatements.P40.productDigits positions limit = solve positions := by
  unfold ProjectEulerStatements.P40.productDigits solve
  have aux (items : List Nat) (hp : ∀ p ∈ items, 0 < p)
      (hb : ∀ p ∈ items, p ≤ limit) (initial : Nat) :
      items.foldl (fun acc p => acc * ProjectEulerStatements.P40.digitAt p limit) initial =
      items.foldl (fun acc p => acc * digitAt p) initial := by
    induction items generalizing initial with
    | nil => rfl
    | cons head tail ih =>
      have same := ProjectEulerStatements.ChampernowneBlocks.blockDigit_correct head limit
        (hp head (by simp)) (hb head (by simp))
      simp only [List.foldl_cons, digitAt, same]
      exact ih (fun p h => hp p (by simp [h])) (fun p h => hb p (by simp [h])) _
  exact aux positions positive bounded 1
end ProjectEulerSolutions.P40
