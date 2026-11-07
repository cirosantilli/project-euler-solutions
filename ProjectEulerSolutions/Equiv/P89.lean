import ProjectEulerSolutions.P89
namespace ProjectEulerSolutions.P89
open ProjectEulerStatements.P89 ProjectEulerSolutions.RomanNumerals

private theorem parse_eq (xs : List Char) (total : Nat) :
    parseAcc xs total = total + ProjectEulerStatements.P89.romanToInt xs := by
  match xs with
  | [] => simp [parseAcc, ProjectEulerStatements.P89.romanToInt]
  | [x] => rfl
  | x :: y :: ys =>
    simp only [parseAcc, ProjectEulerStatements.P89.romanToInt, valueOf]
    change (if value x < value y then _ else _) = _
    split
    · rw [parse_eq ys]
      simp only [Nat.add_assoc]
      rfl
    · rw [parse_eq (y :: ys)]
      simp only [Nat.add_assoc]
      rfl
termination_by xs.length

private theorem encode_eq (n : Nat) (table : List (Nat × List Char)) :
    encode table n = intToRomanAux.go n table := by
  rw [intToRomanAux.go.eq_def]
  split
  · simp [encode]
  · rename_i v cs tl
    split
    · rename_i zero
      subst v
      simpa [encode, copies] using encode_eq n tl
    · rename_i nonzero
      split
      · rename_i ge
        rw [encode_subtract v n cs tl (by omega) ge, encode_eq (n-v) ((v,cs)::tl)]
      · rename_i lt
        rw [encode_skip v n cs tl (by omega), encode_eq n tl]
termination_by n + table.length
decreasing_by all_goals simp_all <;> omega

theorem equiv (lines : List String) : ProjectEulerStatements.P89.naive lines = solve lines := by
  have parsed (s : String) : romanToInt s = ProjectEulerStatements.P89.romanToInt s.toList := by
    simpa [romanToInt] using parse_eq s.toList 0
  have encoded (n : Nat) : intToMinRoman n = String.ofList (intToRomanAux n) := by
    unfold intToMinRoman intToRomanAux
    rw [encode_eq]
    rfl
  simp only [ProjectEulerStatements.P89.naive, solve, totalCharactersSaved,
    savedChars, minimalRoman, parsed, encoded]
  rfl
end ProjectEulerSolutions.P89
