import ProjectEulerSolutions.Lib.RomanNumerals

namespace ProjectEulerSolutions.RomanNumerals
set_option Elab.async false

/-- A final unpaired symbol contributes its full value. -/
theorem parse_drop_single (input : List String) (index : Nat) (inside : index < input.length)
    (last : input.length ≤ index + 1) :
    parse (input.drop index) = value (input[index]) := by
  rw [List.drop_eq_getElem_cons inside, List.drop_eq_nil_of_le last]
  rfl

/-- An indexed scan has the same one- or two-symbol step as recursive parsing. -/
theorem parse_drop_pair (input : List String) (index : Nat) (inside : index + 1 < input.length) :
    parse (input.drop index) =
      if value (input[index]'(by omega)) < value (input[index + 1]) then
        value (input[index + 1]) - value (input[index]'(by omega)) + parse (input.drop (index + 2))
      else value (input[index]'(by omega)) + parse (input.drop (index + 1)) := by
  rw [List.drop_eq_getElem_cons (show index < input.length by omega), List.drop_eq_getElem_cons inside]
  rfl

end ProjectEulerSolutions.RomanNumerals
