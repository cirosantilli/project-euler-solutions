import ProjectEulerSolutions.Lib.RomanEncoding

namespace ProjectEulerSolutions.RomanNumerals
set_option Elab.async false

/-- Joining arbitrary string chunks emits their concatenated character lists. -/
theorem join_chunks (chunks : List String) :
    String.intercalate "" chunks = String.ofList (chunks.flatMap String.toList) := by
  have intercalate_empty (items : List (List Char)) : [].intercalate items = items.flatten := by
    induction items with
    | nil => rfl
    | cons head tail ih => cases tail <;> simp_all
  apply String.toList_injective
  simp [String.toList_intercalate, intercalate_empty, List.flatMap]

/-- Repeating a string is the same token repetition as the list-based encoding model. -/
theorem repeated_characters (token : String) (count : Nat) :
    (String.join (List.replicate count token)).toList = copies token.toList count := by
  simp [String.toList_join, List.flatMap, copies]

end ProjectEulerSolutions.RomanNumerals
