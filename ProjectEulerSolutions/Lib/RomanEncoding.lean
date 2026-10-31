import ProjectEulerSolutions.Lib.RomanNumerals

/-! Encoding commutes with changing the representation of each emitted character. -/
namespace ProjectEulerSolutions.RomanNumerals
set_option Elab.async false

theorem copies_map (f : α → β) (token : List α) (count : Nat) :
    copies (token.map f) count = (copies token count).map f := by
  induction count with
  | zero => rfl
  | succ count ih => simp only [copies_succ, List.map_append, ih]

theorem encode_map (f : α → β) (table : List (Nat × List α)) (n : Nat) :
    encode (table.map (fun pair => (pair.1, pair.2.map f))) n = (encode table n).map f := by
  induction table generalizing n with
  | nil => rfl
  | cons head rest ih =>
      rcases head with ⟨denomination, token⟩
      simp only [List.map_cons, encode, copies_map, List.map_append, ih]

theorem join_characters (characters : List Char) :
    String.intercalate "" (characters.map String.singleton) = String.ofList characters := by
  have intercalate_empty (items : List (List Char)) : [].intercalate items = items.flatten := by
    induction items with
    | nil => rfl
    | cons head tail ih => cases tail <;> simp_all
  apply String.toList_injective
  simp [String.toList_intercalate, List.map_map, Function.comp_def, intercalate_empty]
  induction characters with
  | nil => rfl
  | cons head tail ih => simpa using congrArg (List.cons head) ih

end ProjectEulerSolutions.RomanNumerals
