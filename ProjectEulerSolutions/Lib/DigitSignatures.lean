import ProjectEulerStatements.Lib.DecimalDigits
import lang.py.FormalSW.HeapCalls
import Mathlib.Data.List.Perm.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.String.Basic

/-! The source expression `"".join(sorted(str(n)))` preserves the decimal digit multiset. -/
namespace ProjectEulerSolutions.DigitSignatures
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

set_option Elab.async false

def characters (n : Int) : List String := (toString n).toList.map String.singleton
def ordered (n : Int) : List String := ScalarSort.strings (characters n)
def signature (n : Int) : String := String.intercalate "" (ordered n)
def values (n : Int) : List Mutable.Value := (ordered n).map (fun text => .scalar (.string text))

private theorem intercalate_empty (items : List (List α)) :
    [].intercalate items = items.flatten := by
  induction items with
  | nil => rfl
  | cons head tail ih => cases tail <;> simp_all

theorem signature_characters (n : Int) : (signature n).toList.Perm (toString n).toList := by
  have permutation := (ScalarSort.strings_perm (characters n)).flatMap_right String.toList
  simpa [signature, ordered, characters, String.toList_intercalate, List.flatMap_map,
    Function.comp_def, intercalate_empty, ← List.flatMap_def] using permutation

private theorem decimal_digits (n : Nat) (positive : 0 < n) :
    (toString (n : Int)).toList.map (fun c => c.toNat - '0'.toNat) = (Nat.digits 10 n).reverse := by
  have text : (toString (n : Int)).toList = (Nat.digits 10 n).reverse.map Nat.digitChar := by
    simpa [toString, Int.repr, Nat.repr] using
      ProjectEulerStatements.DecimalDigits.toDigits_eq_digitChars n positive
  rw [text, List.map_map]
  calc
    _ = (Nat.digits 10 n).reverse.map (fun digit => digit) := by
      apply List.map_congr_left
      intro digit member
      exact ProjectEulerStatements.DecimalDigits.digitChar_val digit
        (Nat.digits_lt_base (by decide) (by simpa using member))
    _ = _ := List.map_id' _

/-- Equal source keys imply equal digit multisets; sorting and joining lose no digits. -/
theorem equal_implies_permutation (a b : Nat) (aPositive : 0 < a) (bPositive : 0 < b)
    (equal : signature a = signature b) : (Nat.digits 10 a).Perm (Nat.digits 10 b) := by
  have characters : (toString (a : Int)).toList.Perm (toString (b : Int)).toList :=
    (signature_characters a).symm.trans ((by rw [equal] :
      (signature (a : Int)).toList.Perm (signature (b : Int)).toList).trans (signature_characters b))
  have digits := characters.map (fun c => c.toNat - '0'.toNat)
  rw [decimal_digits a aPositive, decimal_digits b bPositive] at digits
  exact (List.reverse_perm _).symm.trans (digits.trans (List.reverse_perm _))

/-- Mathlib replaces the core string order; the interpreter retains the original instance. -/
private theorem comparison (a b : String) : @LT.lt String String.instLT a b ↔ a < b :=
  String.lt_iff.trans ((List.lt_iff_lex_lt _ _).trans String.lt_iff_toList_lt.symm)

private theorem insert_ordered (value : String) (items : List String)
    (ordered : items.Pairwise (· ≤ ·)) : (ScalarSort.insertString value items).Pairwise (· ≤ ·) := by
  induction items with
  | nil => simp [ScalarSort.insertString]
  | cons first rest ih =>
      obtain ⟨lower, tailOrdered⟩ := List.pairwise_cons.mp ordered
      by_cases smaller : first < value
      · simp only [ScalarSort.insertString, if_pos ((comparison first value).mpr smaller), List.pairwise_cons]
        refine ⟨?_, ih tailOrdered⟩
        intro item member
        have member := (ScalarSort.insertString_perm value rest).mem_iff.mp member
        rcases List.mem_cons.mp member with rfl | member
        · exact le_of_lt smaller
        · exact lower item member
      · simp only [ScalarSort.insertString, if_neg ((comparison first value).not.mpr smaller), List.pairwise_cons]
        refine ⟨?_, lower, tailOrdered⟩
        intro item member
        rcases List.mem_cons.mp member with rfl | member
        · exact le_of_not_gt smaller
        · exact le_trans (le_of_not_gt smaller) (lower item member)

private theorem strings_ordered (items : List String) : (ScalarSort.strings items).Pairwise (· ≤ ·) := by
  induction items with
  | nil => exact .nil
  | cons first rest ih => exact insert_ordered first (ScalarSort.strings rest) ih

/-- Permuting the input characters does not change the stable sort's joined text. -/
theorem of_characters_permutation (a b : Int) (permutation : (characters a).Perm (characters b)) :
    signature a = signature b := by
  have same : ordered a = ordered b :=
    List.Perm.eq_of_pairwise (fun _ _ _ _ lower upper => le_antisymm lower upper)
      (strings_ordered _) (strings_ordered _)
      ((ScalarSort.strings_perm _).trans (permutation.trans (ScalarSort.strings_perm _).symm))
  exact congrArg (String.intercalate "") same

/-- The same decimal digit multiset produces the same key in the original Python expression. -/
theorem of_digits_permutation (a b : Nat) (aPositive : 0 < a) (bPositive : 0 < b)
    (permutation : (Nat.digits 10 a).Perm (Nat.digits 10 b)) : signature a = signature b := by
  have text (n : Nat) (positive : 0 < n) :
      characters n = (Nat.digits 10 n).reverse.map (fun d => String.singleton (Nat.digitChar d)) := by
    have digits : (toString (n : Int)).toList = (Nat.digits 10 n).reverse.map Nat.digitChar := by
      simpa [toString, Int.repr, Nat.repr] using
        ProjectEulerStatements.DecimalDigits.toDigits_eq_digitChars n positive
    simp only [characters, digits, List.map_map, Function.comp_def]
  apply of_characters_permutation
  rw [text a aPositive, text b bPositive]
  exact ((List.reverse_perm _).trans (permutation.trans (List.reverse_perm _).symm)).map _

theorem equal_iff_permutation (a b : Nat) (aPositive : 0 < a) (bPositive : 0 < b) :
    signature a = signature b ↔ (Nat.digits 10 a).Perm (Nat.digits 10 b) :=
  ⟨equal_implies_permutation a b aPositive bPositive, of_digits_permutation a b aPositive bPositive⟩

theorem winner_signatures : signature 2969 = signature 6299 ∧ signature 2969 = signature 9629 := by
  constructor
  · exact of_digits_permutation 2969 6299 (by decide) (by decide) (by decide +kernel)
  · exact of_digits_permutation 2969 9629 (by decide) (by decide) (by decide +kernel)

def expression (name : String) : Expr :=
  .call (.attribute (.constant (.string "")) "join")
    [.builtinCall .sorted [.builtinCall .str [.name name]]]

private theorem sorted_string (heap : Mutable.Heap) (n : Int) :
    primitive (.builtin .sorted) [.scalar (.string (toString n))] heap =
      some (.ok (.ref heap.length), heap ++ [.list (values n)]) := by
  have sorted : HeapSort.sort heap
      ((toString n).toList.map (fun c => Mutable.Value.scalar (.string (String.singleton c)))) =
      some (.ok (values n)) := by
    have result := HeapSort.sort_scalars heap ((characters n).map Constant.string)
    rw [ScalarSort.sort_strings] at result
    simpa only [Except.map, List.map_map, Function.comp_def,
      characters, values, ordered] using result
  simp only [toString] at sorted
  simp [primitive, Mutable.sequence, sorted, Mutable.allocate, Except.map]

/-- The unchanged nested calls allocate exactly the sorted character list. -/
theorem evaluate (program : Module) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (name : String) (n : Int) (lookup : env name = some (.scalar (.int n)))
    (globals : Mutable.Globals := []) :
    Evaluates (globals := globals) program env (expression name) heap (.scalar (.string (signature n)))
      (heap ++ [.list (values n)]) := by
  have text : Evaluates (globals := globals) program env (.builtinCall .str [.name name]) heap
      (.scalar (.string (toString n))) heap :=
    Evaluates.builtin (.cons (Evaluates.lookup lookup) .nil) rfl
  have sorted : Evaluates (globals := globals) program env
      (.builtinCall .sorted [.builtinCall .str [.name name]]) heap
      (.ref heap.length) (heap ++ [.list (values n)]) :=
    Evaluates.builtin (.cons text .nil) (sorted_string heap n)
  apply Evaluates.join Evaluates.constant (.cons sorted .nil)
  · exact primitive_join_list _ heap.length "" (ordered n) (by simp [values])
  · intro name impossible
    cases impossible

end ProjectEulerSolutions.DigitSignatures
