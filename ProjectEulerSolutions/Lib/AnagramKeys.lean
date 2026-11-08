import ProjectEulerStatements.Lib.StringSorting
import lang.py.FormalSW.HeapCalls

/-! The sorted-character list and joined key identify exactly the same anagram classes. -/
namespace ProjectEulerSolutions.AnagramKeys
open FormalSW.PyAst HeapCalls ProjectEulerStatements.StringSorting
set_option Elab.async false

def characters (word : String) : List String := word.toList.map String.singleton
def ordered (word : String) : List String := ScalarSort.strings (characters word)
def key (word : String) : String := String.intercalate "" (ordered word)
def values (word : String) : List Mutable.Value :=
  (ordered word).map fun letter => .scalar (.string letter)

private theorem intercalate_empty (items : List (List α)) :
    [].intercalate items = items.flatten := by
  induction items with
  | nil => rfl
  | cons head tail ih => cases tail <;> simp_all

theorem key_permutation (word : String) : (key word).toList.Perm word.toList := by
  have permutation := (ScalarSort.strings_perm (characters word)).flatMap_right String.toList
  simpa [key, ordered, characters, String.toList_intercalate, List.flatMap_map,
    Function.comp_def, intercalate_empty, ← List.flatMap_def] using permutation

theorem ordered_of_permutation (left right : String) (permutation : left.toList.Perm right.toList) :
    ordered left = ordered right :=
  List.Perm.eq_of_pairwise (fun _ _ _ _ lower upper => le_antisymm lower upper)
    (strings_ordered _) (strings_ordered _)
    ((ScalarSort.strings_perm _).trans
      ((permutation.map String.singleton).trans (ScalarSort.strings_perm _).symm))

theorem key_eq_iff_permutation (left right : String) :
    key left = key right ↔ left.toList.Perm right.toList := by
  constructor
  · intro equal
    exact (key_permutation left).symm.trans (equal ▸ key_permutation right)
  · intro permutation
    exact congrArg (String.intercalate "") (ordered_of_permutation left right permutation)

theorem ordered_eq_iff_key_eq (left right : String) : ordered left = ordered right ↔ key left = key right :=
  ⟨congrArg (String.intercalate ""), fun equal =>
    ordered_of_permutation left right ((key_eq_iff_permutation left right).mp equal)⟩

theorem key_length (word : String) : (key word).toList.length = word.toList.length :=
  (key_permutation word).length_eq

theorem distinct_nonempty (left right : String) (different : left ≠ right) (equal : key left = key right) :
    left.toList ≠ [] ∧ right.toList ≠ [] := by
  have permutation := (key_eq_iff_permutation left right).mp equal
  constructor
  · intro empty
    have other : right.toList = [] := by simpa [empty] using permutation.symm
    exact different (String.toList_inj.mp (empty.trans other.symm))
  · intro empty
    have other : left.toList = [] := by simpa [empty] using permutation
    exact different (String.toList_inj.mp (other.trans empty.symm))

theorem equal_elements (left right : List String) :
    equalElements (left.map fun text => .scalar (.string text))
      (right.map fun text => .scalar (.string text)) = some (decide (left = right)) := by
  induction left generalizing right with
  | nil => cases right <;> simp [equalElements]
  | cons first rest ih =>
      cases right with
      | nil => simp [equalElements]
      | cons other tail =>
          by_cases same : first = other
          · subst other
            simp [equalElements, scalarCompare, ih]
          · simp [equalElements, scalarCompare, same]

theorem equal_lists (left right : List String) :
    equalLists (left.map fun text => .scalar (.string text))
      (right.map fun text => .scalar (.string text)) = some (decide (left = right)) := by
  by_cases lengths : left.length = right.length
  · simp [equalLists, lengths, equal_elements]
  · have different : left ≠ right := fun same => lengths (congrArg List.length same)
    simp [equalLists, lengths, different]

theorem compare_sorted (heap : Mutable.Heap) (left right : Nat) (word other : String)
    (foundLeft : heap[left]? = some (.list (values word)))
    (foundRight : heap[right]? = some (.list (values other))) :
    HeapCalls.compare .eq (.ref left) (.ref right) heap =
      some (.ok (.scalar (.bool (decide (key word = key other)))), heap) := by
  by_cases same : left = right
  · subst right
    have equalValues : values word = values other := by simpa [foundLeft] using foundRight
    have equalOrdered : ordered word = ordered other := by
      apply (List.map_injective_iff.mpr (show Function.Injective (fun text => Mutable.Value.scalar (.string text)) by
        intro a b equal; simpa using equal))
      exact equalValues
    have equalKeys := (ordered_eq_iff_key_eq word other).mp equalOrdered
    simp [HeapCalls.compare, Mutable.array, foundLeft, equalKeys]
    rfl
  · simp [HeapCalls.compare, Mutable.array, foundLeft, foundRight, same, values,
      equal_lists, ordered_eq_iff_key_eq]
    rfl

/-- The builtin sort allocates its full character list and leaves caller objects intact. -/
theorem sorted_string (heap : Mutable.Heap) (word : String) :
    primitive (.builtin .sorted) [.scalar (.string word)] heap =
      some (.ok (.ref heap.length), heap ++ [.list (values word)]) := by
  have sorted : HeapSort.sort heap
      (word.toList.map fun c => Mutable.Value.scalar (.string (String.singleton c))) =
      some (.ok (values word)) := by
    have result := HeapSort.sort_scalars heap ((characters word).map Constant.string)
    rw [ScalarSort.sort_strings] at result
    simpa only [Except.map, List.map_map, Function.comp_def, characters, values, ordered] using result
  simp [primitive, Mutable.sequence, sorted, Mutable.allocate, Except.map]

theorem evaluate_sorted (program : Module) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (expression : Expr) (word : String) (globals : Mutable.Globals)
    (evaluated : Evaluates program env expression heap (.scalar (.string word)) heap globals) :
    Evaluates program env (.builtinCall .sorted [expression]) heap
      (.ref heap.length) (heap ++ [.list (values word)]) globals :=
  Evaluates.builtin (.cons evaluated .nil) (sorted_string heap word)

/-- The optimized joined key and the reference's sorted list retain exactly the same characters. -/
theorem evaluate_key (program : Module) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (expression : Expr) (word : String) (globals : Mutable.Globals)
    (evaluated : Evaluates program env expression heap (.scalar (.string word)) heap globals) :
    Evaluates program env (.call (.attribute (.constant (.string "")) "join")
      [.builtinCall .sorted [expression]]) heap (.scalar (.string (key word)))
      (heap ++ [.list (values word)]) globals := by
  apply Evaluates.join Evaluates.constant (.cons (evaluate_sorted program env heap expression word globals evaluated) .nil)
  · exact primitive_join_list _ heap.length "" (ordered word) (by simp [values])
  · intro name impossible
    cases impossible

end ProjectEulerSolutions.AnagramKeys
